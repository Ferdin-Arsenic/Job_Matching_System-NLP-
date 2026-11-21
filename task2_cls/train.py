import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BertConfig, BertTokenizerFast, get_linear_schedule_with_warmup, set_seed

from task2_cls.baselines import consistent_probabilities
from task2_cls.data import ROOT, SEED, load_prepared, split_frames
from task2_cls.evaluate import OUTPUT_LABELS, confusion_tables, evaluate, predictions_from_probs
from task2_cls.field_map import FIELD_MAP
from task2_cls.model import BACKBONE, JobBertMultiOutput, JobBertPredictor, encode_head_tail, save_checkpoint
from task2_cls.preprocessing import LEVEL_RULE_VERSION, PREPROCESSING_VERSION


def encode_frame(frame, tokenizer, max_length, input_mode):
    column = "text" if input_mode == "full_text" else "entity_text"
    rows = []
    for row in frame.to_dict("records"):
        labels = {key: OUTPUT_LABELS[key].index(row[key]) if row[key] != "Unknown" else -100
                  for key in OUTPUT_LABELS}
        rows.append({"tokens": encode_head_tail(tokenizer, row[column], max_length), "labels": labels})
    return rows


def make_loader(rows, tokenizer, batch_size, *, shuffle=False):
    def collate(batch):
        tokens = tokenizer.pad([r["tokens"] for r in batch], padding=True, return_tensors="pt")
        tokens["labels"] = {k: torch.tensor([r["labels"][k] for r in batch]) for k in OUTPUT_LABELS}
        return tokens
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=torch.Generator().manual_seed(SEED))


def move_batch(batch, device):
    return {key: {k: v.to(device) for k, v in value.items()} if isinstance(value, dict) else value.to(device)
            for key, value in batch.items()}


def predict(model, loader, device):
    model.eval()
    rows = {key: [] for key in OUTPUT_LABELS}
    with torch.inference_mode():
        for batch in loader:
            logits = model(**move_batch(batch, device))["logits"]
            for key, values in logits.items():
                rows[key].append(values.softmax(-1).cpu().numpy())
    return {k: np.concatenate(v) for k, v in rows.items()}


def smoke_components(directory, max_length):
    """Explicit random tiny BERT fixture; never a JobBERT experiment result."""
    vocab = directory / "smoke_vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                                 "python", "developer", "experience", "years", "professional"]) + "\n", encoding="utf-8")
    vocab_key = "vocab" if "vocab" in inspect.signature(BertTokenizerFast.__init__).parameters else "vocab_file"
    tokenizer = BertTokenizerFast(**{vocab_key: str(vocab)}, do_lower_case=False)
    config = BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=max_length)
    return tokenizer, lambda: JobBertMultiOutput(AutoModel.from_config(config))


def run(root=ROOT, *, output=None, learning_rates=(2e-5, 3e-5, 5e-5), epochs=5,
        batch_size=16, max_length=512, sample=0, smoke=False, input_mode="full_text",
        field_policy="derived_field", device=None, threads=4):
    root = Path(root)
    output = Path(output) if output else root / "models/cls" / ("jobbert_smoke" if smoke else "jobbert")
    if epochs < 1 or batch_size < 1 or sample < 0 or threads < 1 or not 3 <= max_length <= 512:
        raise ValueError("Invalid training dimensions")
    if not learning_rates or any(lr <= 0 or not math.isfinite(lr) for lr in learning_rates):
        raise ValueError("Learning rates must be finite and positive")
    if input_mode not in ("full_text", "entity_only") or field_policy not in ("derived_field", "hierarchical"):
        raise ValueError("Invalid input or Field policy")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    data, splits, dataset_metadata = load_prepared(root)
    parts = split_frames(data, splits)
    if smoke:
        sample, epochs, learning_rates, max_length, batch_size = sample or 16, 1, (5e-4,), 64, 4
    if sample:
        parts = {name: part.sample(n=min(sample, len(part)), random_state=SEED).reset_index(drop=True)
                 for name, part in parts.items()}
    if not parts["val"].level.ne("Unknown").any():
        raise ValueError("Validation sample needs at least one known Level; increase --sample")
    torch.set_num_threads(threads)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(SEED)
    output.mkdir(parents=True, exist_ok=True)
    if smoke:
        tokenizer, factory = smoke_components(output, max_length)
    else:
        tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
        factory = lambda: JobBertMultiOutput(AutoModel.from_pretrained(BACKBONE))
    # Test tokenization/inference is deferred until all model selection finishes.
    encoded = {name: encode_frame(parts[name], tokenizer, max_length, input_mode) for name in ("train", "val")}
    status = "smoke (random tiny BERT)" if smoke else "sample experiment" if sample else "trained"
    metadata = {"architecture": "jobbert_multi_output", "status": status, "backbone": BACKBONE if not smoke else "random-tiny-bert",
                "labels": {k: list(v) for k, v in OUTPUT_LABELS.items()}, "field_map": FIELD_MAP,
                "preprocessing_version": PREPROCESSING_VERSION, "level_rule_version": LEVEL_RULE_VERSION,
                "max_length": max_length, "truncation": "head_tail_equal_budget_including_special_tokens",
                "input_mode": input_mode, "field_policy": field_policy, "level_source": "model",
                "unknown_level": "-100; excluded from Level loss and metrics", "loss": "sum of three mean cross-entropies",
                "seed": SEED, "dataset_fingerprint": dataset_metadata["fingerprint"],
                "split_ids": {name: part.text_md5.tolist() for name, part in parts.items()},
                "batch_size": batch_size, "epochs": epochs, "learning_rates": list(learning_rates),
                "optimizer": "AdamW", "weight_decay": 0.01, "warmup_ratio": 0.1,
                "precision": "float16 autocast with gradient scaling" if device.type == "cuda" else "float32",
                "selection": "mean output validation Macro-F1; earlier epoch/lower LR wins ties",
                "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "numpy")},
                "implementation_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                          for name in ("model.py", "train.py", "evaluate.py", "preprocessing.py")}}
    best, history = -1.0, []
    for lr in sorted(set(learning_rates)):
        set_seed(SEED)
        model = factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        train_loader = make_loader(encoded["train"], tokenizer, batch_size, shuffle=True)
        val_loader = make_loader(encoded["val"], tokenizer, batch_size)
        steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    loss = model(**move_batch(batch, device))["loss"]
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                total_loss += loss.item()
            raw = predict(model, val_loader, device)
            probs = consistent_probabilities(raw, field_policy, FIELD_MAP)
            metrics = evaluate(parts["val"], probs)
            score = float(metrics.loc[metrics.metric.eq("macro_f1"), "value"].mean())
            history.append(dict(lr=lr, epoch=epoch, train_loss=total_loss / len(train_loader), validation_macro_f1=score))
            print(f"{status} | lr={lr:g} epoch={epoch} validation mean Macro-F1={score:.4f}", flush=True)
            if score > best:
                best = score
                save_checkpoint(model, tokenizer, {**metadata, "selected_lr": lr, "selected_epoch": epoch,
                                                   "validation_macro_f1": score}, output)
        del optimizer, scheduler, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(history).to_csv(output / "tuning.csv", index=False)
    predictor = JobBertPredictor(output, device=str(device))
    tables, predictions = [], []
    for split in ("val", "test"):
        part = parts[split]
        primary = "text" if input_mode == "full_text" else "entity_text"
        raw = predictor.predict_proba(part[primary].tolist(), batch_size=batch_size, consistent=False)
        for policy in ("direct_heads", field_policy):
            probs = raw if policy == "direct_heads" else consistent_probabilities(raw, field_policy, FIELD_MAP)
            tables.append(evaluate(part, probs).assign(split=split, policy=policy, input_mode=input_mode, status=status))
            pred = part[["text_md5", "field", "sub_field", "level"]].copy()
            pred["split"], pred["policy"] = split, policy
            for key, values in predictions_from_probs(probs).items():
                pred[f"pred_{key}"] = values
                pred[f"probs_{key}"] = [row.tolist() for row in probs[key]]
            predictions.append(pred)
            for key, table in confusion_tables(part, probs).items():
                table.to_csv(output / f"confusion_{policy}_{split}_{key}.csv")
        # Input substitution at inference, explicitly not a retrained ablation.
        alternate = "entity_text" if primary == "text" else "text"
        alternate_probs = predictor.predict_proba(part[alternate].tolist(), batch_size=batch_size)
        tables.append(evaluate(part, alternate_probs).assign(split=split, policy=field_policy,
                      input_mode=f"{alternate} (inference substitution)", status=status))
    results = pd.concat(tables, ignore_index=True)
    results.to_csv(output / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(output / "predictions.parquet", index=False)
    print(f"Saved {status} artifacts: {output}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[2e-5, 3e-5, 5e-5])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Offline random tiny BERT on real data; not JobBERT results")
    parser.add_argument("--input-mode", choices=["full_text", "entity_only"], default="full_text")
    parser.add_argument("--field-policy", choices=["derived_field", "hierarchical"], default="derived_field")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
