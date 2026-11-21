import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from task2_cls.baselines import consistent_probabilities
from task2_cls.evaluate import OUTPUT_LABELS, validate_probabilities
from task2_cls.field_map import FIELD_MAP
from task2_cls.preprocessing import PREPROCESSING_VERSION, normalize_like_resumeatlas

BACKBONE = "jjzha/jobbert-base-cased"


def encode_head_tail(tokenizer, text, max_length=512):
    """Keep equal head/tail token budgets, including all required special tokens."""
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if not 1 <= budget or max_length > 512:
        raise ValueError("max_length must allow content and special tokens, at most 512")
    ids = tokenizer(normalize_like_resumeatlas(text), add_special_tokens=False,
                    truncation=False, verbose=False)["input_ids"]
    if len(ids) > budget:
        head = (budget + 1) // 2
        tail = budget - head
        ids = ids[:head] + (ids[-tail:] if tail else [])
    # JobBERT uses BERT single-sequence layout. Build it explicitly to support
    # Transformers 4 and 5 (prepare_for_model was removed from fast tokenizers).
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None or tokenizer.num_special_tokens_to_add(pair=False) != 2:
        raise ValueError("Expected a BERT tokenizer with CLS/SEP special tokens")
    ids = [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "token_type_ids": [0] * len(ids)}


class JobBertMultiOutput(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(encoder.config.hidden_dropout_prob)
        self.heads = nn.ModuleDict({key: nn.Linear(encoder.config.hidden_size, len(labels))
                                    for key, labels in OUTPUT_LABELS.items()})

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        hidden = self.dropout(self.encoder(**inputs).last_hidden_state[:, 0])
        logits = {key: head(hidden) for key, head in self.heads.items()}
        loss = None
        if labels is not None:
            losses = []
            for key, values in logits.items():
                valid = labels[key].ne(-100)
                # Avoid mean CE over an empty set (NaN), retaining a zero gradient.
                losses.append(nn.functional.cross_entropy(values[valid], labels[key][valid])
                              if valid.any() else values.sum() * 0)
            loss = sum(losses)
        return {"logits": logits, "loss": loss}


def save_checkpoint(model, tokenizer, metadata, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(directory / "encoder_config")
    tokenizer.save_pretrained(directory)
    torch.save(model.state_dict(), directory / "weights.pt")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


class JobBertPredictor:
    """Offline checkpoint inference, including saved preprocessing/output policy."""

    def __init__(self, directory, *, device="cpu"):
        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if (self.metadata["preprocessing_version"] != PREPROCESSING_VERSION
                or self.metadata["field_map"] != FIELD_MAP
                or self.metadata["labels"] != {k: list(v) for k, v in OUTPUT_LABELS.items()}):
            raise ValueError("Checkpoint preprocessing/taxonomy differs from runtime")
        self.mode = self.metadata["field_policy"]
        if self.mode not in ("derived_field", "hierarchical"):
            raise ValueError("Unknown checkpoint Field policy")
        self.field_map = self.metadata["field_map"]
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        config = AutoConfig.from_pretrained(directory / "encoder_config", local_files_only=True)
        self.model = JobBertMultiOutput(AutoModel.from_config(config))
        self.model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()

    def predict_proba(self, texts, *, batch_size=16, consistent=True):
        if isinstance(texts, str) or not texts or any(not isinstance(t, str) for t in texts) or batch_size < 1:
            raise ValueError("Pass a non-empty list of texts and positive batch_size")
        rows = {key: [] for key in OUTPUT_LABELS}
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                encoded = [encode_head_tail(self.tokenizer, t, self.metadata["max_length"])
                           for t in texts[start:start + batch_size]]
                batch = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
                logits = self.model(**{k: v.to(self.device) for k, v in batch.items()})["logits"]
                for key, value in logits.items():
                    rows[key].append(value.softmax(-1).cpu().numpy())
        probs = {key: np.concatenate(values) for key, values in rows.items()}
        if consistent:
            probs = consistent_probabilities(probs, self.mode, self.field_map)
        validate_probabilities(probs, len(texts))
        return probs
