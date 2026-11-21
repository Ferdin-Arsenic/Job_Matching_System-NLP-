"""Cross-Encoder JobBERT (jjzha/jobbert-base-cased). BUTUH GPU (Colab T4) untuk run penuh.

Input pasangan: [CLS] job [SEP] resume [SEP], anggaran job <= 190 + resume <= 319 wordpiece (+3 special = 512).
Varian ablation (split, hyperparameter, seed sama):
  V1 = raw text | V2 = entity Task 1 di depan raw text | V3 = V2 + profil Task 2 di depan resume
Dua run terpisah: reg (num_labels=1, target ats/100, MSE) dan cls (num_labels=3, cross-entropy).
"""
import argparse, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task3_match.common import DATA, ROOT, evaluate, get_splits, load
from common.schemas import validate_profile

BACKBONE, JOB_B, RES_B, SEED = "jjzha/jobbert-base-cased", 190, 319, 42
ENT = [("SKILL", "skills"), ("JOB_TITLE", "titles"), ("EDUCATION", "education"), ("EXPERIENCE", "experience")]


def ent_str(row):
    return " | ".join(f"{name}: {', '.join(row[c])}" for c, name in ENT if len(row[c]))


def build_texts(df, variant, profiles="profiles_stub.parquet"):
    """Teks segmen job & resume per varian. Entity/profil di depan supaya tidak terpotong; raw text mengisi sisa anggaran."""
    job, res = df.job.tolist(), df.resume.tolist()
    if variant == "V1": return job, res
    ner = ROOT / "outputs" / "ner"
    je = pd.read_parquet(ner / "ner_ats_job.parquet").set_index("text_md5").apply(ent_str, axis=1)
    re_ = pd.read_parquet(ner / "ner_ats_resume.parquet").set_index("text_md5").apply(ent_str, axis=1)
    job = [f"{je[j]} || {t}" for j, t in zip(df.job_id, job)]
    pre = re_.reindex(df.resume_id).values
    if variant == "V3":
        p = pd.read_parquet(DATA / "cache" / profiles)
        if not {"resume_id", "field", "sub_field", "level"} <= set(p) or p.resume_id.duplicated().any():
            raise ValueError("Profiles need complete columns and unique resume_id")
        if not set(df.resume_id) <= set(p.resume_id):
            raise ValueError("Missing Task 2 profiles for matching resumes")
        for row in p.to_dict("records"):
            validate_profile({**row, "field_probs": {}, "sub_probs": {}})
        p = p.set_index("resume_id").reindex(df.resume_id)
        pre = [f"field: {f} | sub-field: {s} | level: {l} | {e}" for f, s, l, e in zip(p.field, p.sub_field, p.level, pre)]
    return job, [f"{e} || {t}" for e, t in zip(pre, res)]


def encode(tk, job, res, labels):
    out = []
    for a, b, y in zip(job, res, labels):
        ia = tk(a, add_special_tokens=False, truncation=True, max_length=JOB_B)["input_ids"]
        ib = tk(b, add_special_tokens=False, truncation=True, max_length=RES_B)["input_ids"]
        ids = [tk.cls_token_id, *ia, tk.sep_token_id, *ib, tk.sep_token_id]      # format pasangan BERT
        out.append({"input_ids": ids, "token_type_ids": [0] * (len(ia) + 2) + [1] * (len(ib) + 1),
                    "attention_mask": [1] * len(ids), "labels": y})
    return out


tier_of = lambda s: np.digitize(s, [40, 70])                       # band ATS: <40 No Fit, 40-70 Potential, >70 Good


def metrics_fn(run):
    def f(p):
        pred, y = p.predictions, p.label_ids
        if run == "cls": return {"macro_f1": f1_score(y, pred.argmax(1), average="macro")}
        s, t = pred.reshape(-1) * 100, y * 100
        return {"mae": float(np.abs(s - t).mean()), "macro_f1": f1_score(tier_of(t), tier_of(s), average="macro")}
    return f


if __name__ == "__main__":
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["V1", "V2", "V3"], required=True)
    ap.add_argument("--run", choices=["reg", "cls"], required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--sample", type=int, default=0, help="SAMPLE_N per split untuk tes CPU (0 = penuh)")
    ap.add_argument("--profiles", default="profiles_stub.parquet", help="file di data/cache; ganti ke output Task 2 bila sudah ada")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--backbone", default=BACKBONE, help="hanya untuk smoke test CPU, mis. google/bert_uncased_L-2_H-128_A-2")
    a = ap.parse_args()
    stub = a.variant == "V3" and "stub" in a.profiles
    tag = f"{a.variant}{'-stub' if stub else ''}_{a.run}_lr{a.lr:g}" + (f"_sample{a.sample}" if a.sample else "")
    set_seed(SEED)

    df = load(); sp = get_splits(df)[SEED]
    idx = {k: np.array(v[:a.sample] if a.sample else v) for k, v in sp.items()}
    tk = AutoTokenizer.from_pretrained(BACKBONE)   # tokenizer selalu JobBERT (vocab 28996 muat di model tiny 30522)
    job, res = build_texts(df, a.variant, profiles=a.profiles)
    y = (df.ats_score / 100).astype("float32").tolist() if a.run == "reg" else df.tier.tolist()
    ds = {k: encode(tk, [job[i] for i in v], [res[i] for i in v], [y[i] for i in v]) for k, v in idx.items()}
    print(tag, {k: len(v) for k, v in ds.items()}, "| contoh input:", tk.decode(ds["train"][0]["input_ids"])[:400])

    model = AutoModelForSequenceClassification.from_pretrained(a.backbone, num_labels=1 if a.run == "reg" else 3)
    steps = -(-len(ds["train"]) // a.bs) * a.epochs
    out_dir = ROOT / "models" / "match" / "_ckpt" / tag
    args = TrainingArguments(output_dir=str(out_dir), eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
                             load_best_model_at_end=True, metric_for_best_model="mae" if a.run == "reg" else "macro_f1",
                             greater_is_better=a.run == "cls", learning_rate=a.lr, num_train_epochs=a.epochs,
                             per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs * 2, warmup_steps=int(0.1 * steps),
                             weight_decay=0.01, fp16=torch.cuda.is_available(), seed=SEED, report_to="none", logging_steps=50)
    tr = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"], data_collator=DataCollatorWithPadding(tk),
                 compute_metrics=metrics_fn(a.run), callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])
    tr.train()

    # Prediksi -> (prob tier, skor 0-100). Kalibrator dilatih di val (bukan test), satu fitur.
    pv, pt = (tr.predict(ds[k]).predictions for k in ("val", "test"))
    dv, dt = df.iloc[idx["val"]], df.iloc[idx["test"]]
    if a.run == "reg":
        sv, st = pv.reshape(-1) * 100, pt.reshape(-1) * 100
        lr = LogisticRegression(max_iter=1000).fit(sv[:, None], dv.tier)
        prv, prt = lr.predict_proba(sv[:, None]), lr.predict_proba(st[:, None])
        thr = {k: np.eye(3)[tier_of(s)] for k, s in (("val", sv), ("test", st))}   # tier dari threshold band (pembanding gratis)
    else:
        prv, prt = (torch.tensor(p).softmax(-1).numpy() for p in (pv, pt))
        ev, et = prv @ [0, 1, 2], prt @ [0, 1, 2]
        iso = IsotonicRegression(out_of_bounds="clip").fit(ev, dv.ats_score)
        sv, st = iso.predict(ev), iso.predict(et)
    rows = []
    for part, d, pr, s in (("val", dv, prv, sv), ("test", dt, prt, st)):
        r = evaluate(d, pr, s)
        if a.run == "reg": r["macro_F1_threshold"] = f1_score(d.tier, thr[part].argmax(1), average="macro")
        rows += [dict(model=f"Cross-Encoder {a.variant}{' (stub)' if stub else ''} {a.run}", variant=a.variant, run=a.run, lr=a.lr,
                      best_ckpt=Path(tr.state.best_model_checkpoint or "").name,
                      split=part, metric=k, value=v) for k, v in r.items()]
    (DATA / "results").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(DATA / "results" / f"task3_ce_{tag}.csv", index=False)
    pd.DataFrame({"row": np.r_[idx["val"], idx["test"]], "split": ["val"] * len(dv) + ["test"] * len(dt),
                  "score": np.r_[sv, st], **{f"p{k}": np.r_[prv[:, k], prt[:, k]] for k in range(3)}}
                 ).to_parquet(DATA / "results" / f"task3_pred_{tag}.parquet", index=False)
    if not a.no_save and not a.sample:
        tr.save_model(str(ROOT / "models" / "match" / tag)); tk.save_pretrained(str(ROOT / "models" / "match" / tag))
    shutil.rmtree(out_dir, ignore_errors=True)                     # checkpoint per epoch (+optimizer) ~2,6 GB per run
    print(pd.DataFrame(rows).query("split == 'test'").set_index("metric").value.round(3).to_dict())
