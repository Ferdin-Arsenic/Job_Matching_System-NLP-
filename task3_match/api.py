"""Task 3 API for the CLI: score_match() and rank().

Model: Cross-Encoder V3 (jjzha/jobbert-base-cased fine-tuned, 3-class) = raw text + Task 1 entities + Task 2 profile,
the best ablation variant in task3_match/analysis.py. The input is built by the SAME functions as training
(task3_match/cross_encoder.py), so CLI inputs match what the model saw.

Weights: models/match/V3_cls_lr2e-05/ (Colab output, not in git). If missing, they are downloaded from the
Hugging Face repo in env T3_MATCH_REPO; with neither, a clear error is raised (no silent fallback).

score = isotonic(E[tier]) fitted on the VAL rows of the training run (same as training), tier = argmax of the
softmax probabilities. rank() orders by E[tier], which is continuous; the isotonic score is a step function with many ties.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from common.schemas import FIT_TIERS, validate_match
from task3_match.common import DATA, ROOT
from task3_match.cross_encoder import encode, ent_str

TAG = "V3_cls_lr2e-05"
MODEL_DIR = ROOT / "models" / "match" / TAG
HF_REPO = os.environ.get("T3_MATCH_REPO")
MODEL_NAME = "Cross-Encoder V3 (JobBERT; raw text + Task 1 entities + Task 2 profile)"
BATCH = 8


def _model_dir() -> Path:
    if not (MODEL_DIR / "model.safetensors").exists():
        if not HF_REPO:
            raise FileNotFoundError(f"Task 3 weights missing: {MODEL_DIR}. Copy the folder from the Colab output "
                                    "(MyDrive/task3_match/models/V3_cls_lr2e-05) or set T3_MATCH_REPO to a Hugging Face repo.")
        from huggingface_hub import snapshot_download
        snapshot_download(HF_REPO, local_dir=MODEL_DIR, allow_patterns=["*.json", "*.safetensors"])
    return MODEL_DIR


@lru_cache(maxsize=1)
def _model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = _model_dir()
    return AutoTokenizer.from_pretrained(path), AutoModelForSequenceClassification.from_pretrained(path).eval()


@lru_cache(maxsize=1)
def _isotonic() -> IsotonicRegression:
    """E[tier] -> ATS score, fitted on val rows of the saved training predictions (ats_score from outputs/ner/ats_pairs.parquet)."""
    p = pd.read_parquet(DATA / "results" / f"task3_pred_{TAG}.parquet").query("split == 'val'")
    ats = pd.read_parquet(ROOT / "outputs" / "ner" / "ats_pairs.parquet").ats_score.values[p.row.values]
    return IsotonicRegression(out_of_bounds="clip").fit(p[["p0", "p1", "p2"]].values @ [0, 1, 2], ats)


def _texts(cv_text, cv_entities, cv_profile, job_text, job_entities):
    job = f"{ent_str(job_entities)} || {job_text}"
    cv = (f"field: {cv_profile['field']} | sub-field: {cv_profile['sub_field']} | level: {cv_profile['level']} | "
          f"{ent_str(cv_entities)} || {cv_text}")
    return job, cv


def _probs(jobs: list[str], cvs: list[str]) -> np.ndarray:
    import torch
    from transformers import DataCollatorWithPadding
    tk, model = _model()
    data, collate, out = encode(tk, jobs, cvs, [0] * len(jobs)), DataCollatorWithPadding(tk), []
    with torch.no_grad():
        for i in range(0, len(data), BATCH):
            batch = collate(data[i:i + BATCH]); batch.pop("labels")
            out.append(torch.softmax(model(**batch).logits.float(), -1).numpy())
    return np.concatenate(out)


def _to_match(probs: np.ndarray) -> dict:
    expected = float(probs @ [0, 1, 2])
    match = {"score": float(np.clip(_isotonic().predict([expected])[0], 0, 100)),
             "tier": FIT_TIERS[int(probs.argmax())],
             "probs": {t: float(p) for t, p in zip(FIT_TIERS, probs)},
             "expected_tier": expected}
    validate_match(match)
    return match


def score_match(cv_text, cv_entities, cv_profile, job_text, job_entities) -> dict:
    job, cv = _texts(cv_text, cv_entities, cv_profile, job_text, job_entities)
    return _to_match(_probs([job], [cv])[0])


def rank(cvs: list[dict], job: dict) -> list[dict]:
    """cvs: [{"name", "text", "entities", "profile"}]; job: {"text", "entities"}. Sorted by E[tier], descending."""
    pairs = [_texts(c["text"], c["entities"], c["profile"], job["text"], job["entities"]) for c in cvs]
    probs = _probs([p[0] for p in pairs], [p[1] for p in pairs])
    out = [{**c, "match": _to_match(p)} for c, p in zip(cvs, probs)]
    return sorted(out, key=lambda r: r["match"]["expected_tier"], reverse=True)
