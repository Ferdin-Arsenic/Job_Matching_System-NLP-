"""Task 3 API for the CLI: score_match() and rank().

Model: Cross-Encoder V3 (jjzha/jobbert-base-cased fine-tuned, 3-class) = raw text + Task 1 entities + Task 2 profile,
the variant that matches what main.py feeds it (see README for how it compares across seeds). The input is built by the SAME functions as training
(task3_match/cross_encoder.py), so CLI inputs match what the model saw.

Weights: models/match/V3-jbprof_cls_lr2e-05/ (not in git). If missing, they are downloaded once from the public Hugging Face
repo Arxenarendra/jobbert-task3-match (override with env T3_MATCH_REPO); if that fails, a clear error is raised
(no silent fallback).

The cross-encoder alone saturates (most CVs near the top tier), so its probabilities are combined with explicit match
signals by a small logistic-regression stacker (task3_match/stacking.py, fitted on validation, stored in
data/cache/task3_stacker.json): skill coverage and job-title overlap from Task 1 entities, SBERT cosine, and the same
three relative to the other CVs ranked for the job. So a CV's score depends on the other CVs it is ranked with.
The stacker was fitted with peers, so a single CV (nothing to compare with) is scored by the cross-encoder alone.

score = isotonic(E[tier]) of the stacker, fitted on validation; tier = argmax of the stacker probabilities.
rank() orders by E[tier], which is continuous; the isotonic score is a step function with ties.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from common.schemas import FIT_TIERS, validate_match
from task3_match.common import ROOT
from task3_match.cross_encoder import encode, ent_str
from task3_match.stacking import STACKER, features, signals, stack_probs, stack_score, tokens

TAG = "V3-jbprof_cls_lr2e-05"   # trained with the Task 2 JobBERT profiles, the same source main.py feeds it
MODEL_DIR = ROOT / "models" / "match" / TAG
HF_REPO = os.environ.get("T3_MATCH_REPO", "Arxenarendra/jobbert-task3-match")
MODEL_NAME = "Cross-Encoder V3 (JobBERT; raw text + Task 1 entities + Task 2 profile) + match signals (stacked)"
BATCH = 8


def _model_dir() -> Path:
    if not (MODEL_DIR / "model.safetensors").exists():
        from huggingface_hub import snapshot_download
        try:
            snapshot_download(HF_REPO, local_dir=MODEL_DIR, allow_patterns=["*.json", "*.safetensors"])
        except Exception as exc:  # noqa: BLE001 - offline or repo unavailable
            raise FileNotFoundError(f"Task 3 weights missing in {MODEL_DIR} and download from {HF_REPO} failed ({exc}). "
                                    "Connect to the internet, set T3_MATCH_REPO, or copy models/match/V3-jbprof_cls_lr2e-05 manually.") from exc
    return MODEL_DIR


@lru_cache(maxsize=1)
def _model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = _model_dir()
    return AutoTokenizer.from_pretrained(path), AutoModelForSequenceClassification.from_pretrained(path).eval()


@lru_cache(maxsize=1)
def _stacker() -> dict:
    return json.loads(STACKER.read_text())


@lru_cache(maxsize=1)
def _ce_isotonic() -> dict:
    """Cross-encoder E[tier] -> ATS score, fitted on the val rows of its training run (thresholds, like the stacker's)."""
    from sklearn.isotonic import IsotonicRegression
    p = pd.read_parquet(ROOT / "data" / "results" / f"task3_pred_{TAG}.parquet").query("split == 'val'")
    ats = pd.read_parquet(ROOT / "outputs" / "ner" / "ats_pairs.parquet").ats_score.values[p.row.values]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p[["p0", "p1", "p2"]].values @ [0, 1, 2], ats)
    return {"iso_x": iso.X_thresholds_.tolist(), "iso_y": iso.y_thresholds_.tolist()}


@lru_cache(maxsize=1)
def _sbert():
    from sentence_transformers import SentenceTransformer
    from task3_match.baselines import SBERT
    m = SentenceTransformer(SBERT, device="cpu"); m.max_seq_length = 256
    return m


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


def _to_match(probs: np.ndarray, calibration: dict) -> dict:
    expected = float(probs @ [0, 1, 2])
    match = {"score": float(np.clip(stack_score(calibration, np.array([expected]))[0], 0, 100)),
             "tier": FIT_TIERS[int(probs.argmax())],
             "probs": {t: float(p) for t, p in zip(FIT_TIERS, probs)},
             "expected_tier": expected}
    validate_match(match)
    return match


def score_match(cv_text, cv_entities, cv_profile, job_text, job_entities) -> dict:
    """One CV alone: cross-encoder only (see module docstring)."""
    cv = {"name": "", "text": cv_text, "entities": cv_entities, "profile": cv_profile}
    return rank([cv], {"text": job_text, "entities": job_entities})[0]["match"]


def rank(cvs: list[dict], job: dict) -> list[dict]:
    """cvs: [{"name", "text", "entities", "profile"}]; job: {"text", "entities"}. Sorted by E[tier], descending.
    Two or more CVs: stacker. One CV: cross-encoder alone, with its own isotonic score (fitted on validation)."""
    from task3_match.baselines import sbert_embed
    pairs = [_texts(c["text"], c["entities"], c["profile"], job["text"], job["entities"]) for c in cvs]
    ce = _probs([p[0] for p in pairs], [p[1] for p in pairs])
    if len(cvs) == 1:
        return [{**cvs[0], "match": _to_match(ce[0], _ce_isotonic())}]
    js, jt = tokens(job["entities"]["SKILL"]), tokens(job["entities"]["JOB_TITLE"])
    sig = [signals(js, tokens(c["entities"]["SKILL"]), jt, tokens(c["entities"]["JOB_TITLE"])) for c in cvs]
    ej = sbert_embed(_sbert(), job["text"])
    d = pd.DataFrame({"job_id": 0, "skill_cov": [x[0] for x in sig], "title_jac": [x[1] for x in sig],
                      "sbert": [float(ej @ sbert_embed(_sbert(), c["text"])) for c in cvs]})
    probs = stack_probs(_stacker(), features(d, ce))
    out = [{**c, "match": _to_match(p, _stacker())} for c, p in zip(cvs, probs)]
    return sorted(out, key=lambda r: r["match"]["expected_tier"], reverse=True)
