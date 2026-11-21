"""Task 3 API for the CLI: score_match() and rank()
Current model: SBERT cosine BASELINE (sentence-transformers/all-MiniLM-L6-v2), identical to
task3_match/baselines.py: 150-word chunks, mean-pooled, cosine job vs resume; isotonic -> ATS score
and 1-feature LogReg -> tier probabilities, both fitted on the seed-42 TRAIN split only.
"""
from functools import lru_cache

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from common.schemas import FIT_TIERS, validate_match
from task3_match.common import DATA, ROOT, get_splits, load

SBERT = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "SBERT cosine baseline (raw text; Cross-Encoder planned)"
CALIBRATORS = ROOT / "models" / "match" / "sbert_baseline" / "calibrators.joblib"


@lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SBERT, device="cpu")
    model.max_seq_length = 256
    return model


def embed(texts: list[str]) -> np.ndarray:
    out = []
    for text in texts:
        words = text.split()
        chunks = [" ".join(words[i:i + 150]) for i in range(0, max(len(words), 1), 150)]
        v = _encoder().encode(chunks, normalize_embeddings=True).mean(0)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out)


@lru_cache(maxsize=1)
def _calibrators():
    if CALIBRATORS.exists():
        return joblib.load(CALIBRATORS)
    # Fit once from the cached baseline embeddings (same as baselines.py), train split only.
    cache = DATA / "cache" / "sbert_minilm.npz"
    if not cache.exists():
        raise FileNotFoundError(f"{cache} missing; run python task3_match/baselines.py first")
    emb = dict(np.load(cache, allow_pickle=True)["emb"].item())
    df = load()
    tr = np.array(get_splits(df)[42]["train"])
    d = df.iloc[tr]
    raw = np.array([emb[r] @ emb[j] for r, j in zip(d.resume_id, d.job_id)])
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw, d.ats_score.values)
    lr = LogisticRegression(max_iter=1000).fit(raw[:, None], d.tier.values)
    CALIBRATORS.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"iso": iso, "lr": lr, "fitted_on": "task3 split seed 42, train rows", "n": len(d)}, CALIBRATORS)
    return joblib.load(CALIBRATORS)


def _to_match(cosine: float) -> dict:
    cal = _calibrators()
    probs = cal["lr"].predict_proba([[cosine]])[0]
    match = {"score": float(np.clip(cal["iso"].predict([cosine])[0], 0, 100)),
             "tier": FIT_TIERS[int(probs.argmax())],
             "probs": {t: float(p) for t, p in zip(FIT_TIERS, probs)},
             "cosine": float(cosine)}
    validate_match(match)
    return match


def score_match(cv_text, cv_entities, cv_profile, job_text, job_entities) -> dict:
    cv_vec, job_vec = embed([cv_text, job_text])
    return _to_match(float(cv_vec @ job_vec))


def rank(cvs: list[dict], job: dict) -> list[dict]:
    """cvs: [{"name", "text", "entities", "profile"}]; job: {"text", "entities"}. Sorted by score, descending."""
    job_vec = embed([job["text"]])[0]
    cv_vecs = embed([cv["text"] for cv in cvs])
    out = [{**cv, "match": _to_match(float(v @ job_vec))} for cv, v in zip(cvs, cv_vecs)]
    # Isotonic output is a step function, so break score ties with the raw cosine.
    return sorted(out, key=lambda r: (r["match"]["score"], r["match"]["cosine"]), reverse=True)
