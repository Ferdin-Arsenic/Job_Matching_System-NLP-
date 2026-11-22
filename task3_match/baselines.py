"""Task 3 baselines: majority, BM25 (z-score per job), SBERT cosine. Calibrated on train only.
Run from repo root: python task3_match/baselines.py"""
import re, sys
from collections import Counter
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))   # import as task3_match.* to avoid clashing with common/
from task3_match.common import DATA, SEEDS, evaluate, get_splits, load

SBERT = "sentence-transformers/all-MiniLM-L6-v2"
tok = lambda s: re.findall(r"\w+", s.lower())


def bm25_raw(df, train_idx):
    """BM25 job->resume score per pair; IDF/avgdl from unique train resumes, z-score per job vs train resumes."""
    tr = df.iloc[train_idx].drop_duplicates("resume_id")
    bm = BM25Okapi([tok(r) for r in tr.resume])
    def score(q, d):
        tf, n = Counter(d), len(d)
        return sum(bm.idf.get(w, 0) * tf[w] * (bm.k1 + 1) / (tf[w] + bm.k1 * (1 - bm.b + bm.b * n / bm.avgdl)) for w in q)
    rt = {r: tok(t) for r, t in zip(df.resume_id, df.resume)}
    train_docs = [tok(r) for r in tr.resume]
    raw, z, stats = np.zeros(len(df)), np.zeros(len(df)), {}
    for j, g in df.groupby("job_id"):
        q = tok(g.job.iloc[0]); ref = np.array([score(q, d) for d in train_docs]); stats[j] = (ref.mean(), ref.std() + 1e-9)
        for i, r in zip(g.index, g.resume_id): raw[i] = score(q, rt[r]); z[i] = (raw[i] - stats[j][0]) / stats[j][1]
    return raw, z


def sbert_embed(model, text):
    """One unit vector per text: 150-word chunks (fit 256 wordpieces), mean of chunk embeddings. Also used by the demo."""
    w = text.split(); ch = [" ".join(w[i:i + 150]) for i in range(0, max(len(w), 1), 150)]
    v = model.encode(ch, normalize_embeddings=True).mean(0)
    return v / np.linalg.norm(v)


def sbert_raw(df):
    """Cosine job vs resume: 150-word chunks, embed, average. Cached in data/cache."""
    cache = DATA / "cache" / "sbert_minilm.npz"
    if cache.exists(): emb = dict(np.load(cache, allow_pickle=True)["emb"].item())
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(SBERT, device="cpu"); m.max_seq_length = 256
        texts = {**dict(zip(df.resume_id, df.resume)), **dict(zip(df.job_id, df.job))}
        emb = {k: sbert_embed(m, t) for k, t in texts.items()}
        np.savez(cache, emb=np.array(emb, dtype=object))
    return np.array([emb[r] @ emb[j] for r, j in zip(df.resume_id, df.job_id)])


def calibrate(raw, df, tr, tgt):
    """Raw score -> ATS score (isotonic) and tier probs (LogReg); fit on train only."""
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw[tr], df.ats_score.values[tr])
    lr = LogisticRegression(max_iter=1000).fit(raw[tr, None], df.tier.values[tr])
    return lr.predict_proba(raw[tgt, None]), iso.predict(raw[tgt])


if __name__ == "__main__":
    df, splits = load(), get_splits(df := load())
    sb = sbert_raw(df); raws = {"SBERT cosine": lambda tr: sb}
    rows = []
    for seed in SEEDS:
        sp = splits[seed]; tr = np.array(sp["train"])
        bm_raw, bm_z = bm25_raw(df, tr)
        feats = {"BM25 (raw)": bm_raw, "BM25 (z per job)": bm_z, "SBERT cosine": sb}
        for part in ["val", "test"]:
            tgt = np.array(sp[part]); d = df.iloc[tgt]
            maj = np.tile(np.bincount(df.tier.values[tr], minlength=3) / len(tr), (len(tgt), 1))
            res = {"Majority / mean": evaluate(d, maj, np.full(len(tgt), df.ats_score.values[tr].mean()))}
            # leakage check: guess from the CV's train tier distribution, ignoring the job
            g = df.iloc[tr].groupby("resume_id")
            pri = g.tier.value_counts(normalize=True).unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0)
            pr = pri.reindex(d.resume_id).fillna(pd.Series(maj[0], index=[0, 1, 2])).values
            res["Resume prior (no job)"] = evaluate(d, pr, g.ats_score.mean().reindex(d.resume_id).fillna(df.ats_score.values[tr].mean()).values)
            for name, raw in feats.items():
                pr, sc = calibrate(raw, df, tr, tgt); res[name] = evaluate(d, pr, sc, rank_score=raw[tgt])   # rank by raw score
            rows += [dict(model=m, seed=seed, split=part, metric=k, value=v) for m, r in res.items() for k, v in r.items()]
    out = pd.DataFrame(rows); out.to_csv(DATA / "results" / "task3_baselines.csv", index=False)
    t = out[out.split == "test"].pivot_table(index="model", columns="metric", values="value", aggfunc=["mean", "std"])
    for m in ["macro_F1", "ROC_AUC", "MAE", "Spearman", "Spearman_per_job", "NDCG@5", "MRR"]:
        print(f"{m:18}", {k: f"{t['mean'][m][k]:.3f}±{t['std'][m][k]:.3f}" for k in t.index})
