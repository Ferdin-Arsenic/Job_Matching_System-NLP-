"""Task 3: data, per-job split and metrics shared by all models."""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TIERS = ["No Fit", "Potential Fit", "Good Fit"]
SEEDS = [42, 43, 44]                       # split seeds; 42 = main
md5 = lambda s: hashlib.md5((s or "").encode("utf8")).hexdigest()


def load():
    """One row per pair: resume, job, ids (md5), ats_score, tier (0/1/2)."""
    p = DATA / "resume_ats.parquet"
    if not p.exists():                                       # HF 0xnbk/resume-ats-score-v1-en, train+validation
        from datasets import load_dataset
        d = load_dataset("0xnbk/resume-ats-score-v1-en"); p.parent.mkdir(exist_ok=True)
        pd.concat([d[s].to_pandas().assign(split=s) for s in ["train", "validation"]], ignore_index=True).to_parquet(p, index=False)
    df = pd.read_parquet(p)
    parts = df.text.str.split(r"\s+SEP\s+", n=1, regex=True)
    df["resume"], df["job"] = parts.str[0], parts.str[1].fillna("")
    df["resume_id"], df["job_id"] = df.resume.map(md5), df.job.map(md5)
    df["tier"] = df.original_label.map({t: i for i, t in enumerate(TIERS)})
    return df.drop(columns=["text", "split"])


def make_splits(df, seed):
    """~80/10/10 split by job (StratifiedGroupKFold): fold 0 test, fold 1 val, rest train."""
    f = np.full(len(df), -1)
    for k, (_, idx) in enumerate(StratifiedGroupKFold(10, shuffle=True, random_state=seed).split(df, df.tier, df.job_id)): f[idx] = k
    return {"test": np.where(f == 0)[0].tolist(), "val": np.where(f == 1)[0].tolist(), "train": np.where(f > 1)[0].tolist()}


def get_splits(df):
    p = DATA / "splits" / "task3.json"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({str(s): make_splits(df, s) for s in SEEDS}))
    return {int(s): v for s, v in json.loads(p.read_text()).items()}


def evaluate(df, tier_probs, score, rank_score=None):
    """Metrics for one split. Pass a continuous rank_score (isotonic scores have ties)."""
    rs = score if rank_score is None else rank_score
    y, pred = df.tier.values, tier_probs.argmax(1)
    out = {"macro_F1": f1_score(y, pred, average="macro"), "MAE": mean_absolute_error(df.ats_score, score),
           "Spearman": spearmanr(df.ats_score, score)[0]}
    try: out["ROC_AUC"] = roc_auc_score(y, tier_probs, multi_class="ovr", average="macro")
    except ValueError: out["ROC_AUC"] = np.nan                        # a tier is missing in a small split
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    for i, t in enumerate(TIERS): out.update({f"P_{t}": p[i], f"R_{t}": r[i], f"F1_{t}": f[i]})
    sp, nd, rr = [], [], []
    for _, g in df.assign(s=rs).groupby("job_id"):
        if len(g) < 3: continue                                       # rank only jobs with >= 3 CVs
        g = g.sort_values("s", ascending=False)
        rel = g.tier.values
        if len(set(rel)) > 1: sp.append(spearmanr(g.ats_score, g.s)[0])
        disc = 1 / np.log2(np.arange(2, 7))
        dcg = (((2 ** rel[:5]) - 1) * disc[:len(rel[:5])]).sum()
        ideal = np.sort(rel)[::-1][:5]
        if ideal.max() > 0:
            nd.append(dcg / (((2 ** ideal) - 1) * disc[:len(ideal)]).sum())
        hit = np.where(rel == 2)[0]
        if len(hit): rr.append(1 / (hit[0] + 1))
    out.update({"Spearman_per_job": np.nanmean(sp), "NDCG@5": np.mean(nd), "MRR": np.mean(rr), "n_jobs_ranked": len(nd)})
    return out
