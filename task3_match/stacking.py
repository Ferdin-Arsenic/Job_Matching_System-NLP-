"""Task 3 additional experiment (NOT used by the demo): stack the cross-encoder with explicit match signals.

Why: the demo cross-encoder (V3, JobBERT profiles) saturates; most CVs get E[tier] near 2 regardless of content.
Stacker = logistic regression fitted on the VAL rows of each split seed, evaluated on TEST, using:
  log probs of the cross-encoder (stored Colab predictions, no retraining)
  skill_cov  share of the job's Task 1 SKILL tokens found in the CV's SKILL tokens
  title_jac  token Jaccard of job vs CV JOB_TITLE entities
  sbert      SBERT cosine job vs CV (data/cache/sbert_minilm.npz, same as the SBERT baseline)
  and the three z-scored within each job (relative to the other candidates of that job).
"CE + all" is the pre-specified design; the reduced variants were looked at afterwards and are reported for information
only, not used to choose. Paired bootstrap over jobs (seed 42, 1000 draws) for CE + all minus CE only.

Output: data/results/task3_stacking_seeds.csv, task3_stacking_bootstrap.csv.  Run from repo root (CPU, ~3 min):
    python task3_match/stacking.py
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task3_match.common import DATA, ROOT, SEEDS, evaluate, get_splits, load
from task3_match.baselines import sbert_raw

RES, METRICS = DATA / "results", ["macro_F1", "ROC_AUC", "NDCG@5", "MRR"]
STOP = set("and or the of in to with for a an on at as by is are be from using use".split())
RAW = ["skill_cov", "title_jac", "sbert"]; Z = [f + "_z" for f in RAW]; L = ["l0", "l1", "l2"]
VARIANTS = {"CE only (demo)": None, "CE + all": L + RAW + Z,            # pre-specified
            "CE + all except SBERT (info)": L + RAW[:2] + Z[:2], "CE + raw, no z (info)": L + RAW}


def tok(xs):
    return {w for x in xs for w in re.findall(r"[a-z0-9+#]+", str(x).lower()) if len(w) > 1 and w not in STOP}


df = load(); sp = get_splits(df)
ner = ROOT / "outputs" / "ner"
J = pd.read_parquet(ner / "ner_ats_job.parquet").set_index("text_md5"); R = pd.read_parquet(ner / "ner_ats_resume.parquet").set_index("text_md5")
jsk, rsk = {k: tok(v) for k, v in J.SKILL.items()}, {k: tok(v) for k, v in R.SKILL.items()}
jti, rti = {k: tok(v) for k, v in J.JOB_TITLE.items()}, {k: tok(v) for k, v in R.JOB_TITLE.items()}
df["skill_cov"] = [len(a & b) / len(a) if a else 0.0 for a, b in ((jsk.get(j, set()), rsk.get(r, set())) for j, r in zip(df.job_id, df.resume_id))]
df["title_jac"] = [len(a & b) / len(a | b) if a | b else 0.0 for a, b in ((jti.get(j, set()), rti.get(r, set())) for j, r in zip(df.job_id, df.resume_id))]
df["sbert"] = sbert_raw(df)


def features(d, pr):
    X = pd.DataFrame(np.log(pr + 1e-6), columns=L).assign(**{f: d[f].values for f in RAW})
    for f in RAW:
        X[f + "_z"] = d.groupby("job_id")[f].transform(lambda s: (s - s.mean()) / (s.std() + 1e-6)).fillna(0).values
    return X


rows, keep = [], {}
for seed in SEEDS:
    p = pd.read_parquet(RES / f"task3_pred_V3-jbprof_cls_lr2e-05{'' if seed == 42 else f'_s{seed}'}.parquet").set_index("row")
    part = {}
    for name in ("val", "test"):
        idx = np.array(sp[seed][name]); d = df.iloc[idx].reset_index(drop=True); pr = p.loc[idx, ["p0", "p1", "p2"]].values
        part[name] = (d, pr, features(d, pr))
    (dv, pv, Xv), (dt, pt, Xt) = part["val"], part["test"]
    for name, cols in VARIANTS.items():
        if cols is None:
            qv, qt = pv, pt
        else:
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xv[cols], dv.tier)
            qv, qt = lr.predict_proba(Xv[cols]), lr.predict_proba(Xt[cols])
        iso = IsotonicRegression(out_of_bounds="clip").fit(qv @ [0, 1, 2], dv.ats_score)
        r = evaluate(dt, qt, iso.predict(qt @ [0, 1, 2]), rank_score=qt @ [0, 1, 2])
        rows.append(dict(seed=seed, model=name, **{k: r[k] for k in METRICS}))
        if seed == 42 and name in ("CE only (demo)", "CE + all"):
            keep[name] = (qt, iso)
res = pd.DataFrame(rows); res.to_csv(RES / "task3_stacking_seeds.csv", index=False)
for k in METRICS:
    t = res.pivot(index="model", columns="seed", values=k); t["mean"] = t.mean(axis=1)
    print(f"\n== {k} (test)\n" + t.round(3).to_string())

d = df.iloc[np.array(sp[42]["test"])].reset_index(drop=True)
jobs = d.job_id.unique(); by = {j: np.where(d.job_id.values == j)[0] for j in jobs}; rng, draws = np.random.default_rng(42), []
for _ in range(1000):
    sel = rng.choice(len(jobs), len(jobs)); idx = np.concatenate([by[jobs[k]] for k in sel])
    db = d.iloc[idx].assign(job_id=np.concatenate([np.full(len(by[jobs[k]]), i) for i, k in enumerate(sel)]))
    draws.append({m: evaluate(db, q[idx], iso.predict(q[idx] @ [0, 1, 2]), rank_score=q[idx] @ [0, 1, 2]) for m, (q, iso) in keep.items()})
b = []
for k in METRICS:
    dv_ = np.array([x["CE + all"][k] - x["CE only (demo)"][k] for x in draws])
    b.append(dict(metric=k, diff_mean=np.nanmean(dv_), lo=np.nanpercentile(dv_, 2.5), hi=np.nanpercentile(dv_, 97.5), p_better=np.nanmean(dv_ > 0)))
b = pd.DataFrame(b); b.to_csv(RES / "task3_stacking_bootstrap.csv", index=False)
print("\n== seed 42, CE + all minus CE only (paired bootstrap over jobs)\n" + b.round(3).to_string(index=False))
