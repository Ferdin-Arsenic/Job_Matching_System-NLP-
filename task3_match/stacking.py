"""Task 3 stacker: the cross-encoder combined with explicit match signals. Used by the demo (task3_match/api.py).

Why: the cross-encoder (V3, JobBERT profiles) saturates; most CVs get E[tier] near 2 regardless of content.
Stacker = logistic regression fitted on the VAL rows of a split seed (no BERT retraining), features:
  l0-l2      log probabilities of the cross-encoder (stored Colab predictions)
  skill_cov  share of the job's Task 1 SKILL tokens found in the CV's SKILL tokens
  title_jac  token Jaccard of job vs CV JOB_TITLE entities
  sbert      SBERT cosine job vs CV (same embedding as the SBERT baseline)
  *_z        the three above z-scored within each job, i.e. relative to the other candidates of that job
"CE + all" is the pre-specified design; the reduced variants were looked at afterwards and are reported for information
only, not used to choose. Evaluated on TEST for seeds 42/43/44; paired bootstrap over jobs on seed 42.
The seed-42 stacker (fitted on val of the split the demo cross-encoder was trained on) is saved for the demo.

Output: data/results/task3_stacking_seeds.csv, task3_stacking_bootstrap.csv, data/cache/task3_stacker.json
Run from repo root (CPU, ~3 min):  python task3_match/stacking.py
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task3_match.common import DATA

STACKER = DATA / "cache" / "task3_stacker.json"
STOP = set("and or the of in to with for a an on at as by is are be from using use".split())
L, RAW = ["l0", "l1", "l2"], ["skill_cov", "title_jac", "sbert"]
Z = [f + "_z" for f in RAW]


def tokens(values) -> set:
    return {w for x in values for w in re.findall(r"[a-z0-9+#]+", str(x).lower()) if len(w) > 1 and w not in STOP}


def signals(job_skill: set, cv_skill: set, job_title: set, cv_title: set) -> tuple[float, float]:
    """(skill_cov, title_jac) from token sets."""
    cov = len(job_skill & cv_skill) / len(job_skill) if job_skill else 0.0
    jac = len(job_title & cv_title) / len(job_title | cv_title) if job_title | cv_title else 0.0
    return cov, jac


def features(d: pd.DataFrame, ce_probs: np.ndarray) -> pd.DataFrame:
    """d: one row per (job, CV) with job_id + RAW columns. A job with a single CV gets z = 0."""
    X = pd.DataFrame(np.log(ce_probs + 1e-6), columns=L).assign(**{f: d[f].values for f in RAW})
    for f in RAW:
        X[f + "_z"] = d.groupby("job_id")[f].transform(lambda s: (s - s.mean()) / (s.std() + 1e-6)).fillna(0).values
    return X


def stack_probs(params: dict, X: pd.DataFrame) -> np.ndarray:
    logits = X[params["features"]].values @ np.array(params["coef"]).T + np.array(params["intercept"])
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def stack_score(params: dict, expected: np.ndarray) -> np.ndarray:
    """Isotonic E[tier] -> ATS score (clipped at the ends), stored as thresholds."""
    return np.interp(expected, params["iso_x"], params["iso_y"])


def main():
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from task3_match.baselines import sbert_raw
    from task3_match.common import ROOT, SEEDS, evaluate, get_splits, load

    RES, METRICS = DATA / "results", ["macro_F1", "ROC_AUC", "NDCG@5", "MRR"]
    VARIANTS = {"CE only (cross-encoder)": None, "CE + all": L + RAW + Z,            # pre-specified
                "CE + all except SBERT (info)": L + RAW[:2] + Z[:2], "CE + raw, no z (info)": L + RAW}
    df = load(); sp = get_splits(df)
    ner = ROOT / "outputs" / "ner"
    J = pd.read_parquet(ner / "ner_ats_job.parquet").set_index("text_md5"); R = pd.read_parquet(ner / "ner_ats_resume.parquet").set_index("text_md5")
    js, rs = {k: tokens(v) for k, v in J.SKILL.items()}, {k: tokens(v) for k, v in R.SKILL.items()}
    jt, rt = {k: tokens(v) for k, v in J.JOB_TITLE.items()}, {k: tokens(v) for k, v in R.JOB_TITLE.items()}
    sig = [signals(js.get(j, set()), rs.get(r, set()), jt.get(j, set()), rt.get(r, set())) for j, r in zip(df.job_id, df.resume_id)]
    df["skill_cov"], df["title_jac"] = [s[0] for s in sig], [s[1] for s in sig]
    df["sbert"] = sbert_raw(df)

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
            if seed == 42 and name in ("CE only (cross-encoder)", "CE + all"):
                keep[name] = (qt, iso)
            if seed == 42 and name == "CE + all":
                params = {"features": cols, "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(),
                          "iso_x": iso.X_thresholds_.tolist(), "iso_y": iso.y_thresholds_.tolist(),
                          "fitted_on": "val rows of split seed 42, cross-encoder V3-jbprof_cls_lr2e-05 predictions",
                          "sbert": "sentence-transformers/all-MiniLM-L6-v2, 150-word chunks, mean, L2-normalised"}
                # self-check: the saved parameters reproduce sklearn exactly
                assert np.allclose(stack_probs(params, Xt), qt, atol=1e-9), "stack_probs differs from sklearn"
                assert np.allclose(stack_score(params, qt @ [0, 1, 2]), iso.predict(qt @ [0, 1, 2])), "stack_score differs"
                STACKER.write_text(json.dumps(params, indent=1))
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
        dv_ = np.array([x["CE + all"][k] - x["CE only (cross-encoder)"][k] for x in draws])
        b.append(dict(metric=k, diff_mean=np.nanmean(dv_), lo=np.nanpercentile(dv_, 2.5), hi=np.nanpercentile(dv_, 97.5), p_better=np.nanmean(dv_ > 0)))
    b = pd.DataFrame(b); b.to_csv(RES / "task3_stacking_bootstrap.csv", index=False)
    print("\n== seed 42, CE + all minus CE only (paired bootstrap over jobs)\n" + b.round(3).to_string(index=False))
    print(f"\nstacker for the demo -> {STACKER}")


if __name__ == "__main__":
    main()
