"""Task 3 error propagation: how much do Task 1 (NER) and Task 2 (profile) errors hurt matching? CPU, test seed 42.

Model = the demo's cross-encoder (V3 trained with Task 2 JobBERT profiles), before the stacker in task3_match/stacking.py. The matching dataset has no gold
NER or gold Field for its CVs, so "predicted vs gold" is not possible. Instead the trained model stays fixed and its
upstream input is made wrong at test time (simulated upstream errors), then the drop is measured:
  clean        predicted entities + predicted profile (what the pipeline feeds it)
  profile_50   for 50% of CVs, the Task 2 profile of another test CV          (Task 2 wrong for half the CVs)
  profile_100  same for every CV
  ent_50       for 50% of CVs and jobs, the Task 1 entities of another CV/job (Task 1 wrong for half)
  ent_100      same for every CV and job
  both_100     profile_100 + ent_100
Swaps are derangements (never a CV's own value), but two CVs can share the same profile, so the share of pairs whose text
actually changed is reported. Paired bootstrap over jobs (1000 draws) gives a 95% CI for each drop.
Also: clean accuracy where Task 2 JobBERT and the baseline Task 2 agree vs disagree on Sub-field (disagree => at least one is wrong).

Output: data/results/task3_error_propagation{,_agree}.csv, task3_error_propagation_pred.parquet (cache; delete to recompute)
Run from repo root: python task3_match/error_propagation.py      (~15-30 min CPU for 6 x 656 pairs)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task3_match.common import DATA, ROOT, TIERS, evaluate, get_splits, load
from task3_match.cross_encoder import build_texts, encode, ent_str
from sklearn.isotonic import IsotonicRegression
from task3_match.api import TAG, _model

SEED, DRAWS, RES = 42, 1000, DATA / "results"
METRICS = ["macro_F1", "ROC_AUC", "NDCG@5", "MRR"]
df = load(); te = np.array(get_splits(df)[SEED]["test"]); d = df.iloc[te].reset_index(drop=True)
ner = ROOT / "outputs" / "ner"
je = pd.read_parquet(ner / "ner_ats_job.parquet").set_index("text_md5").apply(ent_str, axis=1)
re_ = pd.read_parquet(ner / "ner_ats_resume.parquet").set_index("text_md5").apply(ent_str, axis=1)
prof = pd.read_parquet(DATA / "cache" / "profiles_jobbert.parquet").set_index("resume_id")
prof_s = pd.Series([f"field: {f} | sub-field: {s} | level: {l}" for f, s, l in zip(prof.field, prof.sub_field, prof.level)], prof.index)


def texts(job_ent, res_ent, res_prof):
    """Same layout as cross_encoder.build_texts(V3), but entities/profile looked up by the (possibly swapped) ids."""
    job = [f"{e} || {t}" for e, t in zip(je.reindex(job_ent).values, d.job)]
    res = [f"{p} | {e} || {t}" for p, e, t in zip(prof_s.reindex(res_prof).values, re_.reindex(res_ent).values, d.resume)]
    return job, res


def swap(ids, frac, rng):
    """Give a random `frac` of the unique ids the value of another chosen id (rolled permutation = never itself)."""
    u = np.unique(ids); chosen = rng.choice(u, int(round(frac * len(u))), replace=False)
    m = dict(zip(u, u)); m.update(zip(chosen, np.roll(chosen, 1)))
    return np.array([m[i] for i in ids])


rng = np.random.default_rng(SEED)
rid, jid = d.resume_id.values, d.job_id.values
p50, p100 = swap(rid, 0.5, rng), swap(rid, 1.0, rng)
r50, r100, j50, j100 = swap(rid, 0.5, rng), swap(rid, 1.0, rng), swap(jid, 0.5, rng), swap(jid, 1.0, rng)
COND = {"clean": (jid, rid, rid), "profile_50": (jid, rid, p50), "profile_100": (jid, rid, p100),
        "ent_50": (j50, r50, rid), "ent_100": (j100, r100, rid), "both_100": (j100, r100, p100)}
T = {c: texts(*a) for c, a in COND.items()}

# self-check: the clean texts are exactly what the model was trained/evaluated on
bj, br = build_texts(df, "V3", profiles="profiles_jobbert.parquet")
assert T["clean"] == ([bj[i] for i in te], [br[i] for i in te]), "clean texts differ from cross_encoder.build_texts"
changed = {c: np.mean([a != b or x != y for a, b, x, y in zip(T[c][0], T["clean"][0], T[c][1], T["clean"][1])]) for c in COND}
assert changed["clean"] == 0 and min(changed[c] for c in COND if c != "clean") > 0.3, changed


def predict(job, res, bs=16):
    tk, model = _model(); enc = encode(tk, job, res, [0] * len(job))
    order, out = np.argsort([len(e["input_ids"]) for e in enc]), np.zeros((len(enc), 3))
    with torch.no_grad():
        for i in range(0, len(enc), bs):
            idx = order[i:i + bs]
            b = tk.pad([{k: enc[j][k] for k in ("input_ids", "token_type_ids", "attention_mask")} for j in idx], return_tensors="pt")
            out[idx] = model(**b).logits.softmax(-1).numpy()
    return out


cache = RES / "task3_error_propagation_pred.parquet"
pr = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(index=range(len(te)))
for c in COND:
    if f"{c}_p0" not in pr:
        print("predict", c, flush=True)
        p = predict(*T[c]); pr = pr.assign(**{f"{c}_p{k}": p[:, k] for k in range(3)}); pr.to_parquet(cache, index=False)
P = {c: pr[[f"{c}_p{k}" for k in range(3)]].values for c in COND}

stored = pd.read_parquet(RES / f"task3_pred_{TAG}.parquet").query("split == 'test'").set_index("row").loc[te][["p0", "p1", "p2"]].values
print(f"clean (CPU fp32) vs Colab predictions (fp16): max |diff| prob = {np.abs(P['clean'] - stored).max():.3f}, "
      f"same tier {np.mean(P['clean'].argmax(1) == stored.argmax(1)):.1%}")

_v = pd.read_parquet(RES / f"task3_pred_{TAG}.parquet").query("split == 'val'")      # same calibration as the training run
iso = IsotonicRegression(out_of_bounds="clip").fit(_v[["p0", "p1", "p2"]].values @ [0, 1, 2], df.ats_score.values[_v.row.values])
ev = lambda db, p: evaluate(db, p, iso.predict(p @ [0, 1, 2]), rank_score=p @ [0, 1, 2])
point = {c: ev(d, p) for c, p in P.items()}
jobs = d.job_id.unique(); by_job = {j: np.where(jid == j)[0] for j in jobs}; brng, draws = np.random.default_rng(SEED), []
for _ in range(DRAWS):
    sel = brng.choice(len(jobs), len(jobs))
    idx = np.concatenate([by_job[jobs[s]] for s in sel])
    db = d.iloc[idx].assign(job_id=np.concatenate([np.full(len(by_job[jobs[s]]), k) for k, s in enumerate(sel)]))
    draws.append({c: ev(db, p[idx]) for c, p in P.items()})
rows = []
for c in COND:
    for k in METRICS:
        dv = np.array([x[c][k] - x["clean"][k] for x in draws])
        rows.append(dict(condition=c, metric=k, value=point[c][k], drop_vs_clean=point[c][k] - point["clean"][k],
                         lo=np.nanpercentile(dv, 2.5), hi=np.nanpercentile(dv, 97.5), share_text_changed=changed[c],
                         share_pred_tier_changed=np.mean(P[c].argmax(1) != P["clean"].argmax(1))))
out = pd.DataFrame(rows); out.to_csv(RES / "task3_error_propagation.csv", index=False)
print("\n== Error propagation (test seed 42, model demo). drop = kondisi - clean, CI 95% bootstrap per job")
print(out.round(3).to_string(index=False))

t2 = pd.read_parquet(DATA / "cache" / "profiles_t2.parquet").set_index("resume_id").sub_field
agree = (t2.reindex(rid).values == prof.sub_field.reindex(rid).values)
pred = P["clean"].argmax(1); rows = []
for name, m in (("Task 2 sources agree on Sub-field", agree), ("Task 2 sources disagree", ~agree)):
    rows.append(dict(group=name, pairs=int(m.sum()), cvs=d.resume_id[m].nunique(), accuracy=np.mean(pred[m] == d.tier.values[m]),
                     macro_F1=f1_score(d.tier.values[m], pred[m], average="macro"),
                     pred_good_fit=np.mean(pred[m] == 2), true_good_fit=np.mean(d.tier.values[m] == 2)))
ag = pd.DataFrame(rows); ag.to_csv(RES / "task3_error_propagation_agree.csv", index=False)
print("\n== Clean predictions split by Task 2 agreement (proxy for a wrong profile)\n" + ag.round(3).to_string(index=False))
