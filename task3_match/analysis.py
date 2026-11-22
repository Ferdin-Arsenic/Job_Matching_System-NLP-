"""Task 3 analysis: bootstrap CIs, paired ablation, error analysis, charts (CPU, seed 42).
Needs the Colab predictions task3_pred_V*.parquet and data/cache/profiles_jobbert.parquet.
Run from repo root: python task3_match/analysis.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task3_match.common import DATA, ROOT, TIERS, evaluate, get_splits, load
from task3_match.baselines import calibrate, sbert_raw

SEED, DRAWS, LR = 42, 1000, "2e-05"
RES, FIG = DATA / "results", ROOT / "figures"; FIG.mkdir(exist_ok=True)
METRICS = ["macro_F1", "NDCG@5", "MRR", "Spearman_per_job"]

df = load(); sp = get_splits(df)[SEED]
tr, te = np.array(sp["train"]), np.array(sp["test"])
d = df.iloc[te].reset_index(drop=True)

# --- predictions per model on test rows ---
P = {}
raw = sbert_raw(df)
P["SBERT cosine"] = (*calibrate(raw, df, tr, te), raw[te])                      # (probs, score, rank score)
g = df.iloc[tr].groupby("resume_id")                                            # baseline: guess from the CV alone
prior = g.tier.value_counts(normalize=True).unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0)
glob = np.bincount(df.tier.values[tr], minlength=3) / len(tr)
_pp = prior.reindex(d.resume_id).fillna(pd.Series(glob, index=[0, 1, 2])).values
_ss = g.ats_score.mean().reindex(d.resume_id).fillna(df.ats_score.values[tr].mean()).values
P["Resume prior (no job)"] = (_pp, _ss, _ss)
for v in ("V1", "V2", "V3"):
    p = pd.read_parquet(RES / f"task3_pred_{v}_cls_lr{LR}.parquet").query("split == 'test'").set_index("row").loc[te]
    pr = p[["p0", "p1", "p2"]].values; P[f"CE {v} (cls)"] = (pr, p.score.values, pr @ [0, 1, 2])   # ranking: E[tier]
CE = [k for k in P if k.startswith("CE")]
# V3 with JobBERT profiles = demo model (kept apart from the CE ablation labels)
_p = pd.read_parquet(RES / f"task3_pred_V3-jbprof_cls_lr{LR}.parquet").query("split == 'test'").set_index("row").loc[te]
_pr = _p[["p0", "p1", "p2"]].values; P["V3-jbprof (cls)"] = (_pr, _p.score.values, _pr @ [0, 1, 2])
ERR = "V3-jbprof (cls)"                                                          # demo model
SHORT = {"CE V1 (cls)": "V1", "CE V2 (cls)": "V2", "CE V3 (cls)": "V3", ERR: "V3jb"}


# --- 1. bootstrap over jobs ---
jobs = d.job_id.unique(); by_job = {j: np.where(d.job_id.values == j)[0] for j in jobs}
rng, draws = np.random.default_rng(SEED), []
for b in range(DRAWS):
    sel = rng.choice(len(jobs), len(jobs))
    idx = np.concatenate([by_job[jobs[s]] for s in sel])
    jid = np.concatenate([np.full(len(by_job[jobs[s]]), k) for k, s in enumerate(sel)])
    db = d.iloc[idx].assign(job_id=jid)
    draws.append({m: evaluate(db, pr[idx], sc[idx], rank_score=rk[idx]) for m, (pr, sc, rk) in P.items()})
point = {m: evaluate(d, pr, sc, rank_score=rk) for m, (pr, sc, rk) in P.items()}
rows = []
for m in P:
    for k in METRICS:
        v = np.array([x[m][k] for x in draws])
        rows.append(dict(model=m, metric=k, point=point[m][k], lo=np.nanpercentile(v, 2.5), hi=np.nanpercentile(v, 97.5)))
boot = pd.DataFrame(rows); boot.to_csv(RES / "task3_bootstrap.csv", index=False)

pairs = [("CE V2 (cls)", "CE V1 (cls)"), ("CE V3 (cls)", "CE V2 (cls)"), ("CE V3 (cls)", "CE V1 (cls)"),
         ("CE V1 (cls)", "SBERT cosine"), ("CE V3 (cls)", "SBERT cosine"), ("CE V1 (cls)", "Resume prior (no job)"),
         ("CE V3 (cls)", "Resume prior (no job)"), ("V3-jbprof (cls)", "CE V3 (cls)"), ("V3-jbprof (cls)", "CE V1 (cls)"),
         ("V3-jbprof (cls)", "SBERT cosine")]
rows = []
for a, b in pairs:
    for k in METRICS:
        dv = np.array([x[a][k] - x[b][k] for x in draws])
        rows.append(dict(a=a, b=b, metric=k, diff=point[a][k] - point[b][k], lo=np.nanpercentile(dv, 2.5), hi=np.nanpercentile(dv, 97.5),
                         p_a_better=np.nanmean(dv > 0), significant=bool(np.nanpercentile(dv, 2.5) > 0 or np.nanpercentile(dv, 97.5) < 0)))
diff = pd.DataFrame(rows); diff.to_csv(RES / "task3_bootstrap_diff.csv", index=False)
print(f"test: {len(d)} pasangan, {len(jobs)} job | skor isotonic V3 yang bernilai kembar: {1 - len(np.unique(P['CE V3 (cls)'][1])) / len(d):.0%} (ranking memakai E[tier] kontinu)")
print("\n== Bootstrap 95% CI (1000 resample job)"); print(boot.pivot(index="model", columns="metric", values="point").round(3).to_string())
print(diff.query("metric in ['macro_F1','NDCG@5','MRR']").round(3).to_string(index=False))

# --- 2. confusion per tier ---
pred = {m: P[m][0].argmax(1) for m in SHORT}
for m in ("CE V1 (cls)", ERR):
    cm = pd.crosstab(pd.Categorical(d.tier, [0, 1, 2]), pd.Categorical(pred[m], [0, 1, 2]), dropna=False)
    cm.index, cm.columns = [f"true {t}" for t in TIERS], [f"pred {t}" for t in TIERS]
    cm.to_csv(RES / f"task3_confusion_{SHORT[m]}.csv"); print(f"\n== Confusion {m}\n{cm.to_string()}")

# --- 3. profile, entity, seniority ---
prof = pd.read_parquet(DATA / "cache" / "profiles_jobbert.parquet").set_index("resume_id")
ner_r = pd.read_parquet(ROOT / "outputs" / "ner" / "ner_ats_resume.parquet").set_index("text_md5")
ner_j = pd.read_parquet(ROOT / "outputs" / "ner" / "ner_ats_job.parquet").set_index("text_md5")
d["field"], d["level"] = prof.field.reindex(d.resume_id).values, prof.level.reindex(d.resume_id).values
d["req_years"] = ner_j.experience_years.reindex(d.job_id).values
d["cv_years"] = ner_r.experience_years.reindex(d.resume_id).values
d["too_junior"] = ((d.level == "Entry") & (d.req_years >= 2)) | ((d.level == "Mid") & (d.req_years > 5))   # same rule as pipeline/run.py
d["pred"], d["p_pred"] = pred[ERR], P[ERR][0].max(1)
fp = d[(d.pred == 2) & (d.tier == 0)]
sen = pd.DataFrame({"kelompok": ["semua pasangan test", "Good Fit asli", "No Fit asli", "false positive (pred Good, asli No Fit)"],
                    "n": [len(d), (d.tier == 2).sum(), (d.tier == 0).sum(), len(fp)],
                    "persen_resume_terlalu_junior": [x.too_junior.mean() for x in (d, d[d.tier == 2], d[d.tier == 0], fp)],
                    "persen_syarat_tahun_diketahui": [x.req_years.notna().mean() for x in (d, d[d.tier == 2], d[d.tier == 0], fp)]})
sen.to_csv(RES / "task3_error_seniority.csv", index=False); print("\n== Seniority (resume level vs syarat tahun job)\n" + sen.round(3).to_string(index=False))

from sklearn.metrics import f1_score
fld = []
for f, x in d.groupby("field"):
    fld.append(dict(field=f, n=len(x), jobs=x.job_id.nunique(), **{f"acc_{SHORT[m]}": (pred[m][x.index] == x.tier).mean() for m in SHORT}))
fld = pd.DataFrame(fld).sort_values("n", ascending=False); fld.to_csv(RES / "task3_error_by_field.csv", index=False)
print("\n== Akurasi tier per Field resume (n kecil = tidak stabil)\n" + fld.round(3).to_string(index=False))

d["words"] = d.resume.str.split().str.len() + d.job.str.split().str.len()
d["len_bin"] = pd.qcut(d.words, 4, labels=["Q1 pendek", "Q2", "Q3", "Q4 panjang"])
trunc = pd.DataFrame([dict(bin=b, n=len(x), mean_words=x.words.mean(), **{f"acc_{SHORT[m]}": (pred[m][x.index] == x.tier).mean() for m in SHORT})
                      for b, x in d.groupby("len_bin", observed=True)])
trunc.to_csv(RES / "task3_error_truncation.csv", index=False); print("\n== Akurasi vs panjang teks (resume+job, kata)\n" + trunc.round(3).to_string(index=False))

# --- 4. ten confident errors: 5 too high, 5 too low ---
def first(xs, n=3): return ", ".join(list(xs)[:n])
ex = d[d.pred != d.tier].assign(arah=lambda x: np.where(x.pred > x.tier, "terlalu tinggi", "terlalu rendah"))
ex = pd.concat([ex[ex.arah == a].nlargest(5, "p_pred") for a in ("terlalu tinggi", "terlalu rendah")])
ex["job_title"] = [first(ner_j.JOB_TITLE[j]) for j in ex.job_id]
ex["resume_title"] = [first(ner_r.JOB_TITLE[r]) for r in ex.resume_id]
ex["skill_overlap"] = [len({s for s in ner_j.SKILL[j]} & {s for s in ner_r.SKILL[r]}) for j, r in zip(ex.job_id, ex.resume_id)]
ex["true"], ex["pred_tier"] = [TIERS[i] for i in ex.tier], [TIERS[i] for i in ex.pred]
ex[["arah", "true", "pred_tier", "p_pred", "job_title", "req_years", "resume_title", "field", "level", "cv_years", "skill_overlap"]].to_csv(RES / "task3_error_examples.csv", index=False)
print("\n== 10 contoh salah (V3-jbprof, model demo)\n" + ex[["arah", "true", "pred_tier", "p_pred", "job_title", "resume_title", "field", "level", "skill_overlap"]].round(2).to_string(index=False))

# --- 5. chart ---
INK, MUTED, GRID, GRAY = "#0b0b0b", "#52514e", "#e3e2de", "#9a9993"
BLUE = {"CE V1 (cls)": "#8fbcf0", "CE V2 (cls)": "#4a8fe0", "CE V3 (cls)": "#1f5fb0", "V3-jbprof (cls)": "#e08a2e"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})
order = ["Resume prior (no job)", "SBERT cosine", "CE V1 (cls)", "CE V2 (cls)", "CE V3 (cls)", "V3-jbprof (cls)"]
label = {"Resume prior (no job)": "Resume prior (tanpa job)", "SBERT cosine": "SBERT cosine", "CE V1 (cls)": "Cross-encoder V1 (raw)",
         "CE V2 (cls)": "Cross-encoder V2 (+entity)", "CE V3 (cls)": "Cross-encoder V3 (+profil baseline)", "V3-jbprof (cls)": "Cross-encoder V3 (+profil JobBERT)"}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, k, title in zip(axes, ["macro_F1", "NDCG@5", "MRR"], ["Macro-F1 (tier)", "NDCG@5 (ranking per job)", "MRR (ranking per job)"]):
    for i, m in enumerate(order):
        r = boot[(boot.model == m) & (boot.metric == k)].iloc[0]; c = BLUE.get(m, GRAY)
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=2); ax.plot(r.point, i, "o", color=c, ms=8, mec="white", mew=1.5)
        ax.text(r.hi + 0.01, i, f"{r.point:.2f}", va="center", color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left"); ax.set_yticks(range(len(order)), [label[m] for m in order]); ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_xlim(0.2, 1.08 if k == "MRR" else 0.95)
fig.suptitle("Test seed 42, 26 job: titik = nilai, garis = 95% CI bootstrap per job", x=0.01, ha="left", color=MUTED, fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "t3_res_ablation_ci.png", dpi=200); plt.close(fig)

cm = pd.read_csv(RES / "task3_confusion_V3jb.csv", index_col=0).values; rn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(4.8, 4.2)); ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3): ax.text(j, i, f"{cm[i, j]}\n{rn[i, j]:.0%}", ha="center", va="center", color="white" if rn[i, j] > 0.5 else INK, fontsize=10)
ax.set_xticks(range(3), TIERS, rotation=20); ax.set_yticks(range(3), TIERS); ax.set_xlabel("Prediksi"); ax.set_ylabel("Sebenarnya")
ax.set_title("Confusion matrix: Cross-encoder V3 (profil JobBERT, model demo)", color=INK, loc="left", fontsize=10.5); ax.spines[:].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "t3_res_confusion_v3.png", dpi=200); plt.close(fig)

def ndcg_by_job(db, sc):
    out = {}
    for j, x in db.assign(s=sc).groupby("job_id"):
        if len(x) < 3: continue
        rel = x.sort_values("s", ascending=False).tier.values; disc = 1 / np.log2(np.arange(2, 7)); ideal = np.sort(rel)[::-1][:5]
        if ideal.max() > 0: out[j] = (((2 ** rel[:5]) - 1) * disc[:len(rel[:5])]).sum() / (((2 ** ideal) - 1) * disc[:len(ideal)]).sum()
    return pd.Series(out)
n1, n3 = ndcg_by_job(d, P["CE V1 (cls)"][2]), ndcg_by_job(d, P[ERR][2]); jn = pd.concat([n1, n3], axis=1, keys=["V1", "V3"]).sort_values("V3")
jn.round(3).to_csv(RES / "task3_ndcg_per_job.csv"); fig, ax = plt.subplots(figsize=(6.4, 5.4)); y = np.arange(len(jn))
ax.hlines(y, jn.V1, jn.V3, color=GRID, lw=2); ax.plot(jn.V1, y, "o", color=BLUE["CE V1 (cls)"], ms=7, label="V1 (raw)", mec="white"); ax.plot(jn.V3, y, "o", color=BLUE[ERR], ms=7, label="V3 (+profil JobBERT)", mec="white")
ax.set_yticks(y, [f"job {i + 1}" for i in y]); ax.set_xlabel("NDCG@5"); ax.legend(frameon=False, loc="lower left"); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title(f"NDCG@5 per job (V3 > V1 di {int((jn.V3 > jn.V1).sum())} dari {len(jn)} job)", color=INK, loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / "t3_res_ndcg_per_job.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.2)); r2 = np.random.default_rng(0)
for t in range(3):
    s = d.ats_score[d.tier == t].values; sc = P[ERR][1][(d.tier == t).values]
    ax.scatter(s + r2.normal(0, 0.6, len(s)), sc + r2.normal(0, 0.4, len(s)), s=12, alpha=0.5, color=["#9a9993", "#4a8fe0", "#1f5fb0"][t], label=TIERS[t], lw=0)
ax.plot([15, 92], [15, 92], color=INK, lw=1, ls=(0, (4, 3))); ax.set_xlabel("Skor ATS asli"); ax.set_ylabel("Skor prediksi (V3 profil JobBERT)")
ax.legend(frameon=False, loc="upper left"); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("Skor prediksi vs asli (garis putus-putus = prediksi sempurna)", color=INK, loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "t3_res_pred_vs_actual.png", dpi=200); plt.close(fig)
print("\nchart ->", ", ".join(sorted(p.name for p in FIG.glob("t3_res_*.png"))))

# --- 6. final table, seed 42 ---
import glob, re
rows = []
for f in sorted(glob.glob(str(RES / "task3_pred_V*_lr*.parquet"))):
    m = re.search(r"pred_(V\d(?:-jbprof)?)_(cls|reg)_lr([\d.e-]+)\.parquet$", f)
    if not m: continue                                                          # seeds 43/44: see seed_check.py
    v, kind, lr = m.groups()
    p = pd.read_parquet(f).query("split == 'test'").set_index("row").loc[te]; pr = p[["p0", "p1", "p2"]].values
    r = evaluate(d, pr, p.score.values, rank_score=pr @ [0, 1, 2] if kind == "cls" else None)
    rows.append(dict(model=f"Cross-Encoder {v} {kind}", lr=float(lr), **{k: r[k] for k in ["macro_F1", "ROC_AUC", "MAE", "Spearman", "Spearman_per_job", "NDCG@5", "MRR"]}))
ce_all = pd.DataFrame(rows); ce_all.to_csv(RES / "task3_ce_rescored_seed42.csv", index=False)
base = pd.read_csv(RES / "task3_baselines.csv").query("split == 'test' and seed == 42").pivot_table(index="model", columns="metric", values="value").reset_index()
cols = ["model", "macro_F1", "ROC_AUC", "MAE", "Spearman", "Spearman_per_job", "NDCG@5", "MRR"]
chosen = ce_all[(ce_all.model.str.endswith("cls") & (ce_all.lr == 2e-5)) | (ce_all.model.str.endswith("reg") & (ce_all.lr == 3e-5))]   # lr chosen on val (Colab notebook)
final = pd.concat([base[cols], chosen[cols]], ignore_index=True); final.to_csv(RES / "task3_final_seed42.csv", index=False)
print("\n== Tabel final (test, seed 42, ranking dgn skor kontinu)\n" + final.round(3).to_string(index=False))
