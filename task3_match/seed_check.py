"""Task 3: variant ranking across split seeds 42/43/44, recomputed from task3_pred_{tag}.parquet with analysis.py's evaluate().
Test sets differ per seed: compare variants within a seed. Run: python task3_match/seed_check.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task3_match.common import DATA, SEEDS, evaluate, get_splits, load

RES, METRICS = DATA / "results", ["macro_F1", "ROC_AUC", "NDCG@5", "MRR"]
VARIANTS = {"V1": "V1_cls_lr2e-05", "V2": "V2_cls_lr2e-05", "V3 baseline profile": "V3_cls_lr2e-05", "V3 JobBERT profile": "V3-jbprof_cls_lr2e-05"}
df = load(); sp = get_splits(df); rows = []
for name, tag in VARIANTS.items():
    for seed in SEEDS:
        f = RES / f"task3_pred_{tag}{'' if seed == 42 else f'_s{seed}'}.parquet"
        if not f.exists(): print("missing:", f.name); continue
        p = pd.read_parquet(f).query("split == 'test'").set_index("row").loc[np.array(sp[seed]["test"])]
        pr = p[["p0", "p1", "p2"]].values
        r = evaluate(df.iloc[sp[seed]["test"]].reset_index(drop=True), pr, p.score.values, rank_score=pr @ [0, 1, 2])
        rows.append(dict(model=name, seed=seed, **{k: r[k] for k in METRICS}))
d = pd.DataFrame(rows); d.to_csv(RES / "task3_seed_check_runs.csv", index=False)
for m in METRICS:
    t = d.pivot(index="model", columns="seed", values=m); t["mean"], t["std"] = t.mean(axis=1), t.std(axis=1)
    print(f"\n== {m}\n" + t.round(3).to_string())
d.groupby("model")[METRICS].agg(["mean", "std"]).round(3).to_csv(RES / "task3_seed_check.csv")
print("\nperingkat Macro-F1 per seed (1 = terbaik):\n" + d.assign(r=d.groupby("seed").macro_F1.rank(ascending=False).astype(int)).pivot(index="model", columns="seed", values="r").to_string())
