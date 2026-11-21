"""Task 3: profil STUB untuk 778 resume ATS (dipakai ablation V3 sampai model Task 2 jadi). Hasil wajib ditandai "(stub)".
- sub_field: TF-IDF + LogReg dilatih di ResumeAtlas (beda dataset dari ATS, jadi tidak bocor ke test Task 3).
- field: FIELD_MAP dari task2_cls/field_map.py.
- level: dari experience_years Task 1 (<2 Entry, 2-5 Mid, >5 Senior, kosong = Unknown), aturan CLAUDE.md bagian 3.
Jalankan dari root repo:  python task3_match/profile_stub.py   -> data/cache/profiles_stub.parquet
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# Support both direct script execution and package imports from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task3_match.common import DATA, ROOT, load
from task2_cls.field_map import FIELD_MAP
from task2_cls.preprocessing import normalize_like_resumeatlas as norm, level_from_years as level

if __name__ == "__main__":
    from datasets import load_dataset
    atlas = load_dataset("ahmedheakl/resume-atlas", split="train").to_pandas()
    assert set(atlas.Category) <= set(FIELD_MAP), set(atlas.Category) - set(FIELD_MAP)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=30_000)  # shortcut: 30k (bukan 100k di TASKS.md) agar muat di RAM laptop; stub saja
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(vec.fit_transform(atlas.Text.map(norm)), atlas.Category)
    res = load().drop_duplicates("resume_id")[["resume_id", "resume"]]
    sub = clf.predict(vec.transform(res.resume.map(norm)))
    ner = pd.read_parquet(ROOT / "outputs" / "ner" / "ner_ats_resume.parquet").set_index("text_md5")
    out = pd.DataFrame({"resume_id": res.resume_id.values, "sub_field": sub, "field": [FIELD_MAP[s] for s in sub],
                        "level": [level(y) for y in ner.experience_years.reindex(res.resume_id).values]})
    (DATA / "cache").mkdir(exist_ok=True); out.to_parquet(DATA / "cache" / "profiles_stub.parquet", index=False)
    print(len(out), "resume |", out.field.value_counts().to_dict(), "|", out.level.value_counts().to_dict())
