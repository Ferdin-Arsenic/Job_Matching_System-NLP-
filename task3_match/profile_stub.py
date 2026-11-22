"""Task 3 stub profiles for the 778 ATS resumes (V3 ablation before the Task 2 model existed); mark results "(stub)".
Sub-field: TF-IDF + LogReg on ResumeAtlas. Field: FIELD_MAP. Level: Task 1 experience_years (<2 Entry, 2-5 Mid, >5 Senior).
Run from repo root: python task3_match/profile_stub.py -> data/cache/profiles_stub.parquet"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
# allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task3_match.common import DATA, ROOT, load
from task2_cls.field_map import FIELD_MAP
from task2_cls.preprocessing import normalize_like_resumeatlas as norm, level_from_years as level

if __name__ == "__main__":
    from datasets import load_dataset
    atlas = load_dataset("ahmedheakl/resume-atlas", split="train").to_pandas()
    assert set(atlas.Category) <= set(FIELD_MAP), set(atlas.Category) - set(FIELD_MAP)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=30_000)  # shortcut: 30k features to fit laptop RAM; stub only
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(vec.fit_transform(atlas.Text.map(norm)), atlas.Category)
    res = load().drop_duplicates("resume_id")[["resume_id", "resume"]]
    sub = clf.predict(vec.transform(res.resume.map(norm)))
    ner = pd.read_parquet(ROOT / "outputs" / "ner" / "ner_ats_resume.parquet").set_index("text_md5")
    out = pd.DataFrame({"resume_id": res.resume_id.values, "sub_field": sub, "field": [FIELD_MAP[s] for s in sub],
                        "level": [level(y) for y in ner.experience_years.reindex(res.resume_id).values]})
    (DATA / "cache").mkdir(exist_ok=True); out.to_parquet(DATA / "cache" / "profiles_stub.parquet", index=False)
    print(len(out), "resume |", out.field.value_counts().to_dict(), "|", out.level.value_counts().to_dict())
