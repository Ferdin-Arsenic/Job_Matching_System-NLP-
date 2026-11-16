"""Task 3: profil STUB untuk 778 resume ATS (dipakai ablation V3 sampai model Task 2 jadi). Hasil wajib ditandai "(stub)".
- sub_field: TF-IDF + LogReg dilatih di ResumeAtlas (beda dataset dari ATS, jadi tidak bocor ke test Task 3).
- field: FIELD_MAP dari notebooks/cv_job_matching_all.ipynb (asumsi tim, belum dibekukan Task 2).
- level: dari experience_years Task 1 (<2 Entry, 2-5 Mid, >5 Senior, kosong = Unknown), aturan CLAUDE.md bagian 3.
Jalankan dari root repo:  python task3_match/profile_stub.py   -> data/cache/profiles_stub.parquet
"""
import re, sys
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import DATA, ROOT, load

FIELD_MAP = {
    **{k: "IT" for k in ["Java Developer", "Python Developer", "DotNet Developer", "SQL Developer", "SAP Developer", "ETL Developer", "React Developer",
                         "Web Designing", "Database", "DevOps", "Testing", "Network Security Engineer", "Data Science", "Blockchain", "Information Technology"]},
    **{k: "Engineering & Construction" for k in ["Electrical Engineering", "Mechanical Engineer", "Civil Engineer", "Building and Construction", "Architecture", "Automobile"]},
    **{k: "Finance" for k in ["Accountant", "Finance", "Banking"]},
    **{k: "Business & Management" for k in ["Business Analyst", "Consultant", "Management", "Operations Manager", "PMO", "Sales", "Human Resources", "BPO", "Advocate"]},
    **{k: "Creative & Media" for k in ["Designing", "Digital Media", "Arts", "Apparel", "Public Relations"]},
    **{k: "Education & Health" for k in ["Education", "Health and Fitness"]},
    **{k: "Services & Industry" for k in ["Food and Beverages", "Aviation", "Agriculture"]}}
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (s or "").lower())).strip()   # gaya teks ResumeAtlas
level = lambda y: "Unknown" if pd.isna(y) else "Entry" if y < 2 else "Mid" if y <= 5 else "Senior"

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
