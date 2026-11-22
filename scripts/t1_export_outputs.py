"""Task 1: clean NER predictions (data/ner_pred_*.parquet) and write text-free outputs to outputs/ner/.
Run after: python scripts/t1_ner_predict.py --export"""
import hashlib, re, shutil
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D, O = ROOT / "data", ROOT / "outputs" / "ner"
O.mkdir(parents=True, exist_ok=True)
E = ["SKILL", "JOB_TITLE", "EDUCATION", "EXPERIENCE"]
md5 = lambda s: hashlib.md5((s or "").encode("utf8")).hexdigest()

FUNC = {"to", "of", "in", "and", "or", "the", "a", "an", "for", "with", "at", "from", "since", "till", "until", "minimum", "min", "at least"}
NUMW = r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\b"
MAX_YEARS = 40                                    # without punctuation "9.5 years" reads as "95 years"

def keep(L, v):
    if all(x in FUNC for x in v.split()): return False                              # function words only
    if L == "EXPERIENCE": return bool(re.search(r"\d", v) or re.search(NUMW, v))   # "years", "present" = fragments
    if L == "EDUCATION": return v not in {"gpa", "of science", "of arts", "certified", "certifications"}
    return len(v) > 1 or v in {"c", "r"}

def clean(df):
    out = df.copy()
    for L in E: out[L] = out[L].map(lambda xs: [v for v in xs if keep(L, v)])
    out.loc[out.experience_years > MAX_YEARS, "experience_years"] = np.nan
    out.insert(0, "text_md5", out.text.map(md5))
    return out.drop(columns="text")

a = clean(pd.read_parquet(D / "ner_pred_atlas.parquet")).rename(columns={"source_index": "row"})
a[["row", "text_md5", "Category", *E, "experience_years"]].to_parquet(O / "ner_atlas.parquet", index=False)
for n in ["ats_resume", "ats_job"]:
    clean(pd.read_parquet(D / f"ner_pred_{n}.parquet")).drop(columns="source_index").to_parquet(O / f"ner_{n}.parquet", index=False)

ats = pd.read_parquet(D / "resume_ats.parquet")
parts = ats.text.str.split(r"\s+SEP\s+", n=1, regex=True)
pd.DataFrame({"split": ats.split, "row_in_split": ats.groupby("split").cumcount(), "resume_md5": parts.str[0].map(md5),
              "job_md5": parts.str[1].fillna("").map(md5), "ats_score": ats.ats_score}).to_parquet(O / "ats_pairs.parquet", index=False)

for f in ["t1_variant_comparison.csv", "t1_seed_check.csv", "t1_crf_check.csv", "t1_ner_results.csv"]: shutil.copy(D / f, O / f)
print("ditulis ke", O)
