# Output Task 1 (NER) untuk Task 2 & Task 3

Model: **JobBERT-CRF**: `jjzha/jobbert-base-cased` + lapisan CRF, data varian **B** (`notebooks/task1_ner.ipynb`, `notebooks/task1_ner_crf.ipynb`).
Dibuat dengan `python scripts/t1_ner_predict.py --export` lalu `python scripts/t1_export_outputs.py`. Untuk teks baru: `from t1_ner_predict import load, extract`.
File di sini **tidak berisi teks CV/JD**; gabungkan ke dataset asli lewat `text_md5`.

## File

| File | Baris | Kunci join | Untuk |
|---|---|---|---|
| `ner_atlas.parquet` | 13.389 | `text_md5` (md5 dari kolom `Text` ResumeAtlas), `row` (urutan baris) | Task 2 |
| `ner_ats_resume.parquet` | 778 resume unik | `text_md5` (md5 bagian resume) | Task 3 |
| `ner_ats_job.parquet` | 320 JD unik | `text_md5` (md5 bagian JD) | Task 3 |
| `ats_pairs.parquet` | 6.374 pasangan | `split`, `row_in_split` → `resume_md5`, `job_md5`, `ats_score` | Task 3 |
| `t1_*.csv` | | | angka evaluasi untuk laporan |

Kolom entity: `SKILL`, `JOB_TITLE`, `EDUCATION`, `EXPERIENCE` (list string, lowercase, tanpa duplikat) dan `experience_years` (float atau null).

## Cara join

```python
import hashlib, pandas as pd
md5 = lambda s: hashlib.md5((s or "").encode("utf8")).hexdigest()

# Task 2 — ResumeAtlas (ahmedheakl/resume-atlas)
atlas["text_md5"] = atlas["Text"].map(md5)
ner = pd.read_parquet("outputs/ner/ner_atlas.parquet").drop(columns=["row", "Category"]).drop_duplicates("text_md5")  # Atlas punya teks duplikat
atlas = atlas.merge(ner, on="text_md5", how="left")

# Task 3 — resume-ats (0xnbk/resume-ats-score-v1-en); teks = "<resume> SEP <job>"
parts = ats["text"].str.split(r"\s+SEP\s+", n=1, regex=True)
ats["resume_md5"], ats["job_md5"] = parts.str[0].map(md5), parts.str[1].fillna("").map(md5)
res = pd.read_parquet("outputs/ner/ner_ats_resume.parquet").add_prefix("resume_")
job = pd.read_parquet("outputs/ner/ner_ats_job.parquet").add_prefix("job_")
ats = ats.merge(res, left_on="resume_md5", right_on="resume_text_md5").merge(job, left_on="job_md5", right_on="job_text_md5")
```

## Pembersihan yang sudah diterapkan (`scripts/t1_export_outputs.py`)

- Entity yang hanya berisi kata fungsi (`to`, `of`, `in`, `since`, `minimum`, …) dibuang.
- `EXPERIENCE` tanpa angka (mis. `years`, `present`, `experience`) dibuang karena hanya fragmen.
- `EDUCATION` `gpa`, `of science`, `certified`, … dibuang.
- `experience_years` > 40 → null. Di ResumeAtlas tanda baca hilang, sehingga "9.5 years" terbaca "95 years".

## Kualitas & batasan — baca sebelum dipakai

| Entity | Kesiapan | Catatan |
|---|---|---|
| SKILL | **siap** | rata-rata 11 per CV Atlas, 11 per JD; ~4% teks tanpa SKILL. Gaya campur: CV 1–2 kata, JD bisa frasa panjang → **cocokkan secara semantik**, jangan exact string. |
| JOB_TITLE | **siap** | ~3% judul JD menempel dengan kata berikutnya (`business analystlocation`). |
| EDUCATION | **cukup** | span umumnya utuh (`bachelor s degree in computer science`); kadang tanpa jenjang (`computer science`). 35% JD & 18% CV Atlas tanpa EDUCATION. |
| EXPERIENCE / `experience_years` | **cukup, hati-hati** | span umumnya utuh (`5 years of experience`, `3-5 years`). Pada JD yang menyebut "N years", 67% terdeteksi (sebagian terlewat karena teks menempel, mis. `field.5 years`). `experience_years` terisi ~50% teks. Nilainya = maksimum antara durasi eksplisit dan total rentang tahun, jadi bisa lebih besar dari klaim di teks. Definisi ini belum disepakati dengan Task 2. |

- Performa di test set (CRF): CV F1 0.47 (exact) / 0.60 (partial), JD 0.57 / 0.80; EDUCATION/EXPERIENCE exact di CV 0.54 / 0.64. Data latih didominasi CV IT; domain non-IT ResumeAtlas belum punya gold.
- CRF lebih konservatif dari model softmax sebelumnya: jumlah entity per teks lebih sedikit (sebagian karena fragmen hilang). Proxy recall SKILL di JD resume-ats 0.63 (softmax 0.65; variasi antar-seed ±0.08). Prediksi softmax lama masih disimpan lokal di `data/softmax_pred_backup/` jika Task 3 lebih butuh recall.
- Teks resume-ats punya kata yang menempel (`degreepmp`); entity bisa ikut menempel (< 1% SKILL).
