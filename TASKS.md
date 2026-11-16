# TASKS.md — Spesifikasi implementasi per task (CV-Job Matching, Group 6)

Pelengkap `CLAUDE.md` (konteks, dataset, desain eksperimen). Dokumen ini = **cara membangunnya**. Nilai default (hyperparameter, ukuran chunk, dll.) adalah usulan awal; boleh diubah tim, tapi catat perubahannya.

Urutan baca: `CLAUDE.md` -> bagian 0 di sini (kontrak bersama) -> bagian task milikmu.

---

## 0. Kontrak bersama (wajib disepakati 3 orang SEBELUM coding)

Tiga task dikerjakan paralel, jadi antarmuka harus tetap. Semua fungsi menerima/menghasilkan struktur berikut.

```python
# common/schemas.py
Entity   = {"text": str, "label": str, "start": int, "end": int}          # offset karakter, end eksklusif
Profile  = {"field": str, "sub_field": str, "level": str,                  # level: "Entry"|"Mid"|"Senior"|"Unknown"
            "field_probs": dict, "sub_probs": dict}                        # probs boleh kosong
Match    = {"score": float, "tier": str, "probs": dict}                    # score 0-100; tier: "No Fit"|"Potential Fit"|"Good Fit"

# task1_ner/api.py
def extract_entities(text: str) -> list[Entity]: ...                       # menangani teks panjang (chunking) sendiri

# task2_cls/api.py
def classify_cv(text: str, entities: list[Entity] | None = None) -> Profile: ...

# task3_match/api.py
def score_match(cv_text, cv_entities, cv_profile, job_text, job_entities) -> Match: ...
def rank(cvs: list[dict], job: dict) -> list[dict]: ...                    # urut turun berdasarkan score
```

Label entity (13, `OTHER` di-drop): `ACTION, CERTIFICATION, COLLABORATION, COMPANY, DESIGNATION, EDUCATION, EMAIL, EXPERIENCE, EXPERTISE, LANGUAGE, LOCATION, PERSON, SKILL`.

Aturan bersama:
- Struktur: `common/` (schemas, seed, io), `task1_ner/`, `task2_cls/`, `task3_match/`, `pipeline/`, `models/`, `data/`, `figures/`, `main.py`.
- Seed 42. Simpan **id split ke file** (`data/splits/<task>.json`) supaya baseline dan proposed memakai split yang persis sama.
- Setiap `api.py` punya stub yang jalan tanpa model terlatih (mis. Task 1 = rule+gazetteer dari notebook, Task 2 = TF-IDF+LR), supaya task lain tidak menunggu.
- Satu file `tests/test_contracts.py`: panggil tiap `api` dengan teks contoh, assert bentuk output sesuai skema di atas.
- Hasil eksperimen ditulis ke `data/results/<task>_<model>.csv` (satu baris per metrik) dan chart ke `figures/`.

---

## Task 1 — NER (PIC 13523065)

**Tujuan:** ekstrak entity dari teks CV dan teks job description.

**Data:** Kaggle `yashpwrr/resume-ner-training-dataset` via `kagglehub`. Kolom `text`, `annotations` (list `[start, end, label]`).

**Langkah preprocessing (urutan penting):**
1. Ganti karakter *lone surrogate* (U+D800–U+DFFF) 1:1 dengan spasi, supaya offset anotasi tetap valid (ada 1 CV bermasalah).
2. **Dedup** by hash teks ternormalisasi (strip, lowercase, rapikan whitespace): 5.960 -> sekitar 4.290 teks unik.
3. Split 80/10/10 **di level dokumen**, simpan id.
4. Bersihkan anotasi: strip whitespace di tepi span (3.008 kasus), buang `OTHER`, buang span > 60 karakter (umumnya blob section "Skills").
5. **Kebijakan overlap** (ada 67.786 span yang overlap, SKILL mendominasi): untuk span yang bertabrakan, pertahankan yang **terpanjang**; jika sama panjang, yang mulai lebih awal. Tulis kebijakan ini di README karena langsung memengaruhi F1.
6. Tokenisasi regex `\w+|[^\w\s]` dengan offset -> tag BIO per token (token mendapat label span yang menutupinya).

**Model:**

| Model | Detail |
|---|---|
| Rule + gazetteer | sudah ada di notebook (micro-F1 ≈ 0,51, hampir semuanya dari SKILL; tipe lain ≈ 0). Dipakai sebagai lower bound. |
| **CRF (baseline)** | `sklearn-crfsuite`, algoritma `lbfgs`. Fitur per token: lowercase, suffix 2–3, `isupper/istitle/isdigit`, shape (Xxxx/dd), POS (nltk) bila tersedia, jendela konteks ±2. Tuning di val: `c1, c2 ∈ {0.01, 0.1, 1}`, `max_iterations` 100–200. |
| **JobBERT (proposed)** | `jjzha/jobbert-base-cased` (terverifikasi ada: BERT-base-cased yang dilanjutkan pre-training pada ±3,2 juta kalimat job posting; **bukan** model NER siap pakai, jadi head token-classification baru dengan 27 label = 13×2+1). `AutoModelForTokenClassification`. |

**Fine-tuning JobBERT:**
- Teks > 512 wordpiece (42% CV) -> `tokenizer(..., max_length=512, stride=128, return_overflowing_tokens=True, is_split_into_words=True)`. Label hanya pada subword pertama, sisanya `-100`.
- Gabungkan prediksi antar-chunk: untuk token yang muncul di dua chunk, ambil prediksi dari chunk tempat token itu paling jauh dari tepi.
- Default: lr ∈ {2e-5, 3e-5, 5e-5}, epoch 3–5, batch 16, warmup 0,1, weight decay 0,01, fp16. Pilih checkpoint terbaik berdasar val **macro-F1 antar tipe entity**.
- Colab GPU. Sediakan `SAMPLE_N` agar bisa dites di CPU.

**Evaluasi:** entity-level exact-span P/R/F1 **per tipe** + micro + macro (pakai `seqeval` atau fungsi `evaluate()` di notebook, **satu saja untuk semua model**). Headline = macro, karena SKILL ≈ 95% anotasi dan micro-F1 menyembunyikan tipe langka.

**Domain shift CV vs job:** dataset `khaliladimassi/cleaned-job-description` **tidak punya label NER**, jadi tidak ada F1 untuk sisi job. Ambil 500 posting (filter bahasa Inggris dengan rasio stopword; baris pertama file berbahasa Swedia), jalankan model, bandingkan **densitas entity per 1000 token** vs CV test. Opsional: anotasi manual ±50 posting untuk F1 sungguhan.

**Error analysis:** boundary error vs type error (hitung dari prediksi yang overlap tapi tidak persis); F1 per tipe langka; string berlabel ganda (1.290, mis. "10 years"); contoh 10 kesalahan.

**Keluaran:** `models/ner/` (model + tokenizer), `extract_entities()` sesuai kontrak, tabel hasil per model.

---

## Task 2 — Classification (PIC 13523007)

**Tujuan:** dari CV, prediksi Field (7), Sub-field (43), Level (Entry/Mid/Senior).

**Data:** HF `ahmedheakl/resume-atlas` (`Category`, `Text`; hanya split `train`).

**Preprocessing:**
1. Dedup exact lalu near-duplicate (200 karakter pertama): 13.389 -> **11.345**. Cek bahwa tiap kelas punya ≥ 3 sampel setelah dedup (syarat stratified split).
2. `Sub-field = Category`. `Field = FIELD_MAP[Category]`: ambil `FIELD_MAP` dari notebook (mapping manual 43 -> 7, **asumsi tim**), pindahkan ke `task2_cls/field_map.py`, dan **minta konfirmasi tim** sebelum dibekukan.
3. Split 80/10/10 stratified by Sub-field (dua kali `train_test_split`), simpan id.
4. **Level:** regex tahun pengalaman, ≈47% CV punya nilai. Perbaiki false positive yang ditemukan validasi manual ("15 years university utah" terbaca Senior): hanya hitung angka yang berada dalam jendela ±5 kata dari `experience|exp|working|professional|career`, ambil nilai terbesar yang lolos. Sisanya `Unknown` (dikeluarkan dari metrik Level, jangan dibuang dari training Field/Sub).
5. **Normalisasi format:** teks ResumeAtlas sudah lowercase dan tanpa tanda baca. Buat satu fungsi `normalize_like_resumeatlas(text)` dan **panggil juga saat inference** pada CV dari PDF. Tanpa ini ada train/inference mismatch.

**Model:**

| Model | Detail |
|---|---|
| Majority class | per output |
| **TF-IDF + Logistic Regression (baseline)** | `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=100_000)`; `LogisticRegression(max_iter=1000, class_weight="balanced")`; tuning `C ∈ {0.1, 1, 10}`. Tiga classifier independen (Field, Sub, Level). `LinearSVC` boleh sebagai pembanding. |
| Hierarchical | classifier Field dulu; untuk Sub-field, **maskir probabilitas** ke sub-field milik Field hasil prediksi (atau satu classifier per Field). |
| **JobBERT multi-output (proposed)** | encoder `jjzha/jobbert-base-cased` bersama + 3 head linear (7, 43, 3). Loss = jumlah cross-entropy; label Level `-100` untuk baris Unknown. `max_length=512` dengan **head+tail truncation** (≈40% CV > 512). lr 2e-5–5e-5, epoch 3–5, batch 16, Colab GPU. |

**Catatan desain yang harus dijelaskan, bukan disembunyikan:**
- **Field adalah fungsi deterministik dari Sub-field** (lewat `FIELD_MAP`). Maka "Field akurat" sebagian besar berarti "Sub-field akurat". Tambahkan baris pembanding: Field langsung vs Field turunan dari Sub-field prediksi.
- **Level = fungsi dari teks yang sama dengan inputnya.** Model ML hanya mempelajari ulang aturan regex. Laporkan Level dari aturan (dari entity `EXPERIENCE` hasil Task 1) dan dari model ML, dan akui sirkularitasnya sebagai limitasi.

**Metrik:** Accuracy Top-1 dan **Top-5** (untuk Sub-field), Precision, Recall, **Macro-F1** (rasio kelas 9,4:1), plus **exact match** (Field, Sub, Level semuanya benar; hanya baris Level-nya terisi).

**Eksperimen train/inference mismatch (inti error analysis):** bandingkan dua input classifier pada test set: (A) teks CV penuh, (B) hanya string entity hasil Task 1 (`SKILL, DESIGNATION, EXPERIENCE, EDUCATION, ...` digabung). B butuh model Task 1, jadi pakai stub dulu. **Peringatan:** JobBERT-cased dilatih pada teks ber-huruf-kapital, sedangkan ResumeAtlas lowercase, jadi NER pada teks ter-lowercase akan menurun. Ukur penurunan itu dan laporkan.

**Error analysis:** confusion matrix antar sub-field mirip (Java/.NET/Python Developer, Web Designing/Designing), validasi manual threshold Level di sekitar 2 dan 5 tahun (sampel 20), kelas dengan F1 terendah.

**Keluaran:** `models/cls/`, `classify_cv()` sesuai kontrak, tabel hasil per output dan per model.

---

## Task 3 — Matching / Ranking (PIC 13523117)

**Tujuan:** menilai kecocokan CV vs job: skor, fit tier, dan ranking.

**Data:** HF `0xnbk/resume-ats-score-v1-en`. Kolom `text` (`resume SEP job`), `ats_score`, `original_label`. Bawaan: 5.099 train + 1.275 validation.

**Fakta penting tentang dataset (dari dataset card):** dataset ini **diturunkan dari `cnamuangtoun/resume-job-description-fit`**, yang dataset card-nya kosong, jadi asal label tidak terdokumentasi. Band skor per tier tidak tumpang tindih (19–30 / 50–60 / 79–90), sehingga `ats_score` pada praktiknya adalah **label tier yang diubah menjadi angka**, bukan penilaian independen. Konsekuensi: regresi skor ≈ prediksi tier, dan MAE akan didominasi kesalahan tier. Tulis ini sebagai limitasi, jangan diklaim sebagai skor ATS riil. Dataset `med2425/resume-job-fit-merged-v1` yang pernah dipertimbangkan juga berasal dari sumber yang sama dan labelnya dibuat LLM, jadi **jangan digabung**.

**Preprocessing:**
1. Gabungkan train+validation (6.374), pisahkan `text` pada regex `\s+SEP\s+` menjadi `resume`, `job`. Buat `job_id` = hash teks job (320 grup) dan `resume_id` (778 unik).
2. **Split per job:** `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` dengan `groups=job_id`, `y=tier`. Satu fold = test, satu fold = val, sisanya train -> sekitar 80/10/10 dengan proporsi tier terjaga. Cetak ukuran tiap split, jumlah job, dan distribusi tier; satu job bisa punya sampai 112 resume, jadi ukuran split akan agak melenceng, dan itu wajar.
3. Karena test hanya berisi ≈ 30 job, **ulangi dengan 3 pilihan fold/seed** dan laporkan rata-rata ± std untuk baseline murah. Untuk cross-encoder cukup 1 split, dengan catatan variansi.
4. Limitasi yang harus dicatat: resume yang sama (778 unik) masih bisa muncul di train dan test.
5. Cache hasil Task 1 dan 2 untuk **teks unik saja** (≈778 resume + 320 job): `data/cache/entities.parquet`, `data/cache/profiles.parquet`. Ini murah dan dibutuhkan untuk ablation.

**Model baseline:**

| Model | Detail |
|---|---|
| **BM25** | `rank-bm25` (`BM25Okapi`). Query = token job, dokumen = resume; IDF dari resume train. Skor mentah tidak sebanding antar-query, jadi gunakan **normalisasi per-job (z-score)** atau kalibrasi (lihat bawah). |
| **SBERT cosine** | `sentence-transformers/all-MiniLM-L6-v2` (max 256 wordpiece, jadi **chunk** teks ≤256, embed per chunk, rata-ratakan). Alternatif untuk teks panjang: `jinaai/jina-embeddings-v2-base-en` (konteks 8k; dipakai penyedia dataset sendiri). Verifikasi id saat pertama kali memuat. |
| Kalibrasi | skor mentah -> skor ATS lewat regresi 1-fitur (atau `IsotonicRegression`) yang **dilatih hanya di train**; tier lewat `LogisticRegression` pada skor itu (untuk probabilitas dan ROC-AUC). Notebook sudah punya pola ini untuk TF-IDF cosine; pakai ulang. |

**Model proposed — Cross-Encoder JobBERT:**
- Backbone `jjzha/jobbert-base-cased`, input pasangan `[CLS] job [SEP] resume [SEP]`.
- **Panjang input adalah masalah utama:** rata-rata resume ≈ 721 dan job ≈ 374 token whitespace, setara ≈ 1,5 ribu wordpiece. Hitung jumlah sebenarnya dengan tokenizer. Strategi: anggaran `job ≤ 192`, `resume ≤ 320`, sisanya padding; truncation per segmen.
- Varian "entities + raw text": teks = `job entities | resume entities | profile(field, sub_field, level)` di depan, diikuti raw text yang dipotong ke sisa anggaran.
- **Dua run terpisah** (lebih sederhana daripada dua head): (A) regresi, `num_labels=1`, target `ats_score/100`, MSE; (B) klasifikasi, `num_labels=3`, cross-entropy. Tier juga bisa diturunkan dari run A lewat threshold sebagai pembanding gratis.
- lr ∈ {2e-5, 3e-5}, epoch 3–4, batch 16, warmup 0,1, fp16, `max_length=512`; early stopping pada val Macro-F1 (run B) atau val MAE (run A). Data train ≈ 5 ribu pasangan, muat di Colab T4.

**Ablation (inti kontribusi tim):**

| Varian | Input cross-encoder |
|---|---|
| V1 | raw text saja |
| V2 | V1 + entity hasil Task 1 |
| V3 | V2 + profile hasil Task 2 |

Ketiganya memakai split, hyperparameter, dan seed yang sama. Gunakan entity/profile **hasil prediksi** model teman (bukan label emas; tidak ada label emas untuk teks ini), sehingga ablation sekaligus mengukur error propagation. Selama model teman belum jadi, pakai stub dan tandai hasilnya `(stub)`.

**Metrik:**
- Tier: Macro-F1, ROC-AUC (macro one-vs-rest), per-tier P/R/F1, confusion matrix.
- Skor: MAE dan Spearman (keseluruhan, plus rata-rata Spearman per-job).
- Ranking: per job di test yang punya ≥ 3 resume, urutkan resume menurut skor prediksi; **NDCG@5** (relevansi = tier 0/1/2 atau skor asli, pilih satu dan catat) dan **MRR** (peringkat resume Good Fit pertama).
- Baseline "tebak No Fit terus" menghasilkan akurasi 50%, maka laporkan Macro-F1, bukan accuracy.

**Error analysis:** (1) false positive: Good Fit diprediksi untuk pasangan No Fit; periksa apakah skill sama tetapi seniority beda dengan membandingkan `level` CV (Task 2) terhadap tahun pengalaman yang diminta job (entity `EXPERIENCE` job); (2) error per Field; (3) efek truncation (akurasi vs panjang teks); (4) error propagation: hasil dengan entity prediksi vs stub; (5) 10 contoh kesalahan dengan teks ringkas.

**Keluaran:** `models/match/` , `score_match()` dan `rank()` sesuai kontrak, tabel hasil + ablation, chart (confusion matrix, skor prediksi vs asli, NDCG per job).

---

## Pipeline & CLI (dikerjakan bersama)

`main.py`:
```
python main.py --cv examples/cv1.pdf --job examples/job1.txt [--verbose]
python main.py --cv-dir examples/cvs/ --job examples/job1.txt   # mode ranking banyak CV
```
Alur: PDF -> teks (`pdfplumber`, fallback `pypdf`; CV dua kolom bisa tertukar urutannya, cek manual contoh demo) -> `extract_entities` (CV dan job) -> `normalize_like_resumeatlas` + `classify_cv` -> `score_match` -> cetak entities, profile, skor, tier (verbose), atau tabel ranking.

Syarat: berjalan di CPU dengan model tersimpan, tanpa training ulang; waktu < 1 menit per pasangan; pesan error jelas bila PDF kosong atau hasil scan (OCR di luar scope). Siapkan 3–5 pasangan contoh yang beragam (1 cocok, 1 tidak cocok, 1 beda bidang) untuk demo.

## Urutan kerja yang disarankan

1. Hari 1: sepakati kontrak (bagian 0), buat stub dan `test_contracts.py`, bekukan `FIELD_MAP`.
2. Paralel: Task 1, 2, 3 mengerjakan baseline masing-masing dengan split yang disimpan.
3. Model proposed (JobBERT) dilatih di Colab; simpan checkpoint ke `models/`.
4. Task 3 menjalankan cache entity/profile pada ≈1.100 teks unik begitu model teman siap, lalu ablation V1–V3.
5. Integrasi CLI, uji dengan contoh demo, tulis error analysis untuk slide.
