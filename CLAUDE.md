# CV-Job Matching (IF5153 NLP, Group 6) — Project Brief untuk Claude Code

Taruh file ini di root repo sebagai `CLAUDE.md`. Baca seluruhnya sebelum menulis kode.

## 1. Konteks

Tugas besar IF5153 Natural Language Processing (ITB). Satu tim = 3 mahasiswa, satu pipeline berisi 3 NLP task, **tiap mahasiswa memegang 1 task dan melakukan eksperimen lengkap pada task itu**.

| Task | NLP category | PIC (NIM) |
|---|---|---|
| 1. Skill/Requirement Extraction | Sequential labeling (NER) | 13523065 |
| 2. Qualification Classification | Text classification | 13523007 |
| 3. Matching Score / Ranking | Text comparison | 13523117 (Ferdin, pemilik repo ini) |

**Deliverable akhir (presentasi minggu 14–15):**
- Demo: menjalankan model NLP **via command prompt**, dengan beberapa input berbeda.
- PPT: penjelasan algoritma dan hasil eksperimen **termasuk error analysis**.

Bahasa kerja: dokumen dan slide berbahasa Inggris, diskusi tim berbahasa Indonesia. Komentar kode dan README boleh bahasa Inggris.

## 2. Pipeline (sudah final, jangan diubah)

```
INPUT: CV (PDF) + Job Description (teks)
  -> Preprocessing: CV PDF -> teks (pdfplumber/PyPDF2)
  -> Task 1 NER: teks CV dan teks Job Desc -> entity (SKILL, EXPERIENCE, EDUCATION, DESIGNATION, ...)
  -> Task 2 Classification (HANYA CV): entity/teks CV -> Field, Sub-field, Seniority Level
  -> Task 3 Matching: (CV entities + CV profile) vs (Job entities) -> matching score / fit tier / ranking
OUTPUT: matching score (+ fit tier) dan ranking kandidat untuk satu lowongan
```

Keputusan yang sudah diambil: Job Description **hanya** diproses sampai NER (tidak diklasifikasi Field/Level). Ini disengaja.

## 3. Dataset (sudah diverifikasi dari notebook, angka ini yang dipakai)

| Task | Dataset | Fakta terverifikasi |
|---|---|---|
| NER | Kaggle `yashpwrr/resume-ner-training-dataset` | 5.960 CV, 579.575 anotasi, 13 tipe entity + `OTHER` (di-drop). SKILL ≈ 95% anotasi. Hanya 72% teks CV unik. 42% CV > 512 token. Label bernama `DESIGNATION`, **bukan** `JOB_TITLE`. |
| NER (job side) | Kaggle `khaliladimassi/cleaned-job-description` | **Tidak punya anotasi NER** (kolom: job_title, company_name, job_description, clean_text). Hanya dipakai untuk cek domain shift (densitas entity), **tidak bisa dihitung F1-nya**. Klaim "160.000 posting teranotasi" di laporan BEAM salah/tidak terverifikasi. |
| Classification | HF `ahmedheakl/resume-atlas` | 13.389 raw -> **11.345** setelah dedup exact + near-dup (200 char awal), -15,3%. 43 kategori = Sub-field. Field = mapping manual 43 -> 7 grup (`FIELD_MAP`, asumsi tim). Level = regex "N years" (<2 Entry, 2–5 Mid, >5 Senior), hanya tersedia untuk ~47% CV dan sebagian salah tangkap. Rasio kelas terbanyak/tersedikit 9,4. |
| Matching | HF `0xnbk/resume-ats-score-v1-en` | **6.374** pasangan (5.099 train + 1.275 val), bukan 6.400. Hanya **778 resume unik dan 320 job unik**. Tier: No Fit 3.206 (50%), Potential 1.592, Good Fit 1.576. Skor ATS 19–90, band tier tidak tumpang tindih (19–30 / 50–60 / 79–90). Spearman keyword-overlap vs skor hanya 0,12. **Dataset card menyatakan dataset ini diturunkan dari `cnamuangtoun/resume-job-description-fit`** (card dataset asalnya kosong), jadi skor ATS pada praktiknya adalah label tier yang diubah jadi angka, bukan skor independen. |

Catatan: kolom `text` dataset Matching berisi resume dan job digabung dengan pemisah `SEP`.

## 4. Desain eksperimen (ini yang sudah dipresentasikan di slide, implementasi harus mengikutinya)

**Prinsip umum:** tiap task dievaluasi terpisah terhadap baseline. Baseline dan proposed model memakai split dan preprocessing yang sama. Split **80/10/10 (train/val/test)**, tuning di validation, hold-out test, fixed seed (42). Dedup **sebelum** split.

### Task 1 — NER
- Pendekatan: rule + gazetteer (lower bound) -> **CRF (baseline)** -> **fine-tuned JobBERT (proposed)**.
- Preprocessing: regex tokenization -> BIO; span > 60 karakter di-skip; label `OTHER` di-drop; CV > 512 token di-chunk (sliding window); dedup by CV text lalu split di level dokumen.
- Metrik: entity-level (exact span) Precision/Recall/F1 **per entity type** (+ micro). Wajib per-tipe karena SKILL mendominasi.
- Tuning: regularisasi CRF; learning rate dan epoch BERT.
- Error analysis: boundary vs type error, tipe langka, overlapping span (67k) dan string berlabel ganda (1,3k). Domain shift CV vs job desc lewat densitas entity.

### Task 2 — Classification
- Output: Field (7), Sub-field (43), Level (Entry/Mid/Senior).
- Pendekatan: majority class -> **TF-IDF + Logistic Regression (baseline)** -> hierarchical Field -> Sub-field -> **fine-tuned JobBERT multi-output head (proposed)**.
- Preprocessing: dedup (sudah), stratified split by Sub-field 80/10/10.
- Metrik: Accuracy (Top-1, Top-5), Precision, Recall, **Macro-F1**, plus exact-match ketiga output.
- Error analysis: bandingkan input **teks CV mentah vs output NER saja** (train/inference mismatch: di pipeline Task 2 hanya menerima hasil NER); manual check threshold seniority (2 dan 5 tahun); confusion antar sub-field mirip (Java vs .NET Developer).

### Task 3 — Matching (tugas Ferdin, prioritas tertinggi)
- Pendekatan: **SBERT cosine** dan **BM25** (baseline) -> **Cross-Encoder dengan backbone JobBERT (proposed)**, input = entities + raw text.
- **Ablation:** raw text saja -> + NER entities -> + NER + classification. Tujuannya mengukur kontribusi Task 1 dan Task 2 ke matching.
- Preprocessing: pisahkan `text` pada `SEP` menjadi resume dan job. **Split per job** (GroupShuffleSplit / group split 80/10/10) supaya job di test tidak pernah muncul di train. Catat sebagai limitasi: resume yang sama masih bisa muncul di train dan test.
- Metrik: Macro-F1 dan ROC-AUC (fit tier), MAE dan Spearman (ATS score), NDCG@5 dan MRR (ranking; ranking per job dengan >= 3 resume di test).
- Error analysis: false positive karena skill sama tapi seniority beda; **error propagation** (umpan hasil *prediksi* NER dan Field, bukan gold label); limitasi: tier/skor adalah band yang kemungkinan dihasilkan otomatis, bukan penilaian recruiter.

## 5. State repo saat ini (cek ulang sebelum mulai, mungkin sudah berubah)

File utama: `cv_job_matching_all.ipynb` (104 cell: EDA Task 1–3, lalu Eksperimen Task 1–3; chart ke `../figures/`, tabel ke `../data/`).

Terakhir diperiksa:
- **EDA Task 1–3: sudah jalan** (kecuali sub-bagian job description NER, kena `KeyboardInterrupt`, tidak perlu diulang).
- **Eksperimen:** hanya baseline rule+gazetteer NER yang punya output (micro-F1 ≈ 0,51; hampir semua nilai berasal dari SKILL, tipe lain ≈ 0). CRF, BERT, Classification, dan Matching **belum dijalankan**.
- **Ketidaksesuaian notebook vs slide (harus diselaraskan, jangan sebaliknya):**
  1. Notebook memakai split 80/20 tanpa validation. Slide menyatakan 80/10/10.
  2. Notebook NER belum melakukan dedup sebelum split. Slide menyatakan dedup.
  3. Notebook memakai `dslim/bert-base-NER`, Ridge, LogReg, dan TF-IDF cosine. **JobBERT, SBERT, BM25, dan Cross-Encoder belum ada sama sekali.**
  4. Classification: notebook mengevaluasi 3 classifier independen dan hierarchical, tetapi belum ada majority-class baseline, Top-5 accuracy, maupun JobBERT multi-output.

## 6. Yang diminta dikerjakan (urutan prioritas)

1. **Task 3 dulu** (PIC Ferdin): implementasikan baseline BM25 dan SBERT cosine, split per job 80/10/10, metrik lengkap di atas, lalu Cross-Encoder JobBERT dan ablation tiga varian. Simpan hasil ke `../data/` sebagai CSV dan chart ke `../figures/`.
2. Selaraskan notebook dengan slide: split 80/10/10 di Task 1 dan 2, dedup NER, majority-class baseline dan Top-5 di Task 2.
3. Implementasi CRF dan JobBERT untuk NER; JobBERT multi-output untuk Task 2 (dua ini PIC rekan satu tim, kerjakan hanya bila diminta atau skeleton-nya saja).
4. **CLI demo end-to-end** (`main.py`): argumen `--cv path.pdf --job "teks atau path.txt"`. Mencetak hasil tiap tahap (entities, profile, matching score dan tier) dengan flag `--verbose`. Harus berjalan di CPU dengan model terlatih yang sudah disimpan (`models/`), tanpa training ulang. Siapkan 3–5 contoh pasangan CV-job di `examples/`.
5. `requirements.txt` yang pin versi, dan README singkat cara menjalankan notebook dan CLI.

## 7. Aturan kerja

- **Jujur soal hasil.** Jangan menulis angka metrik atau klaim "sudah dieksekusi" yang tidak benar-benar keluar dari run. Jika sebuah model belum dilatih, tandai `(planned)` di output dan laporan.
- Jangan mengganti dataset, pipeline, atau metrik yang tercantum di bagian 3 dan 4 tanpa bertanya. Jika ada alasan kuat (misalnya model tidak tersedia), usulkan alternatif dan jelaskan, jangan diam-diam mengganti.
- JobBERT = `jjzha/jobbert-base-cased` (terverifikasi ada; BERT-base-cased yang dilanjutkan pre-training pada ±3,2 juta kalimat job posting, bukan model NER siap pakai). Model ini **cased**, sedangkan teks ResumeAtlas lowercase tanpa tanda baca; perhatikan mismatch ini. SBERT kandidat `sentence-transformers/all-MiniLM-L6-v2` belum diverifikasi, cek saat memuat; jika gagal, beri tahu.
- Detail implementasi per task (kontrak antarmuka, hyperparameter, langkah preprocessing) ada di `TASKS.md`.
- Tidak ada kebocoran data: dedup sebelum split, split per job di Matching, jangan fit TF-IDF/vocab pada data test.
- Fixed seed 42 di semua tempat. Hasil harus reproducible.
- Eksperimen berat (fine-tuning BERT) dirancang untuk Colab GPU; sediakan opsi sampel kecil (`RUN_BERT`/`SAMPLE_N`) supaya bisa dites di CPU.
- Dataset Kaggle butuh `kaggle.json`; Hugging Face boleh pakai `HF_TOKEN` agar lebih cepat. Jika kredensial tidak ada, berhenti dan minta, jangan membuat data tiruan secara diam-diam.
- Kode sederhana dulu. Jangan menambah abstraksi atau framework yang tidak diperlukan.

## 8. Definition of done (untuk presentasi)

- [ ] Task 3: tabel hasil baseline vs proposed vs ablation, plus chart dan 3–5 contoh error.
- [ ] Notebook konsisten dengan slide (split, dedup, model yang tercantum), dan seluruh cell bisa di-Run All.
- [ ] `python main.py --cv examples/cv1.pdf --job examples/job1.txt` menghasilkan skor, tier, dan entities.
- [ ] README menjelaskan batasan: job desc tanpa gold NER, Field = mapping manual, Level = heuristik (cakupan 47%), tier = band skor.
