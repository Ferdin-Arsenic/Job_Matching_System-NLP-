# Urutan Slide PPT Bersama: CV-Job Matching (IF5153 NLP, Group 6)

Alur mengikuti pipeline: Task 1 (NER) → Task 2 (Classification) → Task 3 (Matching) → gabungan → demo → penutup.
Pembagian pembicara hanya usulan, perlu disepakati tim.

| # | Slide | Isi | Pembicara |
|---|---|---|---|
| **Pembuka** | | | |
| 1 | Judul | Judul proyek, tim, IF5153 | Semua |
| 2 | Masalah & tujuan | Kenapa screening CV manual bermasalah, apa yang dibangun | Bebas |
| 3 | Pipeline end-to-end | Diagram: CV PDF → NER → Classification → Matching → skor, tier, ranking (job desc hanya sampai NER, disengaja) | Bebas |
| 4 | Ringkasan dataset | Satu tabel: 3 dataset, ukuran, sumber, fakta kunci | Bebas |
| 5 | Prinsip eksperimen | Split 80/10/10, seed 42, dedup sebelum split, baseline vs proposed | Bebas |
| **Task 1: NER** | | | |
| 6 | Dataset & EDA | 5.960 CV, SKILL ≈ 95%, 42% CV > 512 token | PIC 13523065 |
| 7 | Preprocessing & model | Regex → BIO, chunking; rule+gazetteer → CRF → JobBERT | PIC 13523065 |
| 8 | Hasil | F1 per tipe entity dan micro, baseline vs proposed | PIC 13523065 |
| 9 | Error analysis | Boundary vs type error, tipe langka, domain shift CV vs job desc | PIC 13523065 |
| **Task 2: Classification** | | | |
| 10 | Dataset & EDA | 11.345 CV setelah dedup, 43 sub-field, 7 field, level | PIC 13523007 |
| 11 | Preprocessing & model | FIELD_MAP, heuristik level; majority → TF-IDF+LR → hierarchical → JobBERT | PIC 13523007 |
| 12 | Hasil | Top-1/Top-5, Macro-F1, exact-match | PIC 13523007 |
| 13 | Error analysis | Teks mentah vs output NER, threshold seniority, confusion antar sub-field | PIC 13523007 |
| **Task 3: Matching** | | | |
| 14 | Dataset & EDA | 6.374 pasangan, tier, keyword-overlap lemah (Spearman 0,12) | Ferdin |
| 15 | Preprocessing & model | Split per job; BM25 dan SBERT → Cross-Encoder JobBERT | Ferdin |
| 16 | Ablation | V1 / V2 / V3 (raw → + NER → + NER + klasifikasi) | Ferdin |
| 17 | Hasil | Tabel baseline vs proposed vs ablation, dengan CI | Ferdin |
| 18 | Error analysis | 3–5 contoh error, resume yang sama di train dan test | Ferdin |
| **Gabungan** | | | |
| 19 | Kontribusi antar task | Apa yang diberikan NER dan Classification ke matching (dari ablation) | Ferdin / bergantian |
| 20 | Error propagation | Input Task 1/Task 2 ke model demo sengaja dibuat salah di test: entity salah → ROC-AUC turun ke 0,51 (≈ acak), profil salah → hanya −0,03. Tabel di README bagian *Error propagation*, data `data/results/task3_error_propagation.csv` | Ferdin |
| 20b | Perbaikan: stacking | Cross-encoder jenuh (CV backend 1,77 vs web designer 1,82 vs guru 1,64). Digabung dengan sinyal kecocokan eksplisit (skill, jabatan, SBERT): Macro-F1 0,388 → 0,434, MRR 0,742 → 0,864 (rata-rata 3 seed, naik di 3/3 seed; CI seed 42 masih mencakup 0). Dipakai demo saat meranking ≥2 CV; satu CV tetap cross-encoder saja. Sumber: README bagian *Additional experiment*, `data/results/task3_stacking_*.csv` | Ferdin |
| 21 | Tabel ringkasan semua task | Satu baris per task: baseline, proposed, metrik utama | Bebas |
| **Demo** | | | |
| 22 | Cara menjalankan | `python main.py --cv ... --job ... --verbose`, input dan output | Bebas |
| 23 | Demo live | 2–3 input berbeda via command prompt (siapkan rekaman cadangan, ±20 detik per CV) | Bebas |
| **Penutup** | | | |
| 24 | Limitasi | Job desc tanpa gold NER, Field = mapping manual, Level = heuristik (cakupan 47%), tier = band skor, test set kecil (26 job; cross-encoder 3 seed) | Bebas |
| 25 | Kesimpulan & future work | Poin per task | Semua |
| 26 | Kontribusi anggota | Siapa mengerjakan apa | Semua |
| 27 | Q&A | | Semua |
| **Appendix** | | | |
| A1+ | Cadangan Q&A | Hyperparameter, tabel per tipe entity, confusion matrix lengkap | |

## Catatan

- **Demo setelah hasil**: penonton sudah paham tiap tahap sehingga output demo mudah dibaca. Jika dosen lebih suka demo di awal, pindahkan slide 22–23 ke setelah slide 4 dan ulang singkat di akhir.
- **Slide 20 (error propagation)** sudah dikerjakan (`task3_match/error_propagation.py`). Sebutkan batasannya: tidak ada label gold di dataset matching, jadi yang diukur adalah sensitivitas terhadap output upstream yang salah total, bukan error nyata model Task 1/Task 2.
- **Waktu 15–20 menit**: gabungkan slide 6–7, 10–11, 14–15 sehingga jadi ±22 slide. Slide error analysis (9, 13, 18) tetap terpisah karena wajib.
- **Angka di slide** harus diambil dari file hasil (`data/*.csv`, `data/results/`), bukan dari CLAUDE.md. Model yang belum dijalankan ditandai `(planned)`.
