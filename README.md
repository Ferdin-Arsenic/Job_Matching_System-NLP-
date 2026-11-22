# CV-Job Matching (IF5153 NLP, Group 6)

A three-stage NLP pipeline that scores how well a CV fits a job description and ranks candidates for one job.

```
CV (PDF/TXT) + job description (TXT or text)
  -> Task 1  NER             entities from the CV and the job (SKILL, JOB_TITLE, EDUCATION, EXPERIENCE)
  -> Task 2  Classification  CV only: Field, Sub-field, Level (Entry / Mid / Senior / Unknown)
  -> Task 3  Matching        cross-encoder: score 0-100, fit tier (No Fit / Potential Fit / Good Fit), ranking
```

| Task | What | Model | Weights |
|---|---|---|---|
| 1. NER | extract entities | JobBERT + CRF | [WwzFwz/jobbert-crf-cv-jd-ner](https://huggingface.co/WwzFwz/jobbert-crf-cv-jd-ner) |
| 2. Classification | Field / Sub-field / Level | JobBERT multi-output | [ritesaph/jobbert-task2-cls](https://huggingface.co/ritesaph/jobbert-task2-cls) |
| 3. Matching | fit score, tier, ranking | JobBERT cross-encoder (V3: text + entities + profile) | [Arxenarendra/jobbert-task3-match](https://huggingface.co/Arxenarendra/jobbert-task3-match) |

All JobBERT models start from [`jjzha/jobbert-base-cased`](https://huggingface.co/jjzha/jobbert-base-cased).

## Run the demo

Python 3.12, CPU only, no training needed.

```bash
pip install -r requirements.txt

# one CV against one job (add --verbose for entities and sub-field probabilities)
python main.py --cv examples/cvs/cv1.pdf --job examples/job1.txt --verbose

# rank every CV in a folder for one job
python main.py --cv-dir examples/cvs --job examples/job1.txt
```

`--job` takes a `.txt` path or the job text itself; `--cv` takes `.pdf` or `.txt`.

- **First run downloads about 1.3 GB** of weights (Task 1 ~430 MB, Task 2 ~415 MB, Task 3 ~433 MB) into `models/`. It took about 5 minutes on one connection; later runs reuse them.
- After that a run takes about 20 s for one CV and about 25-30 s for five CVs on a laptop CPU (measured, including model loading).
- Scanned or empty PDFs stop with an explicit error (OCR is out of scope). Two-column CVs may be read in the wrong order.

### Example output (real runs, `examples/`)

`examples/` holds 2 jobs and 5 CVs. **All of them are synthetic**, written for the demo, and are not an evaluation set.
`cv1` is a backend developer, `cv2` a junior web designer, `cv3` a mechanical engineer, `cv4` an accountant, `cv5` a school teacher.

```
python main.py --cv-dir examples/cvs --job examples/job2.txt        (job2 = Staff Accountant)
#  CV        Score  Tier      Field / Sub-field / Level
1  cv4.pdf    50.9  Potential Fit  Finance / Accountant / Senior
2  cv5.pdf    42.8  No Fit    Education & Health / Education / Senior
3  cv2.pdf    25.3  No Fit    IT / Web Designing / Mid
4  cv3.pdf    25.3  No Fit    Engineering & Construction / Mechanical Engineer / Senior
5  cv1.pdf    24.5  No Fit    IT / Python Developer / Senior

python main.py --cv-dir examples/cvs --job examples/job1.txt        (job1 = Backend Software Engineer)
1  cv2.pdf    74.9  Good Fit  IT / Web Designing / Mid
2  cv1.pdf    64.8  Good Fit  IT / Python Developer / Senior
3  cv5.pdf    55.3  Good Fit  Education & Health / Education / Senior
4  cv3.pdf    50.9  No Fit    Engineering & Construction / Mechanical Engineer / Senior
5  cv4.pdf    34.9  No Fit    Finance / Accountant / Senior
```

Read these honestly: for the accountant job the order is sensible. For the backend job the model ranks the junior web
designer (`cv2`) above the backend developer (`cv1`) and calls the teacher (`cv5`) a Good Fit, which is wrong. The
test-set numbers below (Good Fit precision of about 0.4) say the same: tier calls are unreliable, so use the **ranking
within one job** rather than the absolute tier or score. Scores come from a step function, so different CVs can share
the same score; the ranking breaks ties with the expected tier.

## Results (Task 3, test set, seed 42, 26 jobs)

One split, one run. 95% bootstrap intervals over jobs are in `data/results/task3_bootstrap.csv`.

| Model | Macro-F1 | ROC-AUC | MAE | NDCG@5 | MRR |
|---|---|---|---|---|---|
| Majority class | 0.222 | 0.500 | 22.5 | 0.590 | 0.540 |
| Resume prior (guess from the CV alone, ignores the job) | 0.367 | 0.602 | 20.8 | 0.588 | 0.545 |
| BM25 | 0.241 | 0.565 | 22.0 | 0.543 | 0.670 |
| SBERT cosine | 0.321 | 0.584 | 21.5 | 0.640 | 0.803 |
| Cross-encoder V1 (raw text) | 0.353 | 0.590 | 21.8 | 0.659 | 0.867 |
| Cross-encoder V2 (+ Task 1 entities) | 0.399 | 0.637 | 20.2 | 0.682 | 0.833 |
| Cross-encoder V3 (+ Task 2 profile, baseline profiles) | 0.481 | 0.685 | 19.7 | 0.733 | 0.883 |
| Cross-encoder V3 (+ Task 2 profile, JobBERT profiles) | 0.379 | 0.635 | 20.2 | 0.636 | 0.792 |

- The two V3 rows differ only in where the Task 2 profile comes from: the baseline profiles (`profiles_t2.parquet`,
  used for training the published weights) or the final JobBERT Task 2 model (`profiles_jobbert.parquet`, what `main.py`
  produces). The two sources agree on Sub-field for only 54% of CVs.
- With baseline profiles, V3 beats V1 and SBERT on Macro-F1 (the intervals exclude 0). **With the JobBERT profiles this
  advantage disappears**: Macro-F1 0.379 is within noise of V1 (0.353) and SBERT (0.321). Both V3 variants are
  statistically indistinguishable from each other (paired bootstrap, intervals include 0), so we cannot claim the profile
  helps in general; it helped only when the profile source matched the one seen in training.
- **Not** supported: V2 over V1 and V3 over V2 (the intervals include 0); read those as "tends to improve".

Re-running with two more split/training seeds (43, 44; each seed has a different test set, so compare within a seed;
`python task3_match/seed_check.py`, runs in `data/results/task3_seed_check*.csv`). Macro-F1 on test:

| Model | seed 42 | seed 43 | seed 44 | mean ± std |
|---|---|---|---|---|
| V1 (raw text) | 0.353 | 0.319 | 0.392 | 0.355 ± 0.036 |
| V2 (+ entities) | 0.399 | 0.441 | 0.412 | 0.418 ± 0.021 |
| V3 (+ profile, baseline profiles) | 0.481 | 0.450 | 0.386 | 0.439 ± 0.048 |
| V3 (+ profile, JobBERT profiles) | 0.379 | 0.370 | 0.414 | 0.388 ± 0.023 |

- On average V1 is lowest and V2/V3 are higher, but the gaps (0.03-0.08) are about one standard deviation. With three
  seeds we can say that adding Task 1 entities *tends* to help; we cannot rank V2 against either V3.
- The best V3 (baseline profiles) was first on seeds 42 and 43 and last on seed 44, so its lead over V1 on seed 42 was
  partly luck of the split. Ranking metrics are even noisier (MRR std up to 0.18 over about 26 jobs per test set).
- The published weights (`Arxenarendra/jobbert-task3-match`) are the JobBERT-profile row (the second V3 row), so the demo
  feeds the model the same kind of profile it was trained on. The baseline-profile weights stay on Hugging Face under tag
  `v3-baseline-profiles`.
- Task 1 (reported in its notebook): entity F1 on CV test 0.47 exact / 0.60 partial, on JD test 0.57 / 0.80.
- Task 2: JobBERT was chosen on validation over the TF-IDF + Logistic Regression baselines; see
  `notebooks/task2_classification.ipynb` and the Hugging Face repo for its results.
- Full table: `data/results/task3_final_seed42.csv`; error analysis: `data/results/task3_error_*.csv`; charts are
  produced by `task3_match/analysis.py` into `figures/`.

## Limitations

- **Job descriptions have no gold NER labels** in the dataset we planned for (`khaliladimassi/cleaned-job-description`),
  so there is no F1 for the job side from it. Task 1 evaluates JD extraction on a separate labelled JD corpus instead.
- **Field is a manual mapping** of the 43 ResumeAtlas categories to 7 groups (a team assumption), so Field accuracy is
  mostly Sub-field accuracy.
- **Level is a heuristic**: years of experience found by Task 1 (<2 Entry, 2-5 Mid, >5 Senior, otherwise Unknown). Only
  about half of CVs state experience, and the estimate can exceed the claimed years (it merges overlapping date ranges).
- **Tier and score are not independent recruiter judgements.** The matching dataset (`0xnbk/resume-ats-score-v1-en`) is
  derived from another dataset whose labels are undocumented; the score bands do not overlap (19-30 / 50-60 / 79-90), so the
  "score" is effectively a tier turned into a number. Predicted scores stay mostly between 25 and 74.
- **CV overlap between train and test:** 95-97% of test pairs use a CV that also appears in training (the split is per
  job, so jobs never overlap). A baseline that guesses the tier from the CV alone reaches Macro-F1 0.37, so Macro-F1
  alone overstates matching ability; the per-job ranking metrics are the more honest ones.
- Small test set (26 jobs, 23 with at least 3 CVs) and a single seed. Entities from ATS-style text can contain words
  glued together (for example `analystlocation`).
- The demo CVs and jobs in `examples/` are synthetic and the "expected" fits in `scripts/make_examples.py` are the
  authors' judgement.

## Repository layout

```
main.py                 CLI demo
pipeline/               PDF/TXT reading and the end-to-end flow
task1_ner/  task2_cls/  task3_match/     one folder per task (task1_ner/api.py wraps scripts/t1_ner_predict.py)
common/                 shared schemas and validators
scripts/                Task 1 inference/export, Task 2 profile export, make_examples.py, upload_task3_model.py
notebooks/              Task 1, 2, 3 experiments and the EDA notebook; Colab notebook for the cross-encoder
data/splits, data/cache, data/results    fixed splits, cached profiles, result tables (small files, in git)
outputs/ner/            Task 1 outputs for the 778 resumes and 320 jobs used by Task 3
models/                 downloaded weights (not in git)
```

## Reproduce Task 3

```bash
python task3_match/baselines.py     # majority, resume prior, BM25, SBERT over 3 split seeds (CPU)
python task3_match/analysis.py      # bootstrap intervals, error analysis, charts, final table (CPU)
```

The cross-encoder runs need a GPU: open `notebooks/task3_crossencoder_colab.ipynb` in Colab (it clones the repo with a
`GH_TOKEN` secret, runs 8 fine-tuning runs, and saves results to Google Drive). Seed 42 everywhere; the per-job splits
are stored in `data/splits/task3.json`.

## Team

| Task | Owner (NIM) |
|---|---|
| 1. NER | 13523065 |
| 2. Classification | 13523007 |
| 3. Matching | 13523117 |
