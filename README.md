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

- **First run downloads about 1.4 GB** of weights (Task 1 ~430 MB, Task 2 ~415 MB, Task 3 ~433 MB, SBERT ~90 MB) into `models/` and the Hugging Face cache. It took about 5 minutes on one connection; later runs reuse them.
- After that a run takes about 20 s for one CV and about 25-30 s for five CVs on a laptop CPU (measured, including model loading).
- Scanned or empty PDFs stop with an explicit error (OCR is out of scope). Two-column CVs may be read in the wrong order.

### Example output (real runs, `examples/`)

`examples/` holds 2 jobs and 5 CVs. **All of them are synthetic**, written for the demo, and are not an evaluation set.
`cv1` is a backend developer, `cv2` a junior web designer, `cv3` a mechanical engineer, `cv4` an accountant, `cv5` a school teacher.

```
python main.py --cv-dir examples/cvs --job examples/job2.txt        (job2 = Staff Accountant)
#  CV        Score  Tier           Field / Sub-field / Level
1  cv4.pdf    65.9  Potential Fit  Finance / Accountant / Senior
2  cv5.pdf    34.4  No Fit         Education & Health / Education / Senior
3  cv2.pdf    25.5  No Fit         IT / Web Designing / Mid
4  cv3.pdf    25.4  No Fit         Engineering & Construction / Mechanical Engineer / Senior
5  cv1.pdf    24.5  No Fit         IT / Python Developer / Senior

python main.py --cv-dir examples/cvs --job examples/job1.txt        (job1 = Backend Software Engineer)
1  cv1.pdf    65.9  Good Fit       IT / Python Developer / Senior
2  cv2.pdf    56.2  Good Fit       IT / Web Designing / Mid
3  cv5.pdf    54.3  Potential Fit  Education & Health / Education / Senior
4  cv3.pdf    44.5  No Fit         Engineering & Construction / Mechanical Engineer / Senior
5  cv4.pdf    25.5  No Fit         Finance / Accountant / Senior
```

Read these honestly: both orders are sensible, but tier calls are still loose (the junior web designer is a Good Fit
for the backend job, the teacher a Potential Fit), so use the **ranking within one job** rather than the absolute tier
or score. Before the stacker (cross-encoder alone) the backend job ranked the web designer above the backend developer
and called the teacher a Good Fit. With `--cv` (one CV) the cross-encoder alone is used, because the stacker compares
candidates with each other; one CV alone gets the same result as before (cv1 vs job1: 64.8, Good Fit). Scores come from a step function, so different CVs can share
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
- Full table: `data/results/task3_final_seed42.csv`; error analysis of the demo model (V3 with JobBERT profiles):
  `data/results/task3_error_*.csv` and `task3_confusion_V3jb.csv`; charts are produced by `task3_match/analysis.py`
  into `figures/`. The demo model over-predicts Good Fit: 94 of 328 true No Fit pairs are called Good Fit (44 for the
  baseline-profile V3), and only 2% of those false positives involve a CV that is too junior for the job.

### Error propagation (how much Task 1 and Task 2 errors hurt matching)

The matching dataset has no gold NER or gold Field for its CVs, so we cannot compare predicted against gold upstream
labels. Instead the demo model stays fixed and its upstream input is made wrong on the test set: each CV (or job) gets the
Task 1 entities or Task 2 profile of another random one, for 50% or 100% of them (`task3_match/error_propagation.py`,
CPU; the clean run reproduces the Colab predictions, 99.8% same tier). Change versus clean input, 95% paired bootstrap
interval over jobs in brackets:

| Upstream input made wrong | Macro-F1 | ROC-AUC | NDCG@5 | Tier predictions changed |
|---|---|---|---|---|
| None (clean) | 0.379 | 0.635 | 0.642 | - |
| Task 2 profile, 50% of CVs | -0.011 [-0.033, 0.011] | -0.003 [-0.020, 0.013] | +0.040 [-0.005, 0.092] | 9% |
| Task 2 profile, all CVs | -0.008 [-0.046, 0.033] | **-0.026** [-0.046, -0.003] | +0.039 [-0.021, 0.105] | 16% |
| Task 1 entities, 50% of CVs and jobs | **-0.056** [-0.116, -0.006] | **-0.089** [-0.177, -0.009] | -0.023 [-0.154, 0.092] | 28% |
| Task 1 entities, all | -0.067 [-0.190, 0.047] | **-0.125** [-0.224, -0.036] | -0.096 [-0.279, 0.058] | 42% |
| Both, all | -0.085 [-0.198, 0.023] | **-0.140** [-0.233, -0.060] | -0.145 [-0.320, 0.013] | 42% |

- **Task 1 errors propagate**: wrong entities push ROC-AUC down to chance (0.51) and change 28-42% of tier calls.
- **Task 2 errors barely do**: a completely wrong profile moves ROC-AUC by -0.03 and leaves Macro-F1 and ranking within
  noise. The model has learned to rely little on the profile, which matches the ablation (V3 is not reliably above V2).
  Consistently, clean accuracy is not lower where the two Task 2 models disagree on Sub-field (0.46, 348 pairs) than
  where they agree (0.42, 308 pairs).
- Limits: a swap is a worst-case error (another CV's output), not the partial errors a real model makes; and the model
  was trained on predicted, already noisy, upstream outputs, which may be why it discounts the profile.

### Stacking with explicit match signals (used by the demo when ranking two or more CVs)

The demo cross-encoder saturates: most CVs get an expected tier near the top whatever their content (on `examples/`,
job1 gives the backend CV 1.77, the web designer 1.82 and the teacher 1.64 out of 2). `task3_match/stacking.py` fits a
logistic regression on the validation rows that combines the cross-encoder probabilities with three explicit signals:
share of the job's skills found in the CV, job-title overlap, and SBERT cosine, each also relative to the other
candidates of the same job. No BERT model is retrained. Test set:

| Mean over seeds 42/43/44 | Macro-F1 | ROC-AUC | NDCG@5 | MRR |
|---|---|---|---|---|
| Cross-encoder only | 0.388 | 0.656 | 0.689 | 0.742 |
| Cross-encoder + match signals (demo ranking) | 0.434 | 0.670 | 0.717 | 0.864 |

- Macro-F1, NDCG@5 and MRR improve on all three seeds; ROC-AUC on two of three.
- On seed 42 alone the paired bootstrap intervals still include 0 (Macro-F1 +0.06 [-0.01, +0.14], MRR +0.10 [0.00, +0.24]),
  so this is a consistent direction, not a proven gain.
- In the demo (`task3_match/api.py`, parameters in `data/cache/task3_stacker.json`, fitted on seed-42 validation) a CV's
  score depends on the other CVs in the folder, because of the relative features. One CV alone (`--cv`) is scored by the
  cross-encoder only, since the stacker was fitted with peers to compare against.
- The demo examples were only checked, never used for fitting: the backend CV moves to first place for job1 and the
  teacher from Good Fit to Potential Fit; job2 keeps the accountant first.
- The error analysis and error propagation sections above describe the cross-encoder alone (before stacking).

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
- Small test set (26 jobs, 23 with at least 3 CVs); the cross-encoder was run on three split seeds, the other analyses on seed 42 only. Entities from ATS-style text can contain words
  glued together (for example `analystlocation`).
- The demo CVs and jobs in `examples/` are synthetic and the "expected" fits in `scripts/make_examples.py` are the
  authors' judgement.

## Datasets

Copies of all datasets are in Google Drive:
https://drive.google.com/drive/folders/193WEVcSjKlk4OgJllRtZalp6pGBpJJ2I?usp=sharing
(BEAM design files: https://drive.google.com/drive/folders/1W985tma4c4wzifBoCa7Y40uBAGiDcEVx?usp=sharing).
Large files are split into parts; after downloading the folder, run
`python merge_parts.py <download_dir>` from the repo root to rebuild the files under `data/`, then
`python -m task2_cls.data --rebuild` once (Task 2 pins the file hash of the original export; content and splits are the same).
The notebooks can also download them from the original sources:

| Task | Dataset | Size | License |
|---|---|---|---|
| 1 | [Resume NER Training Dataset](https://www.kaggle.com/datasets/yashpwrr/resume-ner-training-dataset) | 5,960 CVs (824 kept) | MIT |
| 1 | [Green et al. 2022 skill extraction corpus](https://github.com/acp19tag/skill-extraction-dataset) | job-description sentences, 336-sentence gold test | CC BY 4.0 |
| 2 | [ResumeAtlas](https://huggingface.co/datasets/ahmedheakl/resume-atlas) | 13,389 CVs, 43 categories (11,345 after dedup) | MIT |
| 3 | [resume-ats-score-v1-en](https://huggingface.co/datasets/0xnbk/resume-ats-score-v1-en) | 6,374 CV-job pairs | Apache-2.0 |

`khaliladimassi/cleaned-job-description` (Kaggle) was used only for exploration and is not in the Drive folder.

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
python task3_match/seed_check.py    # cross-encoder variants over seeds 42/43/44 (needs the Colab predictions)
python task3_match/error_propagation.py   # corrupt Task 1/Task 2 inputs to the demo model (CPU, ~40 min)
python task3_match/stacking.py      # stacker used by the demo ranking: fit, evaluate, save data/cache/task3_stacker.json (CPU, ~3 min)
```

The cross-encoder runs need a GPU: open `notebooks/task3_crossencoder_colab.ipynb` in Colab (it clones the repo with a
`GH_TOKEN` secret, runs 8 fine-tuning runs, and saves results to Google Drive). Seed 42 everywhere; the per-job splits
are stored in `data/splits/task3.json`. `notebooks/task3_seed_check_colab.ipynb` reruns the variants on seeds 43 and 44.

## Team

| Task | Owner (NIM) |
|---|---|
| 1. NER | 13523065 |
| 2. Classification | 13523007 |
| 3. Matching | 13523117 |
