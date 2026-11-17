# Task 2 foundations (Slice 1)

No dataset split, classifier training, or inference API is implemented yet.

## Contracts

`common.schemas.Entities` represents the actual grouped Task 1 output: four
lists (`SKILL`, `JOB_TITLE`, `EDUCATION`, `EXPERIENCE`) and nullable
`experience_years`. `adapt_task1_entities()` accepts inference dictionaries or
export rows (including Parquet array cells), copies the lists, converts NaN years
to `None`, and excludes unrelated metadata. It does not fabricate character
offsets or convert the old 13-label span contract. Missing groups are errors.

`Profile` uses `field`, `sub_field`, `level`, `field_probs`, and `sub_probs`.
Runtime validation requires Field/Sub-field consistency by default; independent
classifier experiments can explicitly disable that check. Probability dictionaries
may be empty; otherwise they must cover all output labels and sum to one.
`Match` uses a finite score in [0, 100], a fit tier, and tier probabilities.

`field_map.py` is the module source for the existing 43-to-7 mapping, copied
unchanged from the notebook. Task 3's profile stub imports it. Historical notebook
cells remain unchanged. **Mapping status: pending team confirmation**, not frozen.
Label tuples define deterministic order; later checkpoints must persist them.

## Preprocessing

- Keep original text for NER and MD5 joins. Normalize only a separate classifier
  input with `normalize_like_resumeatlas()` for both training and inference.
- Normalization matches the existing stub: lowercase, replace non-ASCII
  alphanumerics with spaces, collapse whitespace. This loses punctuation in skill
  names such as C++ and decimal numbers; it cannot restore punctuation already
  removed from ResumeAtlas.
- `entities_to_text()` concatenates values in fixed order: SKILL, JOB_TITLE,
  EDUCATION, EXPERIENCE, then normalizes. It excludes numeric metadata and does not
  reorder or deduplicate source values.

## Separate sources for seniority

`training_level_from_text()` forms weak training labels from durations with an
`experience`, `exp`, `working`, `professional`, or `career` cue within five words
before/after the duration. It uses the maximum valid duration. Parse before
normalization where original punctuation is available: decimals are retained,
months converted to years, and ranges use their upper endpoint. Reversed ranges
are rejected. Durations above 40 years are rejected, following Task 1 export
cleanup; this assumption needs the planned manual validation.

`ner_level_from_entities()` is a separate comparison rule using the existing
Task 1 `experience_years`, which may include merged calendar intervals. It does
not rerun the training-label regex. Values above 40 become Unknown to match
export cleanup. The adapter itself preserves finite non-negative years so source
data is not silently rewritten by a schema operation.

Both sources use `<2 = Entry`, `2–5 = Mid`, `>5 = Senior`, unknown = `Unknown`.
Known zero years means Entry. These are heuristic experience bands, not human
seniority annotations. Context matching can still produce false positives;
manual auditing remains part of the later evaluation slice.

## Checks

From the repository root:

```text
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover export adaptation, title preservation, normalization, duration/context
edge cases, level boundaries, independent NER estimates, and invalid contracts.
