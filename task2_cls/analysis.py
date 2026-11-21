"""Task 2 error analysis on the saved baseline split (no test-set tuning)."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from task2_cls.baselines import load_model, train_baselines
from task2_cls.data import ROOT, SEED, load_prepared, split_frames
from task2_cls.evaluate import OUTPUT_LABELS, evaluate, predictions_from_probs
from task2_cls.field_map import LEVEL_LABELS, SUB_FIELD_LABELS
from task2_cls.preprocessing import _DURATION

RESULTS = ROOT / "data/results"
MODELS = ROOT / "models/cls/baselines"


def input_comparison(parts, model, dataset_metadata, threads):
    rows = []
    for split in ("val", "test"):
        part = parts[split]
        for name, column in (("full_text", "text"), ("entity_only (inference substitution)", "entity_text")):
            rows.append(evaluate(part, model.predict_proba(part[column].tolist())).assign(split=split, input=name))
    # Retrained ablation: same TF-IDF+LR protocol, trained and tuned on entity strings.
    as_entities = {k: v.assign(text=v.entity_text) for k, v in parts.items()}
    with threadpool_limits(limits=threads):
        models, tuning = train_baselines(as_entities["train"], as_entities["val"], dataset_metadata)
    retrained = models[model.mode]
    for split in ("val", "test"):
        part = as_entities[split]
        rows.append(evaluate(part, retrained.predict_proba(part.text.tolist()))
                    .assign(split=split, input="entity_only (retrained)"))
    return pd.concat(rows, ignore_index=True), tuning.assign(input="entity_only")


def level_comparison(test, model_pred):
    rows = []
    known = test.level.ne("Unknown")
    sources = {"ml_model (full text)": model_pred, "rule from Task 1 experience_years": test.ner_level.to_numpy()}
    for name, pred in sources.items():
        pred = np.asarray(pred)
        mask = known.to_numpy() & (pred != "Unknown")
        y, p = test.level.to_numpy()[mask], pred[mask]
        rows.append(dict(source=name, coverage_of_known=mask.sum() / known.sum(), n=int(mask.sum()),
                         accuracy=accuracy_score(y, p),
                         macro_f1=f1_score(y, p, labels=list(LEVEL_LABELS), average="macro", zero_division=0)))
    rows.append(dict(source="Task 1 rule coverage on all CVs of this split", coverage_of_known=np.nan,
                     n=int(test.ner_level.ne("Unknown").sum()),
                     accuracy=test.ner_level.ne("Unknown").mean(), macro_f1=np.nan))
    return pd.DataFrame(rows)


def per_class(test, pred_sub):
    p, r, f, s = precision_recall_fscore_support(test.sub_field, pred_sub, labels=list(SUB_FIELD_LABELS), zero_division=0)
    table = pd.DataFrame({"sub_field": SUB_FIELD_LABELS, "precision": p, "recall": r, "f1": f, "support": s})
    confusions = (pd.DataFrame({"true": test.sub_field, "pred": pred_sub}).query("true != pred")
                  .value_counts().rename("count").reset_index().head(20))
    return table.sort_values("f1"), confusions


def threshold_sample(test, n=20):
    years = test.training_experience_years
    near = test[(years.between(1, 3)) | (years.between(4, 6))]
    sample = near.sample(n=min(n, len(near)), random_state=SEED)

    def context(text):
        snippets = []
        for m in _DURATION.finditer(text):
            snippets.append(text[max(0, m.start() - 60): m.end() + 60])
        return " ... ".join(snippets[:4])

    return pd.DataFrame({"text_md5": sample.text_md5, "sub_field": sample.sub_field,
                         "regex_years": sample.training_experience_years, "regex_level": sample.level,
                         "ner_years": sample.experience_years, "ner_level": sample.ner_level,
                         "context": sample.text.map(context), "manual_level": ""})


def main(threads=4):
    data, splits, dataset_metadata = load_prepared(ROOT)
    parts = split_frames(data, splits)
    selected = json.loads((MODELS / "selection.json").read_text(encoding="utf-8"))["selected_model"]
    model = load_model(MODELS / selected)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Level source is chosen on val (pipeline uses the Task 1 rule); test is reported only.
    levels = []
    for split in ("val", "test"):
        part = parts[split]
        pred = predictions_from_probs(model.predict_proba(part.text.tolist()))
        levels.append(level_comparison(part, pred["level"]).assign(split=split))
        unknown = part.level.eq("Unknown").to_numpy()
        levels.append(pd.DataFrame([dict(source="ml_model prediction on regex-Unknown rows", split=split,
                                         n=int(unknown.sum()), **{f"pred_{k}": float(np.mean(pred["level"][unknown] == k))
                                                                  for k in LEVEL_LABELS})]))
    levels = pd.concat(levels, ignore_index=True)
    levels.to_csv(RESULTS / "task2_analysis_level_sources.csv", index=False)
    test = parts["test"]
    pred = predictions_from_probs(model.predict_proba(test.text.tolist()))
    classes, confusions = per_class(test, pred["sub_field"])
    classes.to_csv(RESULTS / "task2_analysis_per_class_test.csv", index=False)
    confusions.to_csv(RESULTS / "task2_analysis_top_confusions_test.csv", index=False)
    sample = threshold_sample(test)
    sample.to_csv(RESULTS / "task2_analysis_threshold_sample.csv", index=False)
    print(f"Selected model: {selected}\n\nLevel sources:\n{levels.round(4).to_string(index=False)}")
    print(f"\nLowest Sub-field F1 (test):\n{classes.head(10).round(3).to_string(index=False)}")
    print(f"\nTop confusions (test):\n{confusions.head(10).to_string(index=False)}")

    comparison, tuning = input_comparison(parts, model, dataset_metadata, threads)
    comparison.to_csv(RESULTS / "task2_analysis_input_comparison.csv", index=False)
    tuning.to_csv(RESULTS / "task2_analysis_entity_tuning.csv", index=False)
    view = comparison[comparison.metric.isin(["macro_f1", "accuracy_top5"]) & comparison.split.eq("test")]
    print("\nInput comparison (test):")
    print(view.pivot_table(index="input", columns=["output", "metric"], values="value").round(4).to_string())


if __name__ == "__main__":
    main()
