"""Shared Task 2 evaluation with fixed label order and masked Unknown levels."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from task2_cls.field_map import FIELD_LABELS, FIELD_MAP, LEVEL_LABELS, SUB_FIELD_LABELS

OUTPUT_LABELS = {"field": FIELD_LABELS, "sub_field": SUB_FIELD_LABELS, "level": LEVEL_LABELS}


def validate_probabilities(probs: dict, n: int) -> None:
    if set(probs) != set(OUTPUT_LABELS):
        raise ValueError("Probabilities must include field, sub_field, and level")
    for output, labels in OUTPUT_LABELS.items():
        values = np.asarray(probs[output])
        if values.shape != (n, len(labels)):
            raise ValueError(f"Invalid probability shape for {output}")
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(f"Invalid probabilities for {output}")
        if not np.allclose(values.sum(axis=1), 1, atol=1e-6):
            raise ValueError(f"Probabilities for {output} must sum to one")


def predictions_from_probs(probs: dict) -> dict[str, np.ndarray]:
    return {output: np.asarray(labels)[np.asarray(probs[output]).argmax(axis=1)]
            for output, labels in OUTPUT_LABELS.items()}


def evaluate(data: pd.DataFrame, probs: dict) -> pd.DataFrame:
    """Long-format metrics. Unknown Level rows count for Field/Sub-field only; Level and exact match use known-Level rows."""
    validate_probabilities(probs, len(data))
    pred = predictions_from_probs(probs)
    known = data.level.ne("Unknown").to_numpy()
    rows = []

    def record(output, metric, value, n):
        rows.append(dict(output=output, metric=metric, value=value, n=int(n)))

    for output, labels in OUTPUT_LABELS.items():
        mask = known if output == "level" else np.ones(len(data), dtype=bool)
        y, p = data[output].to_numpy()[mask], pred[output][mask]
        if not set(y) <= set(labels):
            raise ValueError(f"Unknown true labels for {output}")
        if len(y):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, p, labels=list(labels), average="macro", zero_division=0)
            metrics = {"accuracy_top1": accuracy_score(y, p), "precision_macro": precision,
                       "recall_macro": recall, "macro_f1": f1}
        else:
            metrics = dict.fromkeys(("accuracy_top1", "precision_macro", "recall_macro", "macro_f1"), np.nan)
        for metric, value in metrics.items():
            record(output, metric, value, len(y))
        if output == "sub_field":
            top = np.argsort(-np.asarray(probs[output]), axis=1, kind="stable")[:, :5]
            top_labels = np.asarray(labels)[top]
            value = (top_labels == data.sub_field.to_numpy()[:, None]).any(axis=1).mean() if len(data) else np.nan
            record(output, "accuracy_top5", value, len(data))
    exact = np.logical_and.reduce([pred[o] == data[o].to_numpy() for o in OUTPUT_LABELS])
    record("all", "exact_match", exact[known].mean() if known.any() else np.nan, known.sum())
    derived = np.asarray([FIELD_MAP[s] for s in pred["sub_field"]])
    record("all", "field_sub_inconsistency", (derived != pred["field"]).mean() if len(data) else np.nan, len(data))
    record("level", "known_fraction", known.mean() if len(data) else np.nan, len(data))
    return pd.DataFrame(rows)


def confusion_tables(data: pd.DataFrame, probs: dict) -> dict[str, pd.DataFrame]:
    validate_probabilities(probs, len(data))
    pred = predictions_from_probs(probs)
    tables = {}
    for output, labels in OUTPUT_LABELS.items():
        mask = data.level.ne("Unknown").to_numpy() if output == "level" else np.ones(len(data), dtype=bool)
        counts = confusion_matrix(data[output].to_numpy()[mask], pred[output][mask], labels=list(labels))
        tables[output] = pd.DataFrame(counts, index=pd.Index(labels, name="true_label"), columns=labels)
    return tables
