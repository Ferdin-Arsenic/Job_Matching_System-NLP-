import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from task2_cls.data import ROOT, SEED, load_prepared, split_frames
from task2_cls.evaluate import (
    OUTPUT_LABELS, confusion_tables, evaluate, predictions_from_probs, validate_probabilities,
)
from task2_cls.field_map import FIELD_MAP
from task2_cls.preprocessing import LEVEL_RULE_VERSION, PREPROCESSING_VERSION, normalize_like_resumeatlas

C_GRID = (0.1, 1.0, 10.0)
VECTORIZER_CONFIG = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100_000)
MODEL_NAMES = ("majority", "tfidf_lr", "derived_field", "hierarchical")


def aligned_probabilities(classifier, features, labels) -> np.ndarray:
    raw = classifier.predict_proba(features)
    result = np.zeros((len(raw), len(labels)))
    indices = {label: i for i, label in enumerate(labels)}
    for j, label in enumerate(classifier.classes_):
        result[:, indices[label]] = raw[:, j]
    return result


def consistent_probabilities(probs: dict, mode: str, field_map: dict) -> dict:
    result = {key: values.copy() for key, values in probs.items()}
    fields, subs = OUTPUT_LABELS["field"], OUTPUT_LABELS["sub_field"]
    membership = np.asarray([[field_map[sub] == field for field in fields] for sub in subs])
    if mode == "derived_field":
        # Derive the hard Field from the top Sub-field, not the largest marginal.
        # Field probabilities are conditional on that selected Field (one-hot).
        chosen = np.asarray([fields.index(field_map[subs[i]]) for i in result["sub_field"].argmax(axis=1)])
        result["field"] = np.eye(len(fields))[chosen]
    elif mode == "hierarchical":
        chosen = result["field"].argmax(axis=1)
        mask = membership[:, chosen].T
        masked = result["sub_field"] * mask
        total = masked.sum(axis=1, keepdims=True)
        result["sub_field"] = np.divide(masked, total, out=mask / mask.sum(axis=1, keepdims=True), where=total > 0)
    return result


@dataclass
class BaselineModel:
    mode: str
    vectorizer: object
    classifiers: dict
    field_map: dict
    metadata: dict

    def predict_proba(self, texts: list[str]) -> dict:
        if isinstance(texts, str) or not texts or any(not isinstance(t, str) for t in texts):
            raise ValueError("Pass a non-empty list of CV text strings")
        if self.metadata["preprocessing_version"] != PREPROCESSING_VERSION or self.field_map != FIELD_MAP:
            raise ValueError("Model preprocessing/taxonomy differs from the current implementation")
        normalized = [normalize_like_resumeatlas(t) for t in texts]
        features = np.zeros((len(texts), 1)) if self.vectorizer is None else self.vectorizer.transform(normalized)
        probs = {output: aligned_probabilities(self.classifiers[output], features, labels)
                 for output, labels in OUTPUT_LABELS.items()}
        probs = consistent_probabilities(probs, self.mode, self.field_map)
        validate_probabilities(probs, len(texts))
        return probs

    def predict(self, texts: list[str]) -> dict:
        return predictions_from_probs(self.predict_proba(texts))


def _macro_f1(y, probs, labels) -> float:
    pred = np.asarray(labels)[probs.argmax(axis=1)]
    return float(f1_score(y, pred, labels=list(labels), average="macro", zero_division=0))


def train_baselines(train: pd.DataFrame, val: pd.DataFrame, dataset_metadata: dict,
                    *, c_grid=C_GRID, max_features=100_000) -> tuple[dict, pd.DataFrame]:
    if set(train.text_md5) & set(val.text_md5):
        raise ValueError("Train/validation IDs overlap")
    vectorizer = TfidfVectorizer(**{**VECTORIZER_CONFIG, "max_features": max_features})
    print("Fitting TF-IDF on train only...", flush=True)
    x_train = vectorizer.fit_transform(train.text.map(normalize_like_resumeatlas))
    x_val = vectorizer.transform(val.text.map(normalize_like_resumeatlas))
    print(f"TF-IDF: {x_train.shape[0]} CVs, {x_train.shape[1]} features", flush=True)
    metadata = {
        "dataset_fingerprint": dataset_metadata["fingerprint"],
        "preprocessing_version": PREPROCESSING_VERSION, "level_rule_version": LEVEL_RULE_VERSION,
        "labels": {o: list(labels) for o, labels in OUTPUT_LABELS.items()},
        "seed": SEED, "vectorizer": {**VECTORIZER_CONFIG, "max_features": max_features},
        "c_grid": list(c_grid), "classifier": {"max_iter": 1000, "class_weight": "balanced", "solver": "lbfgs"},
        "versions": {name: importlib.metadata.version(name) for name in ("scikit-learn", "numpy", "joblib")},
        "implementation_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ("baselines.py", "evaluate.py")},
    }
    candidates, cached, majority, tuning, chosen = {}, {}, {}, [], {}
    for output, labels in OUTPUT_LABELS.items():
        train_mask = train.level.ne("Unknown").to_numpy() if output == "level" else np.ones(len(train), dtype=bool)
        val_mask = val.level.ne("Unknown").to_numpy() if output == "level" else np.ones(len(val), dtype=bool)
        y_train, y_val = train[output].to_numpy()[train_mask], val[output].to_numpy()[val_mask]
        if set(y_train) != set(labels) or not len(y_val):
            raise ValueError(f"Training must contain all {output} labels and validation must have known labels")
        majority[output] = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
        candidates[output], cached[output] = {}, {}
        best_score = -1.0
        for c in sorted(c_grid):
            started = time.monotonic()
            classifier = LogisticRegression(C=c, max_iter=1000, class_weight="balanced", random_state=SEED)
            print(f"Training {output}, C={c:g}...", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                classifier.fit(x_train[train_mask], y_train)
            probs = aligned_probabilities(classifier, x_val, labels)
            score = _macro_f1(y_val, probs[val_mask], labels)
            candidates[output][c], cached[output][c] = classifier, probs
            tuning.append(dict(model="tfidf_lr", output=output, C=c, split="val", metric="macro_f1", value=score))
            print(f"  validation Macro-F1={score:.4f}; {time.monotonic()-started:.1f}s", flush=True)
            if score > best_score:  # Ascending C gives deterministic smaller-C tie break.
                best_score, chosen[output] = score, c
    independent = {o: candidates[o][chosen[o]] for o in OUTPUT_LABELS}
    base = {o: cached[o][chosen[o]] for o in OUTPUT_LABELS}
    best_score, hierarchical_c = -1.0, None
    for c in sorted(c_grid):
        probs = consistent_probabilities({**base, "sub_field": cached["sub_field"][c]}, "hierarchical", FIELD_MAP)
        score = _macro_f1(val.sub_field.to_numpy(), probs["sub_field"], OUTPUT_LABELS["sub_field"])
        tuning.append(dict(model="hierarchical", output="sub_field", C=c, split="val", metric="macro_f1", value=score))
        if score > best_score:
            best_score, hierarchical_c = score, c
    hierarchical = {**independent, "sub_field": candidates["sub_field"][hierarchical_c]}
    metadata["selected_C"] = chosen
    metadata["hierarchical_sub_field_C"] = hierarchical_c
    metadata["selection_policy"] = "per-output validation Macro-F1; hierarchical Sub-field C retuned after masking with selected Field classifier; smaller C wins ties"
    models = {
        mode: BaselineModel(mode, None if mode == "majority" else vectorizer,
                            majority if mode == "majority" else hierarchical if mode == "hierarchical" else independent,
                            dict(FIELD_MAP), {**metadata, "mode": mode})
        for mode in MODEL_NAMES
    }
    return models, pd.DataFrame(tuning)


def save_model(model: BaselineModel, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib", compress=3)
    (directory / "metadata.json").write_text(json.dumps({**model.metadata, "field_map": model.field_map}, indent=2) + "\n", encoding="utf-8")


def load_model(directory: Path) -> BaselineModel:
    return joblib.load(Path(directory) / "model.joblib")


def run(root: Path = ROOT, *, threads: int = 4) -> pd.DataFrame:
    data, splits, dataset_metadata = load_prepared(root)
    parts = split_frames(data, splits)
    with threadpool_limits(limits=threads):
        models, tuning = train_baselines(parts["train"], parts["val"], dataset_metadata)
    results_dir, models_dir = root / "data/results", root / "models/cls/baselines"
    results_dir.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(results_dir / "task2_tuning.csv", index=False)
    # Select a consistent runtime candidate on validation before examining test.
    validation = {name: evaluate(parts["val"], model.predict_proba(parts["val"].text.tolist()))
                  for name, model in models.items()}
    scores = {name: float(validation[name].loc[validation[name].metric.eq("macro_f1"), "value"].mean())
              for name in ("derived_field", "hierarchical")}
    selected = max(scores, key=scores.get)
    print(f"Consistent candidate selected on validation: {selected} (mean output Macro-F1={scores[selected]:.4f})", flush=True)
    metrics, predictions = [], []
    for name, model in models.items():
        save_model(model, models_dir / name)
        reloaded = load_model(models_dir / name)
        for split in ("val", "test"):
            part = parts[split]
            probs = reloaded.predict_proba(part.text.tolist())
            if split == "val":
                before = model.predict_proba(part.text.iloc[:3].tolist())
                for output in OUTPUT_LABELS:
                    np.testing.assert_allclose(before[output], probs[output][:3], atol=1e-12)
            rows = evaluate(part, probs).assign(model=name, split=split)
            metrics.append(rows)
            pred = predictions_from_probs(probs)
            out = part[["text_md5", "field", "sub_field", "level"]].copy()
            out["model"], out["split"] = name, split
            for output in OUTPUT_LABELS:
                out[f"pred_{output}"] = pred[output]
                out[f"probs_{output}"] = list(probs[output])
            predictions.append(out)
            for output, table in confusion_tables(part, probs).items():
                table.to_csv(results_dir / f"task2_confusion_{name}_{split}_{output}.csv")
        pd.concat(metrics[-2:], ignore_index=True).to_csv(results_dir / f"task2_{name}.csv", index=False)
    results = pd.concat(metrics, ignore_index=True)
    results.to_csv(results_dir / "task2_baselines.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(results_dir / "task2_predictions.parquet", index=False)
    (models_dir / "selection.json").write_text(json.dumps({
        "selected_model": selected, "criterion": "mean Field/Sub-field/Level validation Macro-F1",
        "validation_scores": scores, "dataset_fingerprint": dataset_metadata["fingerprint"],
        "eligible_models": list(scores), "field_policy": "consistent Field/Sub-field only",
    }, indent=2) + "\n", encoding="utf-8")
    plot_comparison(results, root / "figures/task2_baseline_comparison.png")
    print(results.loc[results.split.eq("test") & results.metric.eq("macro_f1")].pivot(index="model", columns="output", values="value").round(4).to_string(), flush=True)
    return results


def plot_comparison(results: pd.DataFrame, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        values = results.loc[results.split.eq(split) & results.metric.eq("macro_f1")]
        values.pivot(index="model", columns="output", values="value").plot.bar(ax=ax)
        ax.set(title=f"Task 2 baseline Macro-F1 ({split})", ylim=(0, 1), xlabel="", ylabel="Macro-F1")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4, help="Maximum native training threads")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    run(threads=args.threads)


if __name__ == "__main__":
    # Keep pickle's class path importable when this module is launched with -m.
    from task2_cls.baselines import main as entrypoint
    entrypoint()
