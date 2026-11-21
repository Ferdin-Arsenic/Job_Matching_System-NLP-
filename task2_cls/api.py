from functools import lru_cache
import hashlib
import json
from pathlib import Path
import warnings

from common.schemas import validate_profile
from task2_cls.baselines import load_model
from task2_cls.evaluate import OUTPUT_LABELS, predictions_from_probs
from task2_cls.field_map import FIELD_MAP
from task2_cls.preprocessing import (
    PREPROCESSING_VERSION, entities_to_text, ner_level_from_entities, normalize_like_resumeatlas,
)

DEFAULT_MODELS = Path(__file__).resolve().parents[1] / "models/cls/baselines"


class Task2Classifier:
    def __init__(self, model_dir=None, *, level_source="model", allow_fallback=False):
        if level_source not in ("model", "ner"):
            raise ValueError("level_source must be model or ner")
        self.level_source = level_source
        self.model = None
        try:
            if model_dir is None:
                selection = json.loads((DEFAULT_MODELS / "selection.json").read_text(encoding="utf-8"))
                selected = selection["selected_model"]
                if selected not in ("derived_field", "hierarchical"):
                    raise ValueError("Selection must use a consistent trained baseline")
                model_dir = DEFAULT_MODELS / selected
            self.model_dir = Path(model_dir).resolve()
            model_metadata = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8")) if (self.model_dir / "metadata.json").exists() else {}
            is_jobbert = model_metadata.get("architecture") == "jobbert_multi_output"
            if is_jobbert:
                if model_metadata.get("status") != "trained":
                    raise ValueError("API requires a full trained checkpoint; smoke/sample experiments are excluded")
                from task2_cls.model import JobBertPredictor
                self.model = JobBertPredictor(self.model_dir)
            else:
                self.model = load_model(self.model_dir)
        except FileNotFoundError:
            if not allow_fallback:
                raise FileNotFoundError("Task 2 artifact missing; train baselines first or explicitly set allow_fallback=True") from None
            warnings.warn("Task 2 FALLBACK (untrained): fixed Information Technology profile", RuntimeWarning, stacklevel=2)
        if self.model is not None:
            if self.model.mode not in ("derived_field", "hierarchical"):
                raise ValueError("API requires a consistent trained baseline")
            if self.model.metadata["labels"] != {k: list(v) for k, v in OUTPUT_LABELS.items()}:
                raise ValueError("Model labels differ from runtime labels")
        self.metadata = {
            "status": "trained" if self.model else "fallback (untrained)",
            "model": ("jobbert_multi_output" if is_jobbert else self.model.mode) if self.model else "fixed-information-technology",
            "model_sha256": self._artifact_digest() if self.model else None,
            "preprocessing_version": PREPROCESSING_VERSION,
            "level_source": level_source,
            "missing_experience": "model predicts without NER evidence" if level_source == "model" else "Unknown",
            "input": self.model.metadata.get("input_mode", "full_text") if self.model else "full_text",
            "dataset_fingerprint": self.model.metadata.get("dataset_fingerprint") if self.model else None,
            "field_probability_policy": "one-hot derived from top Sub-field" if self.model and self.model.mode == "derived_field" else "model",
            "implementation_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                      for name in ("api.py", "baselines.py", "model.py", "preprocessing.py", "field_map.py", "evaluate.py")},
        }

    def _artifact_digest(self):
        files = [self.model_dir / "model.joblib"] if (self.model_dir / "model.joblib").exists() else sorted(
            p for p in self.model_dir.rglob("*") if p.is_file() and (p.suffix == ".json" or p.name == "weights.pt"))
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.relative_to(self.model_dir).as_posix().encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()

    def classify_many(self, texts, entities=None):
        if isinstance(texts, str) or not texts or any(not isinstance(t, str) or not normalize_like_resumeatlas(t) for t in texts):
            raise ValueError("Pass non-empty CV text strings with classifier tokens")
        if entities is None:
            entities = [None] * len(texts)
        if len(entities) != len(texts):
            raise ValueError("One entity record per CV is required")
        inputs = texts
        if self.metadata["input"] == "entity_only":
            if any(entity is None for entity in entities):
                raise ValueError("Entity-only checkpoint requires Task 1 entities for every CV")
            inputs = [entities_to_text(entity) for entity in entities]
        probs = self.model.predict_proba(inputs) if self.model else None
        predictions = predictions_from_probs(probs) if probs else None
        profiles = []
        for i, entity in enumerate(entities):
            sub = str(predictions["sub_field"][i]) if predictions else "Information Technology"
            level = str(predictions["level"][i]) if predictions else "Unknown"
            if self.level_source == "ner":
                level = ner_level_from_entities(entity) if entity is not None else "Unknown"
            profile = dict(field=FIELD_MAP[sub], sub_field=sub, level=level,
                           field_probs={k: float(v) for k, v in zip(OUTPUT_LABELS["field"], probs["field"][i])} if probs else {},
                           sub_probs={k: float(v) for k, v in zip(OUTPUT_LABELS["sub_field"], probs["sub_field"][i])} if probs else {})
            validate_profile(profile)
            profiles.append(profile)
        return profiles

    def classify_cv(self, text, entities=None):
        return self.classify_many([text], [entities])[0]


@lru_cache(maxsize=8)
def get_classifier(model_dir=None, *, level_source="model", allow_fallback=False):
    """Load once per configuration; clear with get_classifier.cache_clear()."""
    return Task2Classifier(model_dir, level_source=level_source, allow_fallback=allow_fallback)


def classify_cv(text, entities=None):
    return get_classifier().classify_cv(text, entities)
