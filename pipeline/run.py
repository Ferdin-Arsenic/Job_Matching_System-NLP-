"""End-to-end pipeline: CV text + job text -> Task 1 NER -> Task 2 profile (CV only) -> Task 3 match."""
from pathlib import Path

from pipeline.io import read_cv
from task1_ner.api import extract_many
from task2_cls.api import get_classifier
from task3_match import api as match_api

# Task 2 model chosen on validation mean Macro-F1: JobBERT 0.929 vs TF-IDF+LR derived_field 0.813.
# Falls back to the baseline when the checkpoint (not in git) is absent.
JOBBERT_DIR = Path(__file__).resolve().parents[1] / "models" / "cls" / "jobbert"


def seniority_note(profile: dict, job_entities: dict) -> str | None:
    """Flag the known false-positive pattern: skills overlap but the job asks for more experience."""
    required = job_entities.get("experience_years")
    if required is None:
        return None
    if (profile["level"] == "Entry" and required >= 2) or (profile["level"] == "Mid" and required > 5):
        return f"job asks ~{required:g} years; CV level is {profile['level']}"
    return None


def analyze(cv_paths: list, job_text: str) -> dict:
    cvs = [{"name": str(p), "text": read_cv(p)} for p in cv_paths]
    entities = extract_many([job_text] + [cv["text"] for cv in cvs])  # NER on original text (cased)
    job = {"text": job_text, "entities": entities[0]}
    # Level from Task 1 experience_years: returns Unknown when the CV states no experience, whereas the
    # ML Level heads predict mostly Senior there (TF-IDF ~75%, JobBERT ~82% of such validation CVs).
    model_dir = JOBBERT_DIR if (JOBBERT_DIR / "metadata.json").exists() else None
    classifier = get_classifier(model_dir, level_source="ner")
    profiles = classifier.classify_many([cv["text"] for cv in cvs], entities[1:])
    for cv, ent, profile in zip(cvs, entities[1:], profiles):
        cv["entities"], cv["profile"] = ent, profile
    ranked = match_api.rank(cvs, job)
    for cv in ranked:
        cv["note"] = seniority_note(cv["profile"], job["entities"])
    return {"job": job, "ranked": ranked,
            "models": {"task1": "JobBERT-CRF (WwzFwz/jobbert-crf-cv-jd-ner)",
                       "task2": f"{classifier.metadata['model']} ({classifier.metadata['status']}), "
                                f"Level from Task 1 years",
                       "task3": match_api.MODEL_NAME}}
