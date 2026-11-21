from functools import lru_cache
from pathlib import Path
import re
import sys
from scripts.t1_ner_predict import ENTITIES, extract, load
from common.schemas import adapt_task1_entities  # noqa: E402

FUNC = {"to", "of", "in", "and", "or", "the", "a", "an", "for", "with", "at", "from", "since", "till", "until",
        "minimum", "min", "at least"}
NUMW = r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\b"
MAX_YEARS = 40


def _keep(label, value):
    if all(x in FUNC for x in value.split()):
        return False
    if label == "EXPERIENCE":
        return bool(re.search(r"\d", value) or re.search(NUMW, value))
    if label == "EDUCATION":
        return value not in {"gpa", "of science", "of arts", "certified", "certifications"}
    return len(value) > 1 or value in {"c", "r"}


@lru_cache(maxsize=1)
def _model():
    return load()


def extract_many(texts: list[str]) -> list[dict]:
    model, tokenizer = _model()
    out = []
    for row in extract(texts, model, tokenizer):
        # Export computes experience_years before cleaning; keep that order.
        years = row["experience_years"]
        cleaned = {label: [v for v in row[label] if _keep(label, v)] for label in ENTITIES}
        cleaned["experience_years"] = None if years is not None and years > MAX_YEARS else years
        out.append(adapt_task1_entities(cleaned))
    return out


def extract_entities(text: str) -> dict:
    return extract_many([text])[0]
