import re
from collections.abc import Mapping
from math import isnan
from numbers import Real

from common.schemas import ENTITY_LABELS, adapt_task1_entities

PREPROCESSING_VERSION = "resumeatlas-v1"
LEVEL_RULE_VERSION = "context-window-5-v1"
_CONTEXT = {"experience", "exp", "working", "professional", "career"}
_WORDS = re.compile(r"\b\w+\b")
_DURATION = re.compile(
    r"(?<![\w.])(?P<low>\d{1,2}(?:\.\d+)?)"
    r"(?:\s*[-–]\s*(?P<high>\d{1,2}(?:\.\d+)?))?"
    r"\s*(?:\+|plus)?\s*(?P<unit>years?|yrs?|months?)\b", re.I,
)


def normalize_like_resumeatlas(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def entities_to_text(entities: Mapping) -> str:
    """Concatenate entities in fixed order (SKILL, JOB_TITLE, EDUCATION, EXPERIENCE), normalized once; no numeric metadata."""
    grouped = adapt_task1_entities(entities)
    return normalize_like_resumeatlas(" ".join(v for label in ENTITY_LABELS for v in grouped[label]))


def extract_training_experience_years(text: str) -> float | None:
    words = list(_WORDS.finditer(text.lower()))
    values = []
    for match in _DURATION.finditer(text):
        before = [w.group() for w in words if w.end() <= match.start()][-5:]
        after = [w.group() for w in words if w.start() >= match.end()][:5]
        if not _CONTEXT.intersection(before + after):
            continue
        low = float(match["low"])
        years = float(match["high"]) if match["high"] else low
        if match["high"] and years < low:
            continue
        if match["unit"].lower().startswith("month"):
            years /= 12
        if 0 <= years <= 40:
            values.append(years)
    return max(values) if values else None


def level_from_years(years: float | None) -> str:
    if years is None or (isinstance(years, Real) and isnan(years)):
        return "Unknown"
    if isinstance(years, bool) or not isinstance(years, Real) or not 0 <= years <= 40:
        raise ValueError("Experience must be between 0 and 40 years, or unknown")
    return "Entry" if years < 2 else "Mid" if years <= 5 else "Senior"


def training_level_from_text(text: str) -> str:
    return level_from_years(extract_training_experience_years(text))


def ner_level_from_entities(entities: Mapping) -> str:
    """Level from Task 1's experience estimate; more than 40 years becomes Unknown (as in the export cleanup)."""
    years = adapt_task1_entities(entities)["experience_years"]
    return "Unknown" if years is not None and years > 40 else level_from_years(years)
