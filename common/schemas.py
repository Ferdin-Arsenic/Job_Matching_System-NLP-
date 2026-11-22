"""Runtime contracts for the grouped Task 1 output (strings, no character spans) and downstream tasks."""

from collections.abc import Mapping
from math import isfinite
from numbers import Real
from typing import Literal, TypedDict

ENTITY_LABELS = ("SKILL", "JOB_TITLE", "EDUCATION", "EXPERIENCE")
FIT_TIERS = ("No Fit", "Potential Fit", "Good Fit")


class Entities(TypedDict):
    SKILL: list[str]
    JOB_TITLE: list[str]
    EDUCATION: list[str]
    EXPERIENCE: list[str]
    experience_years: float | None


class Profile(TypedDict):
    field: str
    sub_field: str
    level: Literal["Entry", "Mid", "Senior", "Unknown"]
    field_probs: dict[str, float]
    sub_probs: dict[str, float]


class Match(TypedDict):
    score: float
    tier: Literal["No Fit", "Potential Fit", "Good Fit"]
    probs: dict[str, float]


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def validate_entities(entities: Mapping) -> None:
    for label in ENTITY_LABELS:
        values = entities.get(label)
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f"{label} must be a list of non-empty strings")
    if "experience_years" not in entities:
        raise ValueError("experience_years is required (use None when unknown)")
    years = entities["experience_years"]
    if years is not None and _number(years, "experience_years") < 0:
        raise ValueError("experience_years must be non-negative")


def adapt_task1_entities(row: Mapping) -> Entities:
    # Convert a Task 1 result/export row to canonical Python lists and nulls.
    out = {}
    for label in ENTITY_LABELS:
        if label not in row or isinstance(row[label], (str, bytes, Mapping)):
            raise ValueError(f"Missing or invalid Task 1 group: {label}")
        try:
            out[label] = list(row[label])
        except TypeError as exc:
            raise ValueError(f"{label} must contain an iterable of strings") from exc
    years = row.get("experience_years")
    if isinstance(years, Real) and years != years:
        years = None
    out["experience_years"] = None if years is None else _number(years, "experience_years")
    validate_entities(out)
    return out


def _validate_probs(probs: object, labels: tuple[str, ...], name: str) -> None:
    if not isinstance(probs, dict):
        raise ValueError(f"{name} must be a dictionary")
    if not probs:  # Contract permits empty probabilities for rule/stub outputs.
        return
    if set(probs) != set(labels):
        raise ValueError(f"{name} must contain every expected label")
    values = [_number(v, name) for v in probs.values()]
    if any(v < 0 or v > 1 for v in values) or abs(sum(values) - 1) > 1e-5:
        raise ValueError(f"{name} must be probabilities summing to one")


def validate_profile(profile: Mapping, *, require_consistency: bool = True) -> None:
    from task2_cls.field_map import FIELD_LABELS, FIELD_MAP, PROFILE_LEVELS, SUB_FIELD_LABELS

    if profile.get("field") not in FIELD_LABELS or profile.get("sub_field") not in SUB_FIELD_LABELS:
        raise ValueError("Unknown Field or Sub-field")
    if require_consistency and FIELD_MAP[profile["sub_field"]] != profile["field"]:
        raise ValueError("Field must agree with the Sub-field mapping")
    if profile.get("level") not in PROFILE_LEVELS:
        raise ValueError("Unknown Level")
    _validate_probs(profile.get("field_probs"), FIELD_LABELS, "field_probs")
    _validate_probs(profile.get("sub_probs"), SUB_FIELD_LABELS, "sub_probs")


def validate_match(match: Mapping) -> None:
    score = _number(match.get("score"), "score")
    if not 0 <= score <= 100 or match.get("tier") not in FIT_TIERS:
        raise ValueError("Match needs a score in [0, 100] and a valid fit tier")
    _validate_probs(match.get("probs"), FIT_TIERS, "probs")
