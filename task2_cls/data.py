# Hash-based data splits
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common.schemas import ENTITY_LABELS, adapt_task1_entities
from task2_cls.field_map import FIELD_MAP, FIELD_LABELS, SUB_FIELD_LABELS, LEVEL_LABELS
from task2_cls.preprocessing import (
    LEVEL_RULE_VERSION, PREPROCESSING_VERSION, entities_to_text,
    extract_training_experience_years, level_from_years,
    ner_level_from_entities, normalize_like_resumeatlas,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
DATA_VERSION = "task2-data-v1"
SPLIT_NAMES = ("train", "val", "test")


def text_md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _counts(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def prepare_dataset(atlas: pd.DataFrame, ner: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = {"Text", "Category"}
    if not required <= set(atlas):
        raise ValueError("ResumeAtlas must contain Text and Category")
    if atlas.empty or not atlas.Text.map(lambda t: isinstance(t, str) and bool(t.strip())).all():
        raise ValueError("ResumeAtlas must contain non-empty text strings")
    if atlas.Category.isna().any() or not atlas.Category.isin(FIELD_MAP).all():
        raise ValueError("ResumeAtlas contains missing or unmapped categories")

    data = atlas[["Text", "Category"]].copy().reset_index(drop=True)
    data.insert(0, "source_row", range(len(data)))
    data["text_md5"] = data.Text.map(text_md5)
    report = {"raw_rows": len(data), "exact_category_conflicts": int(
        (data.groupby("Text").Category.nunique() > 1).sum())}
    data = data.drop_duplicates("Text", keep="first").copy()
    report["after_exact_dedup"] = len(data)
    prefix = data.Text.str[:200]
    report["prefix_category_conflicts"] = int(
        (data.groupby(prefix).Category.nunique() > 1).sum())
    data = data.loc[~prefix.duplicated(keep="first")].copy()
    report["after_prefix_dedup"] = len(data)
    report["dedup_policy"] = "exact Text, then first 200 characters; keep first source row"
    if not data.text_md5.is_unique:
        raise ValueError("Text hash collision detected")
    if data.Category.value_counts().min() < 3:
        raise ValueError("Each Sub-field requires at least 3 samples after deduplication")

    required_ner = {"text_md5", *ENTITY_LABELS, "experience_years"}
    if not required_ner <= set(ner) or ner.text_md5.isna().any():
        raise ValueError("NER export must contain text_md5 and all grouped entity columns")
    # Canonicalize before comparing duplicate keys: metadata/category differences
    # do not change entities, but conflicting predictions must not be hidden.
    rows, signatures = [], {}
    for row in ner.to_dict("records"):
        grouped = adapt_task1_entities(row)
        signature = _digest(grouped)
        key = row["text_md5"]
        if key in signatures:
            if signatures[key] != signature:
                raise ValueError(f"Conflicting NER predictions for text_md5={key}")
            continue
        signatures[key] = signature
        rows.append({"text_md5": key, **grouped})
    report["ner_export_rows"] = len(ner)
    report["ner_unique_hashes"] = len(rows)
    report["ner_missing_raw_rows"] = int((~atlas.Text.map(text_md5).isin(signatures)).sum())
    missing = data.loc[~data.text_md5.isin(signatures), "text_md5"]
    if len(missing):
        raise ValueError(f"Missing NER predictions for {len(missing)} retained CVs; first: {missing.iloc[0]}")
    data = data.merge(pd.DataFrame(rows), on="text_md5", how="left", validate="one_to_one")
    data = data.rename(columns={"Text": "text", "Category": "sub_field"})
    data["field"] = data.sub_field.map(FIELD_MAP)
    data["normalized_text"] = data.text.map(normalize_like_resumeatlas)
    data["training_experience_years"] = data.text.map(extract_training_experience_years)
    data["level"] = data.training_experience_years.map(level_from_years)
    grouped_rows = data[[*ENTITY_LABELS, "experience_years"]].to_dict("records")
    data["entity_text"] = [entities_to_text(row) for row in grouped_rows]
    data["ner_level"] = [ner_level_from_entities(row) for row in grouped_rows]
    report.update({"ner_matched_rows": len(data), "field_counts": _counts(data.field),
                   "sub_field_counts": _counts(data.sub_field), "level_counts": _counts(data.level),
                   "ner_level_counts": _counts(data.ner_level),
                   "empty_entity_text_rows": int(data.entity_text.eq("").sum())})
    return data.reset_index(drop=True), report


def make_splits(data: pd.DataFrame, seed: int = SEED) -> dict[str, list[str]]:
    try:
        train, held_out = train_test_split(data, test_size=0.2, random_state=seed,
                                           stratify=data.sub_field)
        val, test = train_test_split(held_out, test_size=0.5, random_state=seed,
                                     stratify=held_out.sub_field)
    except ValueError as exc:
        raise ValueError(f"Cannot create stratified 80/10/10 splits: {exc}") from exc
    splits = {name: part.text_md5.tolist() for name, part in
              (("train", train), ("val", val), ("test", test))}
    validate_splits(data, splits)
    return splits


def validate_splits(data: pd.DataFrame, splits: dict) -> None:
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError("Split file must contain train, val, and test")
    all_ids = []
    indexed = data.set_index("text_md5")
    if not indexed.index.is_unique:
        raise ValueError("Dataset IDs must be unique")
    classes = set(data.sub_field)
    for name in SPLIT_NAMES:
        ids = splits[name]
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids):
            raise ValueError(f"Invalid IDs in {name} split")
        if not set(ids) <= set(indexed.index):
            raise ValueError(f"Unknown dataset IDs in {name} split")
        if set(indexed.loc[ids].sub_field) != classes:
            raise ValueError(f"{name} split does not cover all Sub-fields")
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(indexed.index):
        raise ValueError("Split IDs overlap, repeat, or do not cover the dataset")
    prefixes = data.text.str[:200]
    if not prefixes.is_unique:
        raise ValueError("Dataset still contains duplicate 200-character prefixes")


def split_frames(data: pd.DataFrame, splits: dict) -> dict[str, pd.DataFrame]:
    validate_splits(data, splits)
    indexed = data.set_index("text_md5", drop=False)
    return {name: indexed.loc[splits[name]].reset_index(drop=True) for name in SPLIT_NAMES}


def _paths(root: Path) -> dict[str, Path]:
    return {
        "atlas": root / "data/resume_atlas.parquet",
        "ner": root / "outputs/ner/ner_atlas.parquet",
        "prepared": root / "data/cache/task2.parquet",
        "metadata": root / "data/cache/task2_metadata.json",
        "splits": root / "data/splits/task2.json",
    }


def _input_metadata(paths: dict, seed: int) -> dict:
    for name in ("atlas", "ner"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"Required local input missing: {paths[name]}")
    metadata = {
        "dataset": "ahmedheakl/resume-atlas", "data_version": DATA_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION, "level_rule_version": LEVEL_RULE_VERSION,
        "seed": seed, "split_strategy": "stratified Sub-field 80/10/10, two train_test_split calls",
        "dedup_prefix_chars": 200, "dedup_keep": "first",
        "field_map": FIELD_MAP, "field_labels": list(FIELD_LABELS),
        "sub_field_labels": list(SUB_FIELD_LABELS), "level_labels": list(LEVEL_LABELS),
        "source_sha256": {name: _file_digest(paths[name]) for name in ("atlas", "ner")},
        "implementation_sha256": {name: _file_digest(Path(__file__).with_name(name))
                                  for name in ("data.py", "preprocessing.py", "field_map.py")},
        "schemas_sha256": _file_digest(Path(__file__).resolve().parents[1] / "common/schemas.py"),
    }
    metadata["fingerprint"] = _digest(metadata)
    return metadata


def load_prepared(root: Path = ROOT, seed: int = SEED) -> tuple[pd.DataFrame, dict, dict]:
    """Load artifacts only if inputs, configuration, and saved content still match."""
    paths = _paths(Path(root))
    inputs = _input_metadata(paths, seed)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["splits"].read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != inputs["fingerprint"] or manifest.get("fingerprint") != inputs["fingerprint"]:
        raise ValueError("Task 2 inputs/configuration changed; explicitly rebuild the prepared artifacts")
    if metadata.get("prepared_sha256") != _file_digest(paths["prepared"]):
        raise ValueError("Prepared Task 2 data changed; rebuild the prepared artifacts")
    splits = manifest["splits"]
    if metadata.get("splits_sha256") != _digest(splits):
        raise ValueError("Saved split IDs changed; rebuild the prepared artifacts")
    data = pd.read_parquet(paths["prepared"])
    validate_splits(data, splits)
    return data, splits, metadata


def prepare_assets(root: Path = ROOT, *, rebuild: bool = False, seed: int = SEED) -> tuple[pd.DataFrame, dict, dict]:
    paths = _paths(Path(root))
    outputs = [paths[k] for k in ("prepared", "metadata", "splits")]
    if any(p.exists() for p in outputs) and not rebuild:
        if not all(p.exists() for p in outputs):
            raise ValueError("Incomplete Task 2 artifacts; use --rebuild to recreate them")
        return load_prepared(root, seed)
    metadata = _input_metadata(paths, seed)
    data, report = prepare_dataset(pd.read_parquet(paths["atlas"]), pd.read_parquet(paths["ner"]))
    if set(data.sub_field) != set(SUB_FIELD_LABELS):
        raise ValueError("Prepared ResumeAtlas must cover all 43 Sub-fields")
    splits = make_splits(data, seed)
    report["splits"] = {
        name: {"rows": len(part), "field_counts": _counts(part.field),
               "sub_field_counts": _counts(part.sub_field), "level_counts": _counts(part.level)}
        for name, part in split_frames(data, splits).items()
    }
    metadata["report"] = report
    metadata["splits_sha256"] = _digest(splits)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(paths["prepared"], index=False)
    metadata["prepared_sha256"] = _file_digest(paths["prepared"])
    paths["splits"].write_text(json.dumps({"fingerprint": metadata["fingerprint"], "seed": seed,
                                          "splits": splits}, indent=2) + "\n", encoding="utf-8")
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return data, splits, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Explicitly replace existing prepared data and splits")
    args = parser.parse_args()
    _, _, metadata = prepare_assets(rebuild=args.rebuild)
    report = metadata["report"]
    print(f"Rows: {report['raw_rows']} -> {report['after_exact_dedup']} exact -> {report['after_prefix_dedup']} prefix")
    print(f"NER joined: {report['ner_matched_rows']} | Label conflicts: exact={report['exact_category_conflicts']}, prefix={report['prefix_category_conflicts']}")
    for name, values in report["splits"].items():
        print(f"{name}: {values['rows']} CVs, {len(values['sub_field_counts'])} Sub-fields, Levels={values['level_counts']}")
    print(f"Fingerprint: {metadata['fingerprint']}")


if __name__ == "__main__":
    main()
