import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.schemas import validate_profile
from task2_cls.api import Task2Classifier
from task3_match.common import ROOT, load, md5


def validate_profiles(profiles, pairs):
    required = {"resume_id", "field", "sub_field", "level"}
    if not required <= set(profiles) or profiles.empty:
        raise ValueError("Profile cache must contain IDs, Field, Sub-field and Level")
    if profiles.resume_id.isna().any() or profiles.resume_id.duplicated().any():
        raise ValueError("Profile cache must have one row per resume")
    if set(profiles.resume_id) != set(pairs.resume_id):
        raise ValueError("Profile IDs must exactly cover matching resumes")
    for row in profiles.to_dict("records"):
        validate_profile({**row, "field_probs": row.get("field_probs", {}), "sub_probs": row.get("sub_probs", {})})


def export_profiles(pairs, classifier, output, *, ner=None, batch_size=64, force=False):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if classifier.metadata["status"] != "trained":
        raise ValueError("Production profiles require a trained Task 2 artifact")
    if pairs.empty or pairs.resume.isna().any() or not pairs.resume.map(lambda t: isinstance(t, str) and bool(t.strip())).all():
        raise ValueError("Matching data must contain non-empty resume texts")
    if not pairs.resume.map(md5).eq(pairs.resume_id).all():
        raise ValueError("resume_id must be MD5 of original resume text")
    unique = pairs[["resume_id", "resume"]].drop_duplicates().sort_values("resume_id")
    if unique.resume_id.duplicated().any():
        raise ValueError("Conflicting resume texts for the same ID")
    entities = None
    ner_hash = None
    if classifier.level_source == "ner":
        if ner is None or ner.text_md5.duplicated().any():
            raise ValueError("NER Level requires unique Task 1 records")
        indexed = ner.set_index("text_md5")
        if not set(unique.resume_id) <= set(indexed.index):
            raise ValueError("Missing NER records for matching resumes")
        entities = indexed.loc[unique.resume_id].to_dict("records")
        # Hash canonical input including entity lists/numeric evidence.
        ner_hash = hashlib.sha256(indexed.loc[unique.resume_id].to_json().encode()).hexdigest()
    output = Path(output)
    metadata_path = output.with_suffix(".metadata.json")
    metadata = {"schema_version": 1, "classifier": classifier.metadata, "resume_id": "MD5 original UTF-8 text",
                "resume_ids_sha256": hashlib.sha256("\n".join(unique.resume_id).encode()).hexdigest(),
                "rows": len(unique), "pairs": len(pairs), "ner_sha256": ner_hash,
                "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if not force and output.exists() and metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous == metadata:
            cached = pd.read_parquet(output)
            validate_profiles(cached, pairs)
            return cached, False
    rows = []
    texts = unique.resume.tolist()
    for start in range(0, len(unique), batch_size):
        rows.extend(classifier.classify_many(texts[start:start + batch_size],
                    entities[start:start + batch_size] if entities is not None else None))
    result = pd.DataFrame(rows)
    result.insert(0, "resume_id", unique.resume_id.tolist())
    validate_profiles(result, pairs)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return result, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/cache/profiles.parquet")
    parser.add_argument("--level-source", choices=("model", "ner"), default="model")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    classifier = Task2Classifier(args.model_dir, level_source=args.level_source)
    ner = pd.read_parquet(ROOT / "outputs/ner/ner_ats_resume.parquet") if args.level_source == "ner" else None
    result, updated = export_profiles(load(), classifier, args.output, ner=ner, batch_size=args.batch_size, force=args.force)
    print(f"{'Exported' if updated else 'Reused valid cache'}: {len(result)} unique resumes -> {args.output}")
    print("Level source:", classifier.level_source, "|", result.level.value_counts().to_dict())


if __name__ == "__main__":
    main()
