import argparse
import os
import sys
import time
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore")


def show_entities(title, entities):
    print(f"  {title}")
    for label in ("SKILL", "JOB_TITLE", "EDUCATION", "EXPERIENCE"):
        values = entities[label]
        shown = ", ".join(values[:12]) + (f" (+{len(values) - 12} more)" if len(values) > 12 else "")
        print(f"    {label:<11}: {shown or '-'}")
    years = entities["experience_years"]
    print(f"    {'years':<11}: {'-' if years is None else f'{years:g}'}")


def main():
    parser = argparse.ArgumentParser(description="CV-Job matching: NER -> classification -> matching")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--cv", help="CV file (.pdf or .txt)")
    src.add_argument("--cv-dir", help="folder of CVs (.pdf/.txt) to rank for one job")
    parser.add_argument("--job", required=True, help="job description: .txt path or literal text")
    parser.add_argument("--verbose", action="store_true", help="print entities and profile probabilities")
    args = parser.parse_args()

    from pipeline.io import InputError, list_cvs, read_job
    try:
        job_text = read_job(args.job)
        paths = [args.cv] if args.cv else list_cvs(args.cv_dir)
        print("Loading models and running pipeline (first run downloads models)...", flush=True)
        started = time.time()
        from pipeline.run import analyze
        result = analyze(paths, job_text)
    except InputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    models = result["models"]
    print(f"\nModels: Task1={models['task1']} | Task2={models['task2']} | Task3={models['task3']}")
    if args.verbose:
        print("\n[Task 1] Job description entities")
        show_entities("JOB", result["job"]["entities"])
    ranked = result["ranked"]
    for i, cv in enumerate(ranked, 1):
        p, m = cv["profile"], cv["match"]
        if args.verbose or len(ranked) == 1:
            print(f"\n=== CV {i}: {cv['name']}")
            if args.verbose:
                print("[Task 1] CV entities")
                show_entities("CV", cv["entities"])
            print(f"[Task 2] Field={p['field']} | Sub-field={p['sub_field']} | Level={p['level']}")
            if args.verbose and p["sub_probs"]:
                top = sorted(p["sub_probs"].items(), key=lambda kv: -kv[1])[:3]
                print("         top sub-fields: " + ", ".join(f"{k} {v:.2f}" for k, v in top))
            probs = ", ".join(f"{k} {v:.2f}" for k, v in m["probs"].items())
            print(f"[Task 3] Score={m['score']:.1f}/100 | Tier={m['tier']} | probs: {probs}")
            if cv["note"]:
                print(f"         note: {cv['note']}")
    if len(ranked) > 1:
        print(f"\nRanking for job ({len(ranked)} CVs):")
        print(f"{'#':<3}{'CV':<34}{'Score':>6}  {'Tier':<14}{'Field / Sub-field / Level':<55}Note")
        for i, cv in enumerate(ranked, 1):
            p, m = cv["profile"], cv["match"]
            name = os.path.basename(cv["name"])[:32]
            prof = f"{p['field']} / {p['sub_field']} / {p['level']}"[:53]
            print(f"{i:<3}{name:<34}{m['score']:>6.1f}  {m['tier']:<14}{prof:<55}{cv['note'] or ''}")
    print(f"\nDone in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
