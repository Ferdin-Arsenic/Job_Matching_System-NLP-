"""Upload the Task 3 cross-encoder (models/match/V3_cls_lr2e-05) to Hugging Face as a PUBLIC model.

Run once after logging in (python -c "from huggingface_hub import login; login()"):
    python scripts/upload_task3_model.py [repo_name]      # default repo_name: jobbert-task3-match
Then set T3_MATCH_REPO=<user>/<repo_name> (or make it the default in task3_match/api.py).
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

folder = Path(__file__).resolve().parents[1] / "models" / "match" / "V3_cls_lr2e-05"
if not (folder / "model.safetensors").exists():
    sys.exit(f"Weights not found: {folder}")
api = HfApi()
repo = f"{api.whoami()['name']}/{sys.argv[1] if len(sys.argv) > 1 else 'jobbert-task3-match'}"
if not api.repo_exists(repo):    # a token without "create repo" permission can still upload to a repo made on the website
    try:
        api.create_repo(repo, repo_type="model", private=False)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Cannot create {repo}: {exc}\nUse a token with WRITE permission, or create the empty model "
                 f"repo '{repo}' on huggingface.co/new and run this script again.")
api.upload_folder(folder_path=str(folder), repo_id=repo, allow_patterns=["*.json", "*.safetensors", "README.md"],
                  commit_message="Task 3 cross-encoder V3")
print(f"OK: https://huggingface.co/{repo}\nT3_MATCH_REPO = {repo}")
