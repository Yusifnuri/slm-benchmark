"""Publish the fifteen fine-tuned adapters to the Hugging Face Hub.

Exposé deliverables 2 and 3. Each adapter directory already carries a model card
(README.md) generated from the released benchmark matrix, so this script only
uploads what is on disk — it does not write metadata of its own.

Authentication: run `huggingface-cli login` first, or export HF_TOKEN. Never
pass a token on the command line; it lands in your shell history.

    python scripts/push_to_hub.py --owner <your-hf-username> --dry-run
    python scripts/push_to_hub.py --owner <your-hf-username>

Add --private to stage the repos privately and flip them public later from the
Hub UI.
"""

import argparse
import os
import sys
from pathlib import Path

ADAPTERS = Path(__file__).resolve().parent.parent / "adapters"

# Financial PhraseBank is CC BY-NC-SA 3.0, so these adapters are research
# artefacts rather than deployable assets. Their model cards say so; this list
# is here to make the upload print the reminder too.
NON_COMMERCIAL = "financial_sentiment"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True, help="Hugging Face user or organisation")
    ap.add_argument("--prefix", default="", help="Optional prefix for every repo id")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dirs = sorted(d for d in ADAPTERS.iterdir() if d.is_dir())
    if not dirs:
        print(f"no adapter directories under {ADAPTERS}", file=sys.stderr)
        return 1

    missing = [d.name for d in dirs if not (d / "README.md").exists()]
    if missing:
        print("refusing to upload — these adapters have no model card:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"would upload {len(dirs)} adapters to {args.owner}/:\n")
        for d in dirs:
            flag = "  (research only — non-commercial corpus)" if NON_COMMERCIAL in d.name else ""
            print(f"  {args.owner}/{args.prefix}{d.name}{flag}")
        return 0

    if not (os.environ.get("HF_TOKEN") or (Path.home() / ".cache/huggingface/token").exists()):
        print("not authenticated — run `huggingface-cli login` or export HF_TOKEN", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi  # imported late so --dry-run needs no install

    api = HfApi()
    for d in dirs:
        repo_id = f"{args.owner}/{args.prefix}{d.name}"
        print(f"-> {repo_id}")
        api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(d), repo_id=repo_id, repo_type="model")
        if NON_COMMERCIAL in d.name:
            print("   note: trained on a CC BY-NC-SA corpus — research artefact, not deployable")

    print(f"\n{len(dirs)} adapters published. Put the URLs in Appendix F of the thesis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
