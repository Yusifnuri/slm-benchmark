"""
Orchestrator: sequentially fine-tunes and evaluates all 3 SLMs across all 5
benchmark tasks (15 runs total), matching the Expose's Phase 2 plan
("Sequential to manage GPU queue", Weeks 5-10).

Each model's training logic stays in its own file
(src/training/train_lora.py, src/training/train_qlora.py) — this script only
sequences calls into them, one run at a time, so a single GPU is used safely
and every run is still logged to MLflow individually and reproducibly from
its own script + config.

Why one script instead of running 15 notebook cells by hand:
- Resumable: a run is skipped if its adapter directory already exists, so a
  Jupyter kernel restart / disconnect does not require starting over.
- Fault-isolated: one model/task combination failing (OOM, bad data, etc.)
  is caught and logged; the sweep continues with the next combination
  instead of stopping the whole overnight job.

Usage (from the GPU Jupyter environment):
    # in a notebook cell
    !python scripts/run_all_experiments.py

    # or from a terminal (recommended for multi-hour runs, survives
    # a closed browser tab if launched under nohup/tmux/screen)
    python scripts/run_all_experiments.py 2>&1 | tee logs/run_all_$(date +%Y%m%d_%H%M).log

Run `python scripts/prepare_financial_data.py` once beforehand — the
financial_sentiment task needs data/financial_phrasebank.csv to exist.
"""

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train_lora import train as train_lora
from src.training.train_qlora import train as train_qlora
from src.evaluation.evaluate import run_full_evaluation
from src.evaluation.evaluate_code import evaluate_humaneval

FINANCIAL_PATH = "data/financial_phrasebank.csv"
TASKS = ["classification", "ner", "summarization", "financial_sentiment", "code_generation"]

# (display name, config path, trainer function, HuggingFace model id)
MODELS = [
    ("phi4-lora", "configs/phi4_lora.yaml", train_lora, "microsoft/phi-4-mini-instruct"),
    ("mistral-qlora", "configs/mistral_qlora.yaml", train_qlora, "mistralai/Mistral-7B-v0.3"),
    ("llama-lora", "configs/llama_lora.yaml", train_lora, "meta-llama/Llama-3.2-3B-Instruct"),
]


def adapter_path_for(model_name: str, task: str) -> str:
    return f"./adapters/{model_name.split('/')[-1]}_{task}"


def run_all_training() -> list:
    """Fine-tune every (model, task) combination once, skipping ones already done."""
    failures = []
    total = len(MODELS) * len(TASKS)
    done = 0

    for label, config_path, trainer_fn, model_name in MODELS:
        for task in TASKS:
            done += 1
            progress = f"[{done}/{total}]"

            if os.path.isdir(adapter_path_for(model_name, task)):
                print(f"{progress} Skipping {label}/{task} — adapter already exists")
                continue

            print(f"\n{'=' * 60}\n{progress} Training {label} | {task}\n{'=' * 60}")
            kwargs = {"financial_path": FINANCIAL_PATH} if task == "financial_sentiment" else {}

            start = time.time()
            try:
                trainer_fn(config_path, task, **kwargs)
                print(f"Done: {label}/{task} in {(time.time() - start) / 60:.1f} min")
            except Exception:
                print(f"FAILED: {label}/{task} — continuing with next run")
                traceback.print_exc()
                failures.append((label, task))

    print(f"\nTraining sweep complete. {len(failures)} failure(s): {failures}")
    return failures


def run_all_evaluation() -> list:
    """Evaluate every fine-tuned (model, task) combination and log to MLflow."""
    failures = []

    for label, _config_path, _trainer_fn, model_name in MODELS:
        for task in TASKS:
            adapter_path = adapter_path_for(model_name, task)
            if not os.path.isdir(adapter_path):
                print(f"Skipping evaluation for {label}/{task} — no adapter found, train it first")
                continue

            print(f"\nEvaluating {label} | {task}")
            try:
                if task == "code_generation":
                    evaluate_humaneval(base_model_name=model_name, adapter_path=adapter_path)
                else:
                    run_full_evaluation(
                        base_model_name=model_name,
                        task=task,
                        adapter_path=adapter_path,
                        financial_phrasebank_path=(
                            FINANCIAL_PATH if task == "financial_sentiment" else None
                        ),
                    )
            except Exception:
                print(f"FAILED evaluation: {label}/{task}")
                traceback.print_exc()
                failures.append((label, task))

    print(f"\nEvaluation sweep complete. {len(failures)} failure(s): {failures}")
    return failures


if __name__ == "__main__":
    if not os.path.exists(FINANCIAL_PATH):
        print(
            f"⚠ {FINANCIAL_PATH} not found. Run "
            "`python scripts/prepare_financial_data.py` first — the "
            "financial_sentiment task will fail without it."
        )

    print("PHASE 2 — Fine-tuning all SLMs across all tasks (sequential GPU sweep)")
    train_failures = run_all_training()

    print("\nPHASE 2 — Evaluating all fine-tuned SLMs")
    eval_failures = run_all_evaluation()

    if train_failures or eval_failures:
        print(
            "\nSome runs failed — see tracebacks above. "
            "Re-running this script will retry only the missing (model, task) combinations."
        )
    else:
        print(f"\nAll {len(MODELS) * len(TASKS)} model x task combinations trained and evaluated.")
