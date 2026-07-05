"""
Benchmark matrix compiler.
Expose: "Full benchmark matrix: 6 models x 5 tasks x 5 metrics" (Week 11-12)

Pulls all logged results from MLflow and compiles into:
- benchmark_matrix.csv (for thesis tables)
- summary statistics per model and task
"""

import sys
import mlflow
import pandas as pd
import os
from typing import List

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from evaluation.metrics import LLM_API_COSTS, get_privacy_risk


MODELS = [
    "gpt-4o",
    "claude-haiku-4-5",   # expose specifies Claude-3.5-Haiku; see README.md "Methodology notes"
    "gemini-2.5-flash",   # expose specifies Gemini-1.5-Flash; see README.md "Methodology notes"
    "phi-4-mini-instruct",
    "Mistral-7B-v0.3",
    "Llama-3.2-3B-Instruct",
]

TASKS = [
    "classification",
    "ner",
    "summarization",
    "financial_sentiment",
    "code_generation",
]

METRICS = [
    "accuracy",
    "latency_ms",
    "cost_per_1m_tokens",
    "privacy_risk",
    "roi_breakeven_tokens",
]

# Phase 1 notebooks (02-06) predate BenchmarkLogger — each logs all 3 LLM
# APIs' metrics into ONE mlflow run per task, with metric keys prefixed per
# model (e.g. "openai_accuracy") and no model/task/phase tags at all. That's
# a fundamentally different shape than Phase 2's one-row-per-(model,task)
# BenchmarkLogger runs, so the generic tags.phase-based reader below always
# finds zero rows here — this table lets a separate parser reconstruct the
# same row format from it.
PHASE1_EXPERIMENT_TASKS = {
    "ag_news_baseline": ("classification", "accuracy"),
    "conll2003_baseline": ("ner", "f1"),
    "cnn_dailymail_baseline": ("summarization", "avg_rougeL"),
    "financial_sentiment_baseline": ("financial_sentiment", "accuracy"),
    "humaneval_baseline": ("code_generation", "pass_at_1"),
}

PHASE1_MODEL_PREFIXES = {
    "openai": "gpt-4o",
    "anthropic": "claude-haiku-4-5",
    "gemini": "gemini-2.5-flash",
}


def _parse_phase1_notebook_runs(client: "mlflow.tracking.MlflowClient") -> List[dict]:
    """Reconstruct one row per (model, task) from Phase 1's bundled-metrics runs."""
    rows = []
    for exp in client.search_experiments():
        if exp.name not in PHASE1_EXPERIMENT_TASKS:
            continue
        task, accuracy_key = PHASE1_EXPERIMENT_TASKS[exp.name]
        for run in client.search_runs(experiment_ids=[exp.experiment_id]):
            metrics = run.data.metrics
            for prefix, model_name in PHASE1_MODEL_PREFIXES.items():
                accuracy = metrics.get(f"{prefix}_{accuracy_key}")
                if accuracy is None:
                    continue  # this run didn't log this model (e.g. interrupted mid-notebook)
                latency_s = metrics.get(f"{prefix}_avg_latency_s")
                rows.append({
                    "model": model_name,
                    "task": task,
                    "method": "API",
                    "accuracy": accuracy,
                    "latency_ms": round(latency_s * 1000, 2) if latency_s is not None else None,
                    # API cost is the provider's published per-token price, not
                    # derived from throughput like on-premise SLM cost is.
                    "cost_per_1m_tokens": LLM_API_COSTS[model_name]["blended"],
                    "privacy_risk": get_privacy_risk("api"),
                    # ROI breakeven ("token volume where fine-tuning pays off
                    # vs. this API") is an SLM-side metric computed against a
                    # reference API — not meaningful for the API row itself.
                    "roi_breakeven_tokens": None,
                })
    return rows


def compile_benchmark_matrix(
    # Phase 1 (LLM API baselines) and Phase 2 (fine-tuned SLMs) log to two
    # separate mlflow.db files, because each phase's scripts/notebooks run
    # with a different working directory (notebooks/ vs. repo root) and
    # BenchmarkLogger's tracking_uri default ("sqlite:///mlflow.db") is
    # resolved relative to whichever process set it. Reading only one would
    # silently drop 3 of the 6 models from the matrix.
    tracking_uris: List[str] = ["sqlite:///notebooks/mlflow.db", "sqlite:///mlflow.db"],
    output_path: str = "results/benchmark_matrix.csv",
) -> pd.DataFrame:
    """
    Pull all benchmark results from MLflow (both phases' dbs) and compile
    into the 6x5x5 matrix. Returns DataFrame and saves to CSV.
    """
    rows = []

    for tracking_uri in tracking_uris:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        # Phase 1's notebooks use a different run shape (see
        # _parse_phase1_notebook_runs docstring) than Phase 2's BenchmarkLogger
        # runs — try both parsers against both dbs. Each is naturally a no-op
        # on the "wrong" db (no matching experiment names / no phase tags), so
        # this doesn't double-count anything regardless of which db is which.
        rows.extend(_parse_phase1_notebook_runs(client))

        # Fetch all runs and keep only benchmark ones (phase2_benchmark or
        # phase1_baseline). Filtered in Python, not via filter_string's
        # "IN (...)" syntax — this mlflow version's search DSL only accepts
        # a single quoted value per tag comparison and raises on a tuple.
        experiments = client.search_experiments()
        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            for run in runs:
                tags = run.data.tags
                if tags.get("phase") not in ("phase2_benchmark", "phase1_baseline"):
                    continue
                metrics = run.data.metrics
                rows.append({
                    "model": tags.get("model", "unknown").split("/")[-1],
                    "task": tags.get("task", "unknown"),
                    "method": tags.get("method", "API"),
                    "accuracy": metrics.get("accuracy", None),
                    "latency_ms": metrics.get("latency_ms", None),
                    "cost_per_1m_tokens": metrics.get("cost_per_1m_tokens", None),
                    "privacy_risk": tags.get("privacy_risk", "unknown"),
                    "roi_breakeven_tokens": metrics.get("roi_breakeven_tokens", None),
                })

    df = pd.DataFrame(rows)

    # Pivot to matrix format: models as rows, tasks as columns
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Benchmark matrix saved → {output_path}")
    print(f"   Shape: {df.shape}")
    print(df.groupby(["model", "task"])["accuracy"].mean().unstack())

    return df


if __name__ == "__main__":
    compile_benchmark_matrix()