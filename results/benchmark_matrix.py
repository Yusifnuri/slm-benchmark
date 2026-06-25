"""
Benchmark matrix compiler.
Expose: "Full benchmark matrix: 6 models x 5 tasks x 5 metrics" (Week 11-12)

Pulls all logged results from MLflow and compiles into:
- benchmark_matrix.csv (for thesis tables)
- summary statistics per model and task
"""

import mlflow
import pandas as pd
import os


MODELS = [
    "gpt-4o",
    "claude-3-5-haiku",
    "gemini-1-5-flash",
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


def compile_benchmark_matrix(
    tracking_uri: str = "sqlite:///mlflow.db",
    output_path: str = "results/benchmark_matrix.csv",
) -> pd.DataFrame:
    """
    Pull all benchmark results from MLflow and compile into 6x5x5 matrix.
    Returns DataFrame and saves to CSV.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    rows = []

    # Fetch all benchmark runs (tagged phase = phase2_benchmark or phase1_baseline)
    experiments = client.search_experiments()
    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.phase IN ('phase2_benchmark', 'phase1_baseline')",
        )
        for run in runs:
            tags = run.data.tags
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