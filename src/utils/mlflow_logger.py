"""
Centralized MLflow logger.

Logs all 5 metrics required by the expose (Section 2):
- accuracy
- latency (ms/request)
- cost per 1M tokens
- privacy risk
- ROI breakeven volume
"""

import mlflow
from typing import Dict, Any, Optional
from datetime import datetime


class BenchmarkLogger:
    """Single logger used across Phase 1 (baselines) and Phase 2 (SLMs)."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "sqlite:///mlflow.db",
    ):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def log_training_run(
        self,
        model_name: str,
        task: str,
        method: str,               # "LoRA" or "QLoRA"
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts_path: Optional[str] = None,
    ) -> str:
        """Log a fine-tuning training run."""
        run_name = (
            f"{model_name.split('/')[-1]}_{task}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "model": model_name,
                "task": task,
                "method": method,
                "phase": "phase2_fine_tuning",
            })
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if artifacts_path:
                mlflow.log_artifacts(artifacts_path)
            run_id = mlflow.active_run().info.run_id
        print(f"✅ MLflow logged: {run_name} | run_id: {run_id}")
        return run_id

    def log_benchmark_result(
        self,
        model_name: str,
        task: str,
        accuracy: float,
        latency_ms: float,
        cost_per_1m_tokens: float,
        privacy_risk: str,           # "low" | "medium" | "high"
        roi_breakeven_tokens: float,
    ):
        """
        Log final benchmark entry for the 6x5x5 matrix.
        Called after evaluation — one entry per model-task combination.
        Expose Section 2: '6 models x 5 tasks x 5 metrics'
        """
        run_name = f"benchmark_{model_name.split('/')[-1]}_{task}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "model": model_name,
                "task": task,
                "phase": "phase2_benchmark",
                "privacy_risk": privacy_risk,
            })
            mlflow.log_metrics({
                "accuracy": accuracy,
                "latency_ms": latency_ms,
                "cost_per_1m_tokens": cost_per_1m_tokens,
                "roi_breakeven_tokens": roi_breakeven_tokens,
            })
        print(f"Benchmark result logged: {model_name} | {task}")