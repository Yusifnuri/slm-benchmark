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
from evaluation.metrics import LLM_API_COSTS, get_privacy_risk, calculate_roi_breakeven

# Throughput was measured on the rented H200 instance, so the imputed rate
# must be an H200-class commercial rate — pricing the measured hardware at a
# cheaper card's rate would systematically understate c_slm (fast GPU + cheap
# price). $3.99/hr is a single-H200 on-demand rate (JarvisLabs, verified
# Aug 2026); the specialised-cloud band is roughly $3.68 (Vast.ai) to $4.50
# (Nebius), with hyperscalers (AWS/Azure/OCI) near $10/GPU-hr. All SLM cost
# and ROI numbers scale linearly in this constant, so re-pricing at any
# other rate is a one-line substitution.
GPU_COST_PER_HOUR = 3.99
# The evaluation runs stored in mlflow logged cost_per_1m_tokens at the rate
# in force at eval time ($2.50/hr). c_slm is linear in the rate, so stored
# costs are rescaled to GPU_COST_PER_HOUR here — otherwise the matrix would
# mix a $3.99-based training cost with a $2.50-based inference cost and the
# breakeven division would be meaningless.
LOGGED_GPU_RATE = 2.50
ROI_REFERENCE_API = "gpt-4o"


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


# Phase 1's real source of truth is the per-instance CSV each notebook writes
# to logs/ — mlflow only ever held a derived copy of the same aggregates. The
# CSVs are preferred here because they are per-instance (so confidence
# intervals and error analysis remain computable), they are human-auditable,
# and they do not depend on a sqlite file that a stale kernel can lock. If a
# notebook's mlflow logging fails, the benchmark is unaffected.
PHASE1_CSV_SPECS = {
    # task: (candidate filenames in priority order, per-model column suffix)
    "classification": (["ag_news_baseline.csv"], "correct"),
    "ner": (["conll_baseline.csv"], "f1"),
    "summarization": (["cnn_dailymail_baseline.csv"], "rougeL"),
    # the v3 file is the corrected seeded-split sample; the bare name is the
    # superseded label-clustered one kept only as a fallback (see §4.2.4)
    "financial_sentiment": (["financial_baseline_v3_seededsplit.csv",
                             "financial_baseline.csv"], "correct"),
    "code_generation": (["humaneval_baseline.csv"], "pass"),
}


# Latency samples above this are treated as measurement artefacts, not
# provider behaviour. Five samples across four tasks exceed it: four sit at
# 1,652-1,656 s and one at 727 s, against medians of 0.7-2.0 s and
# non-artefact maxima of ~12 s. Four of the five cluster within seconds of
# each other in wall-clock terms, which is the signature of the client host
# suspending mid-request rather than of five independent slow responses. A
# completion capped at 512 tokens cannot legitimately take 27 minutes. They
# are excluded from the reported latency and counted in
# latency_artefacts_excluded so the exclusion is visible rather than silent.
LATENCY_ARTEFACT_THRESHOLD_S = 60.0


def _parse_phase1_csvs(logs_dir: str = "logs") -> List[dict]:
    """One row per (API model, task), computed from the per-instance CSVs."""
    rows = []
    for task, (candidates, suffix) in PHASE1_CSV_SPECS.items():
        path = next((os.path.join(logs_dir, c) for c in candidates
                     if os.path.exists(os.path.join(logs_dir, c))), None)
        if path is None:
            continue
        df = pd.read_csv(path)
        for prefix, model_name in PHASE1_MODEL_PREFIXES.items():
            score_col, lat_col = f"{prefix}_{suffix}", f"{prefix}_latency_s"
            if score_col not in df.columns:
                continue
            score = df[score_col].dropna()
            if score.empty:
                continue
            raw_latency = df[lat_col].dropna() if lat_col in df.columns else None
            if raw_latency is not None and not raw_latency.empty:
                clean = raw_latency[raw_latency <= LATENCY_ARTEFACT_THRESHOLD_S]
                excluded = int(len(raw_latency) - len(clean))
                lat_mean = round(float(clean.mean()) * 1000, 2) if not clean.empty else None
                lat_median = round(float(clean.median()) * 1000, 2) if not clean.empty else None
                lat_p95 = round(float(clean.quantile(0.95)) * 1000, 2) if not clean.empty else None
            else:
                lat_mean = lat_median = lat_p95 = None
                excluded = 0
            rows.append({
                "model": model_name,
                "task": task,
                "method": "API",
                "accuracy": round(float(score.mean()), 6),
                "latency_ms": lat_mean,
                "latency_median_ms": lat_median,
                "latency_p95_ms": lat_p95,
                "latency_artefacts_excluded": excluded,
                "cost_per_1m_tokens": LLM_API_COSTS[model_name]["blended"],
                "privacy_risk": get_privacy_risk("api"),
                "roi_breakeven_tokens": None,
                "n_instances": int(len(score)),
                "source": os.path.basename(path),
            })
    return rows


def _parse_phase1_notebook_runs(client: "mlflow.tracking.MlflowClient") -> List[dict]:
    """
    Reconstruct one row per (model, task) from Phase 1's bundled-metrics runs.

    Only the latest run per experiment is used — these notebooks were
    re-executed many times over the course of debugging (retry logic added,
    temperature fixed, checkpoints resumed, etc.), and mlflow.start_run()
    creates a new run each time rather than overwriting, so an experiment
    can hold several stale pre-fix runs alongside the final one.
    """
    rows = []
    for exp in client.search_experiments():
        if exp.name not in PHASE1_EXPERIMENT_TASKS:
            continue
        task, accuracy_key = PHASE1_EXPERIMENT_TASKS[exp.name]
        latest_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        for run in latest_runs:
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


def _lookup_training_info(client: "mlflow.tracking.MlflowClient") -> dict:
    """
    Maps (model, task) -> {"method": "LoRA"/"QLoRA", "training_hours": float}
    from the phase2_fine_tuning-tagged runs train_lora.py/train_qlora.py log.

    benchmark_matrix.py uses this rather than trusting whatever "method" and
    "roi_breakeven_tokens" a benchmark run already logged, because:
    - log_benchmark_result never set a "method" tag, so every benchmark row
      (SLM included) fell back to the "API" default.
    - Evaluation runs (run_full_evaluation / evaluate_humaneval) always
      called with fine_tuning_cost_usd=0.0, so every previously-logged
      roi_breakeven_tokens assumed free training — meaningless for the
      thesis's actual "when does fine-tuning pay off" question.
    Deduped to the latest run per (model, task), since retries after bugfixes
    (e.g. the learning_rate/dataset-id fixes mid-sweep) left stale runs
    alongside the final successful ones.
    """
    info = {}
    for exp in client.search_experiments():
        for run in client.search_runs(experiment_ids=[exp.experiment_id]):
            tags = run.data.tags
            if tags.get("phase") != "phase2_fine_tuning":
                continue
            key = (tags.get("model", "unknown").split("/")[-1], tags.get("task", "unknown"))
            if key in info and info[key]["start_time"] >= run.info.start_time:
                continue
            info[key] = {
                "method": tags.get("method", "unknown"),
                "training_hours": run.data.metrics.get("training_hours", 0.0),
                "start_time": run.info.start_time,
            }
    return info


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

    # Phase 1: read the per-instance CSVs first. This is the authoritative
    # source and needs no mlflow connection at all.
    csv_rows = _parse_phase1_csvs()
    if csv_rows:
        covered = {(r["model"], r["task"]) for r in csv_rows}
        print(f"Phase 1 from CSVs: {len(csv_rows)} rows "
              f"({len({t for _, t in covered})} tasks)")
        rows.extend(csv_rows)
    else:
        print("Phase 1 CSVs not found — falling back to mlflow")

    for tracking_uri in tracking_uris:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()

        # Phase 1 rows come from the CSVs (see PHASE1_CSV_SPECS); the mlflow
        # parser below is kept only as a fallback for tasks whose CSV is
        # missing, since the two sources hold the same aggregates.
        if not csv_rows:
            rows.extend(_parse_phase1_notebook_runs(client))

        # (model, task) -> real training method/hours, for the "method" and
        # "roi_breakeven_tokens" fixups below. A no-op on notebooks/mlflow.db
        # (no phase2_fine_tuning runs there).
        training_info = _lookup_training_info(client)

        # Fetch all runs and keep only benchmark ones (phase2_benchmark or
        # phase1_baseline). Filtered in Python, not via filter_string's
        # "IN (...)" syntax — this mlflow version's search DSL only accepts
        # a single quoted value per tag comparison and raises on a tuple.
        #
        # Deduped to the latest run per (model, task): re-running evaluation
        # for the same combo (e.g. after a bugfix) creates a new mlflow run
        # rather than overwriting, so without this a stale pre-fix result
        # could sit alongside the real one and get averaged together.
        latest_by_key = {}
        experiments = client.search_experiments()
        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            for run in runs:
                tags = run.data.tags
                if tags.get("phase") not in ("phase2_benchmark", "phase1_baseline"):
                    continue
                key = (tags.get("model", "unknown"), tags.get("task", "unknown"))
                if key in latest_by_key and latest_by_key[key].info.start_time >= run.info.start_time:
                    continue
                latest_by_key[key] = run

        for (model_name, task), run in latest_by_key.items():
            tags = run.data.tags
            metrics = run.data.metrics
            model_short = model_name.split("/")[-1]
            cost_per_1m = metrics.get("cost_per_1m_tokens", None)
            train_info = training_info.get((model_short, task))

            if train_info is not None:
                # SLM row: log_benchmark_result never tagged "method", and
                # every previously-logged roi_breakeven_tokens assumed
                # $0 fine-tuning cost. Use the real training method/hours
                # instead of trusting either.
                method = train_info["method"]
                # Rescale the eval-time cost to the current imputed rate
                # (see LOGGED_GPU_RATE comment above).
                if cost_per_1m is not None:
                    cost_per_1m = round(cost_per_1m * GPU_COST_PER_HOUR / LOGGED_GPU_RATE, 4)
                fine_tuning_cost_usd = train_info["training_hours"] * GPU_COST_PER_HOUR
                api_cost = LLM_API_COSTS[ROI_REFERENCE_API]["blended"]
                roi_breakeven_tokens = (
                    calculate_roi_breakeven(fine_tuning_cost_usd, api_cost, cost_per_1m)
                    if cost_per_1m is not None else None
                )
            else:
                # No matching fine-tuning run — either a genuine API row or
                # an SLM row whose training run predates this script's fix.
                method = tags.get("method", "API")
                roi_breakeven_tokens = metrics.get("roi_breakeven_tokens", None)

            rows.append({
                "model": model_short,
                "task": task,
                "method": method,
                "accuracy": metrics.get("accuracy", None),
                "latency_ms": metrics.get("latency_ms", None),
                "cost_per_1m_tokens": cost_per_1m,
                "privacy_risk": tags.get("privacy_risk", "unknown"),
                "roi_breakeven_tokens": roi_breakeven_tokens,
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