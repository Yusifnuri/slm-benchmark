"""
Internal validation of the SLM Decision Framework on 10 hypothetical
enterprise scenarios.
Expose: "Internal testing on 10 hypothetical enterprise scenarios"
(Weeks 15-16), the step before the 5 Hamburg Hub company validation.

Runs the exact same decision logic the Streamlit app uses (shared via
src/decision_framework.py — not a reimplementation) against 10 made-up
but realistic enterprise use cases spanning all 5 tasks, both privacy
modes, volumes from 100k to 100M tokens/month, and one deliberately
unsatisfiable case, then writes the recommendations to
results/internal_validation_per_request.csv.

The scenarios keep the thesis's monthly token volumes; each is converted to
requests with the task's mean API tokens per request, because the framework
now ranks on per-request cost. results/internal_validation.csv is the thesis
appendix's output (token-based ranking) and is not overwritten.

Usage:
    python scripts/internal_validation.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.decision_framework import load_matrix, recommend, tokens_per_request

# (name, task, monthly token volume, min accuracy, data must stay on-premise)
SCENARIOS = [
    ("Support ticket routing at a telco",
     "classification", 10_000_000, 0.80, False),
    ("Hospital extracting entities from patient records (GDPR: on-premise)",
     "ner", 5_000_000, 0.75, True),
    ("Media monitoring firm summarising news at scale",
     "summarization", 50_000_000, 0.20, False),
    ("Bank scoring sentiment on internal client communications (on-premise)",
     "financial_sentiment", 2_000_000, 0.80, True),
    ("SaaS vendor adding an AI coding assistant",
     "code_generation", 1_000_000, 0.50, False),
    ("Defence contractor generating code in an air-gapped environment",
     "code_generation", 1_000_000, 0.30, True),
    ("Legal firm demanding near-perfect summarisation on-premise (unsatisfiable)",
     "summarization", 5_000_000, 0.50, True),
    ("Early-stage startup classifying low-volume user feedback",
     "classification", 100_000, 0.80, False),
    ("E-commerce giant running NER over 100M tokens/month of product data",
     "ner", 100_000_000, 0.85, False),
    ("Hedge fund needing top-accuracy financial sentiment, volume-heavy",
     "financial_sentiment", 20_000_000, 0.90, False),
]


def run_internal_validation(
    matrix_path: str = "results/benchmark_matrix.csv",
    cost_path: str = "results/cost_per_request_matrix.csv",
    output_path: str = "results/internal_validation_per_request.csv",
) -> pd.DataFrame:
    df = load_matrix(matrix_path, cost_path)

    rows = []
    for name, task, volume, threshold, privacy in SCENARIOS:
        requests = round(volume / tokens_per_request(df, task))
        recommended, eligible = recommend(df, task, requests, threshold, privacy)
        rows.append({
            "scenario": name,
            "task": task,
            "monthly_volume_tokens": volume,
            "monthly_requests": requests,
            "min_accuracy": threshold,
            "privacy_required": privacy,
            "recommended_model": recommended["model"] if recommended is not None else "NONE (no model meets constraints)",
            "recommended_method": recommended["method"] if recommended is not None else "-",
            "accuracy": round(recommended["accuracy"], 4) if recommended is not None else None,
            "projected_monthly_cost_usd": (
                round(recommended["projected_monthly_cost"], 2) if recommended is not None else None
            ),
            "fine_tuning_cost_usd": (
                round(recommended["fine_tuning_cost_usd"], 2) if recommended is not None else None
            ),
            "n_eligible_candidates": len(eligible) if recommended is not None else 0,
        })

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    print(f"Internal validation results saved -> {output_path}\n")
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    run_internal_validation()
