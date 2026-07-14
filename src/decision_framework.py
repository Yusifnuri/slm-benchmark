"""
Decision logic for the SLM Decision Framework.

Extracted into its own module so the Streamlit app (app.py) and the
internal validation script (scripts/internal_validation.py, the expose's
Week 15-16 "internal testing on 10 hypothetical enterprise scenarios")
run the exact same logic — a copy in each would inevitably drift.

Logic tree (expose, Weeks 13-14): filter candidates by task, drop
API models if data must stay on-premise, drop models below the accuracy
threshold, then recommend the cheapest remaining candidate at the given
monthly token volume.
"""

from typing import Optional, Tuple

import pandas as pd

# benchmark_matrix.py computed every SLM row's roi_breakeven_tokens against
# gpt-4o's blended price (its ROI_REFERENCE_API). Any algebra that recovers
# the implied fine-tuning cost from a stored breakeven must use this same
# reference model, or the recovered cost comes out wrong (negative, even,
# whenever the SLM's per-token cost exceeds the cheaper API's).
ROI_REFERENCE_MODEL = "gpt-4o"


def recommend(
    df: pd.DataFrame,
    task: str,
    monthly_volume: int,
    accuracy_threshold: float,
    privacy_required: bool,
) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    """
    Returns (recommended_row_or_None, eligible_candidates).

    recommended is None when no model meets the constraints; in that case
    eligible_candidates holds the closest-by-accuracy fallbacks so a caller
    can show "here's what almost qualifies" instead of nothing.
    """
    candidates = df[df["task"] == task].copy()
    if privacy_required:
        candidates = candidates[candidates["privacy_risk"] == "low"]

    eligible = candidates[candidates["accuracy"] >= accuracy_threshold].copy()
    if eligible.empty:
        return None, candidates.sort_values("accuracy", ascending=False).head(3)

    eligible["projected_monthly_cost"] = (
        eligible["cost_per_1m_tokens"] / 1_000_000 * monthly_volume
    )
    recommended = eligible.sort_values("projected_monthly_cost").iloc[0]
    return recommended, eligible


def reference_api_cost_per_1m(df: pd.DataFrame) -> float:
    """gpt-4o's per-1M-token price as stored in the benchmark matrix itself."""
    return float(df[df["model"] == ROI_REFERENCE_MODEL]["cost_per_1m_tokens"].iloc[0])


def implied_fine_tuning_cost(row: pd.Series, df: pd.DataFrame) -> float:
    """
    Recovers the fine-tuning cost baked into a row's roi_breakeven_tokens:
    at volume = breakeven, SLM total cost == reference-API total cost, so
    fine_tuning_cost = (ref_cost - slm_cost) * breakeven / 1e6.
    Returns 0.0 when there's no finite breakeven to invert (API rows, or
    SLMs whose per-token cost never undercuts the reference API).
    """
    breakeven = row["roi_breakeven_tokens"]
    if pd.isna(breakeven) or breakeven == float("inf") or row["privacy_risk"] != "low":
        return 0.0
    return (reference_api_cost_per_1m(df) - row["cost_per_1m_tokens"]) / 1_000_000 * breakeven
