"""
Decision logic for the SLM Decision Framework.

Extracted into its own module so the Streamlit app (app.py) and the
internal validation script (scripts/internal_validation.py, the expose's
Week 15-16 "internal testing on 10 hypothetical enterprise scenarios")
run the exact same logic — a copy in each would inevitably drift.

Logic tree (expose, Weeks 13-14): filter candidates by task, drop
API models if data must stay on-premise, drop models below the accuracy
threshold, then recommend the cheapest remaining candidate at the given
monthly request volume.

Cost is per request (results/cost_per_request_matrix.csv), not per 1M tokens:
the matrix's cost_per_1m_tokens divides API cost by total tokens but SLM cost
by generated tokens, so ranking on it compares two different units (see
results/cost_per_request.py). The thesis tool ranked on cost_per_1m_tokens.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Reference API for the breakeven shown alongside a self-hosted recommendation;
# the same model results/cost_per_request_matrix.py computes breakevens against.
ROI_REFERENCE_MODEL = "gpt-4o"

# Cells whose stored score is not a valid measurement, and which must therefore
# never drive a recommendation. In the first evaluation sweep the self-hosted
# named-entity-recognition arm was scored by whitespace-token overlap between
# generated and reference integer tag strings -- a quantity dominated by the
# majority 'O' tag, so it measured output-format imitation rather than entity
# extraction, and was not comparable with the API arm's entity-set F1 (thesis
# SS3.5.1, SS4.2.2). The harness has since been corrected to BIO-decode both
# arms identically, and the three affected cells have been re-measured under
# it, so none remain withdrawn as of this run.
#
# Keyed by (task, method) because the defect is a property of how that arm was
# scored, not of any individual model. Add an entry back here if a future
# instrument defect is found, and remove it once that cell is re-measured.
WITHDRAWN_CELLS: set = set()


def load_matrix(
    matrix_path: str = "results/benchmark_matrix.csv",
    cost_path: str = "results/cost_per_request_matrix.csv",
) -> pd.DataFrame:
    """The benchmark matrix with each cell's per-request cost columns joined on."""
    matrix = pd.read_csv(matrix_path)
    costs = pd.read_csv(cost_path).drop(columns=["method"])
    merged = matrix.merge(costs, on=["model", "task"], how="left", validate="one_to_one")
    missing = merged[merged["usd_per_request"].isna()]
    if not missing.empty:
        cells = ", ".join(f"{m}/{t}" for m, t in zip(missing["model"], missing["task"]))
        raise ValueError(f"No per-request cost for {cells}; re-run results/cost_per_request_matrix.py")
    return merged


def withdrawn(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Rows for `task` that are excluded from recommendation as unmeasured."""
    mask = df.apply(lambda r: (r["task"], r["method"]) in WITHDRAWN_CELLS, axis=1)
    return df[mask & (df["task"] == task)]


def recommend(
    df: pd.DataFrame,
    task: str,
    monthly_requests: int,
    accuracy_threshold: float,
    privacy_required: bool,
) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    """
    Returns (recommended_row_or_None, eligible_candidates).

    `df` must carry usd_per_request (see load_matrix). The ranking key,
    projected_monthly_cost, is the marginal serving cost: usd_per_request x
    monthly_requests. A self-hosted model's one-off fine-tuning cost is left
    out of it and reported separately (fine_tuning_cost_usd,
    breakeven_requests_vs_reference).

    recommended is None when no model meets the constraints; in that case
    eligible_candidates holds the closest-by-accuracy fallbacks so a caller
    can show "here's what almost qualifies" instead of nothing.

    Cells listed in WITHDRAWN_CELLS are dropped before any filtering: a score
    that is not a valid measurement cannot support a recommendation, and
    silently ranking on one would put the instrument defect the thesis reports
    straight back into the advice the tool gives. Callers that want to explain
    the absence to a user can list them with withdrawn().
    """
    candidates = df[df["task"] == task].copy()
    candidates = candidates[
        ~candidates.apply(lambda r: (r["task"], r["method"]) in WITHDRAWN_CELLS, axis=1)
    ]
    if privacy_required:
        candidates = candidates[candidates["privacy_risk"] == "low"]

    eligible = candidates[candidates["accuracy"] >= accuracy_threshold].copy()
    if eligible.empty:
        return None, candidates.sort_values("accuracy", ascending=False).head(3)

    eligible["projected_monthly_cost"] = eligible["usd_per_request"] * monthly_requests
    recommended = eligible.sort_values("projected_monthly_cost").iloc[0]
    return recommended, eligible


def reference_usd_per_request(df: pd.DataFrame, task: str) -> float:
    """ROI_REFERENCE_MODEL's per-request cost on `task`."""
    ref = df[(df["model"] == ROI_REFERENCE_MODEL) & (df["task"] == task)]
    return float(ref["usd_per_request"].iloc[0])


def tokens_per_request(df: pd.DataFrame, task: str) -> float:
    """Mean API tokens (input + output) per request on `task`, across the API rows."""
    api = df[(df["task"] == task) & (df["method"] == "API")]
    return float(np.mean(api["T_in"] + api["T_out"]))
