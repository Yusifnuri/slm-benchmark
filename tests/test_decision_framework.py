"""
Tests for src/decision_framework.py and results/cost_per_request_matrix.py.
Pure Python (pandas, sqlite3): no torch, mlflow or GPU needed.

Usage (from the repository root):
    python -m pytest tests/test_decision_framework.py
    python tests/test_decision_framework.py
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "results"))

import numpy as np
import pandas as pd

from cost_per_request_matrix import breakeven_requests, compile_cost_matrix
from src.decision_framework import load_matrix, recommend, tokens_per_request


def _cells():
    # The two cost columns deliberately disagree on order: cost_per_1m_tokens
    # prefers the SLM, per-request cost prefers the cheap API.
    return pd.DataFrame([
        {"model": "slm", "task": "t", "method": "LoRA", "accuracy": 0.90, "privacy_risk": "low",
         "cost_per_1m_tokens": 1.0, "usd_per_request": 0.0005},
        {"model": "cheap-api", "task": "t", "method": "API", "accuracy": 0.85, "privacy_risk": "high",
         "cost_per_1m_tokens": 3.0, "usd_per_request": 0.0001},
        {"model": "big-api", "task": "t", "method": "API", "accuracy": 0.95, "privacy_risk": "high",
         "cost_per_1m_tokens": 10.0, "usd_per_request": 0.0010},
    ])


def test_ranks_on_per_request_cost_not_per_token():
    rec, eligible = recommend(_cells(), "t", 10_000, 0.80, privacy_required=False)
    assert rec["model"] == "cheap-api"
    assert np.isclose(rec["projected_monthly_cost"], 1.0)
    assert len(eligible) == 3


def test_privacy_requirement_keeps_only_self_hosted():
    rec, _ = recommend(_cells(), "t", 10_000, 0.80, privacy_required=True)
    assert rec["model"] == "slm"


def test_unmet_threshold_returns_closest_fallbacks():
    rec, fallback = recommend(_cells(), "t", 10_000, 0.99, privacy_required=False)
    assert rec is None
    assert list(fallback["model"]) == ["big-api", "slm", "cheap-api"]


def test_breakeven_requests():
    assert breakeven_requests(1.0, 0.0001, 0.0011) == 1000
    assert np.isnan(breakeven_requests(1.0, 0.0011, 0.0001))  # SLM never undercuts the API


def test_cost_matrix_reproduces_thesis_breakeven():
    """
    results/cost_per_request.csv (thesis) gives 1,657 requests for
    phi-4-mini-instruct classification vs gpt-4o. The matrix uses MLflow's
    unrounded training_hours (0.154434 h) where the thesis used 0.1544 h, which
    moves the result to 1,658, so a difference of one request is expected.
    """
    os.chdir(REPO)
    costs = compile_cost_matrix()
    assert len(costs) == 30 and costs["usd_per_request"].notna().all()

    phi = costs[(costs["model"] == "phi-4-mini-instruct") & (costs["task"] == "classification")].iloc[0]
    assert abs(phi["breakeven_requests_vs_reference"] - 1657) <= 1

    thesis = pd.read_csv("results/cost_per_request.csv")
    billed = thesis[(thesis["task"] == "classification") & (thesis["chars_per_token"] == "billed")]
    for _, row in billed.iterrows():
        mine = costs[(costs["model"] == row["api"]) & (costs["task"] == "classification")].iloc[0]
        assert np.isclose(mine["usd_per_request"] * 1000, row["api_usd_per_1k_req"], atol=1e-4)


def test_released_matrix_joins_every_cell():
    os.chdir(REPO)
    df = load_matrix()
    assert len(df) == 30
    assert 90 < tokens_per_request(df, "classification") < 130


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {name}")
