"""
Cost and ROI breakeven calculations.
Expose Section 2: "AWS Pricing API + MLflow cost accounting infrastructure"
Expose Sub-question 2: "cost per 1M tokens" and "monthly token volume ROI breakeven"
"""

from typing import Literal

# LLM API costs (USD per 1M tokens).
#
# The expose names Claude-3.5-Haiku and Gemini-1.5-Flash as baselines, but the
# Phase 1 notebooks (notebooks/02-06) actually call claude-haiku-4-5 and
# gemini-2.5-flash — see README.md "Methodology notes" for why (older models
# deprecated/unavailable by the time baselines were run). Pricing here must
# match whatever model the notebooks actually queried, or the ROI/cost
# numbers are computed against the wrong price. Verified current pricing:
# - claude-haiku-4-5: $1.00 / $5.00 per 1M input/output tokens
# - gemini-2.5-flash: $0.30 / $2.50 per 1M input/output tokens (standard tier)
LLM_API_COSTS = {
    "gpt-4o": {"input": 5.00, "output": 15.00, "blended": 10.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "blended": 3.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "blended": 1.40},
}

# Privacy risk by deployment type
# Expose Sub-question 3: "data-leaves-premises exposure"
PRIVACY_RISK = {
    "api": "high",           # data sent to third-party servers
    "on_premise": "low",     # data never leaves company infrastructure
    "private_cloud": "medium",
}


def calculate_slm_cost_per_1m_tokens(
    gpu_cost_per_hour: float,
    tokens_per_second: float,
) -> float:
    """
    Calculate on-premise SLM inference cost per 1M tokens.

    Args:
        gpu_cost_per_hour: AWS instance hourly cost (e.g. $2.50 for A6000)
        tokens_per_second: measured inference throughput of the SLM

    Returns:
        Cost in USD per 1M tokens
    """
    tokens_per_hour = tokens_per_second * 3600
    cost_per_token = gpu_cost_per_hour / tokens_per_hour
    return round(cost_per_token * 1_000_000, 4)


def calculate_roi_breakeven(
    fine_tuning_cost_usd: float,
    api_cost_per_1m_tokens: float,
    slm_cost_per_1m_tokens: float,
) -> float:
    """
    Calculate total token volume at which fine-tuning becomes cost-positive.

    Formula: breakeven = fine_tuning_cost / (api_cost - slm_cost) per token
    Expose: "monthly token volume at which fine-tuning becomes cost-positive"

    Args:
        fine_tuning_cost_usd: total GPU training cost in USD
        api_cost_per_1m_tokens: LLM API cost per 1M tokens (blended)
        slm_cost_per_1m_tokens: on-premise SLM inference cost per 1M tokens

    Returns:
        Breakeven volume in total tokens (float)
        Returns inf if SLM is more expensive than API
    """
    saving_per_1m = api_cost_per_1m_tokens - slm_cost_per_1m_tokens
    if saving_per_1m <= 0:
        return float("inf")  # SLM never beats API on cost
    breakeven_millions = fine_tuning_cost_usd / saving_per_1m
    return round(breakeven_millions * 1_000_000, 0)


def get_privacy_risk(deployment_type: str) -> str:
    """Return privacy risk level for a deployment type."""
    return PRIVACY_RISK.get(deployment_type, "unknown")


def calculate_training_cost(
    training_hours: float,
    gpu_cost_per_hour: float = 2.50,
) -> float:
    """Calculate total GPU training cost in USD."""
    return round(training_hours * gpu_cost_per_hour, 2)