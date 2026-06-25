"""
Cost and ROI breakeven calculations.
Expose Section 2: "AWS Pricing API + MLflow cost accounting infrastructure"
Expose Sub-question 2: "cost per 1M tokens" and "monthly token volume ROI breakeven"
"""

from typing import Literal

# LLM API costs (USD per 1M tokens) — public pricing as of 2024
# Expose: GPT-4o, Claude-3.5-Haiku, Gemini-1.5-Flash
LLM_API_COSTS = {
    "gpt-4o": {"input": 5.00, "output": 15.00, "blended": 10.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00, "blended": 2.40},
    "gemini-1-5-flash": {"input": 0.075, "output": 0.30, "blended": 0.19},
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