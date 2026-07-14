"""
SLM Decision Framework — Streamlit app.
Expose: Deliverable 1, "SLM Decision Framework Tool" (Weeks 15-16)

Inputs (task type, monthly token volume, accuracy threshold, privacy
requirement) -> recommendation (which model, projected cost, accuracy,
ROI breakeven chart), driven entirely by results/benchmark_matrix.csv —
no model inference happens here. The decision logic itself lives in
src/decision_framework.py, shared with scripts/internal_validation.py.

Run locally:
    streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.decision_framework import (
    ROI_REFERENCE_MODEL,
    implied_fine_tuning_cost,
    recommend,
    reference_api_cost_per_1m,
)

TASKS = ["classification", "ner", "summarization", "financial_sentiment", "code_generation"]
TASK_LABELS = {
    "classification": "Text Classification",
    "ner": "Named Entity Recognition",
    "summarization": "Summarization",
    "financial_sentiment": "Financial Sentiment",
    "code_generation": "Code Generation",
}
VOLUME_OPTIONS = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000]


@st.cache_data
def load_benchmark() -> pd.DataFrame:
    return pd.read_csv("results/benchmark_matrix.csv")


st.set_page_config(page_title="SLM Decision Framework", layout="wide")
st.title("SLM Decision Framework")
st.caption(
    "Fine-Tune or Pay Per Token? Enter your enterprise use case below for a "
    "data-driven recommendation, backed by the accuracy/latency/cost/privacy/ROI "
    "benchmark of 3 fine-tuned SLMs vs 3 frontier LLM APIs across 5 task types."
)

df = load_benchmark()

col1, col2 = st.columns(2)
with col1:
    task = st.selectbox("Task type", TASKS, format_func=lambda t: TASK_LABELS[t])
    accuracy_threshold = st.slider("Minimum accuracy required", 0.0, 1.0, 0.70, 0.01)
with col2:
    monthly_volume = st.select_slider(
        "Monthly token volume",
        options=VOLUME_OPTIONS,
        value=5_000_000,
        format_func=lambda v: f"{v:,}",
    )
    privacy_required = st.toggle("Data must stay on-premise (privacy requirement)")

task_df = df[df["task"] == task].copy()
if privacy_required:
    task_df = task_df[task_df["privacy_risk"] == "low"]

recommended, eligible = recommend(df, task, monthly_volume, accuracy_threshold, privacy_required)

if recommended is None:
    st.warning(
        f"No model reaches {accuracy_threshold:.0%} accuracy on "
        f"{TASK_LABELS[task]} under these constraints. Closest options:"
    )
    st.dataframe(
        eligible[["model", "method", "accuracy", "cost_per_1m_tokens", "privacy_risk"]].reset_index(drop=True),
        use_container_width=True,
    )
else:
    st.success(f"**Recommended: {recommended['model']}** ({recommended['method']})")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{recommended['accuracy']:.1%}")
    m2.metric("Cost / 1M tokens", f"${recommended['cost_per_1m_tokens']:.2f}")
    m3.metric("Projected monthly cost", f"${recommended['projected_monthly_cost']:,.2f}")
    m4.metric("Privacy risk", recommended["privacy_risk"])

st.subheader(f"All candidates for {TASK_LABELS[task]}")
display_cols = ["model", "method", "accuracy", "cost_per_1m_tokens", "latency_ms", "privacy_risk"]
st.dataframe(
    task_df[display_cols].sort_values("accuracy", ascending=False).reset_index(drop=True),
    use_container_width=True,
)

# --- Cost projection chart: all task candidates at the selected volume ---
st.subheader("Projected monthly cost at your token volume")
chart_df = task_df.copy()
chart_df["projected_monthly_cost"] = chart_df["cost_per_1m_tokens"] / 1_000_000 * monthly_volume
chart_df = chart_df.sort_values("projected_monthly_cost")

fig_cost = go.Figure(
    go.Bar(
        x=chart_df["model"],
        y=chart_df["projected_monthly_cost"],
        marker_color=["#2E86AB" if r == "low" else "#E76F51" for r in chart_df["privacy_risk"]],
        text=[f"${v:,.0f}" for v in chart_df["projected_monthly_cost"]],
        textposition="outside",
    )
)
fig_cost.update_layout(
    xaxis_title="Model",
    yaxis_title="Projected monthly cost (USD)",
    showlegend=False,
)
st.plotly_chart(fig_cost, use_container_width=True)
st.caption("Blue = on-premise (low privacy risk), orange = API (high privacy risk)")

# --- ROI breakeven chart, only meaningful when an SLM was recommended ---
# Plotted against gpt-4o — the same reference API the stored
# roi_breakeven_tokens was computed against — so the line intersection on
# the chart and the breakeven number in the caption agree with each other.
if recommended is not None and recommended["privacy_risk"] == "low":
    st.subheader(f"ROI breakeven: {recommended['model']} vs {ROI_REFERENCE_MODEL} (reference API)")
    ref_cost = reference_api_cost_per_1m(df)
    fine_tune_cost = implied_fine_tuning_cost(recommended, df)

    x_max = max(20_000_000, int(monthly_volume * 1.2))
    volumes = list(range(0, x_max, max(x_max // 100, 1)))
    slm_costs = [fine_tune_cost + recommended["cost_per_1m_tokens"] / 1_000_000 * v for v in volumes]
    api_costs = [ref_cost / 1_000_000 * v for v in volumes]

    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(x=volumes, y=slm_costs, name=f"{recommended['model']} (fine-tuned)", mode="lines"))
    fig_roi.add_trace(go.Scatter(x=volumes, y=api_costs, name=f"{ROI_REFERENCE_MODEL} (API)", mode="lines"))
    fig_roi.add_vline(x=monthly_volume, line_dash="dot", annotation_text="Your monthly volume")
    fig_roi.update_layout(
        xaxis_title="Cumulative monthly tokens",
        yaxis_title="Cumulative cost (USD)",
    )
    st.plotly_chart(fig_roi, use_container_width=True)

    breakeven = recommended["roi_breakeven_tokens"]
    if pd.notna(breakeven) and breakeven != float("inf"):
        st.caption(
            f"Breakeven at ~{breakeven:,.0f} tokens/month against {ROI_REFERENCE_MODEL} — "
            f"below that volume, the API is cheaper; above it, the fine-tuned SLM wins."
        )
    else:
        st.caption(
            f"At this task's inference cost, {recommended['model']} never recovers its "
            f"training cost against {ROI_REFERENCE_MODEL} on a pure per-token basis — "
            f"it's recommended here on accuracy and/or privacy grounds instead."
        )

st.divider()
st.caption(
    "Data source: results/benchmark_matrix.csv — 3 fine-tuned SLMs (Phi-4-mini/LoRA, "
    "Mistral-7B-v0.3/QLoRA, Llama-3.2-3B/LoRA) vs 3 frontier LLM APIs (GPT-4o, Claude Haiku 4.5, "
    "Gemini 2.5 Flash) across 5 enterprise task types."
)
