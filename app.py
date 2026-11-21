"""
SLM Decision Framework — Streamlit app.
Expose: Deliverable 1, "SLM Decision Framework Tool" (Weeks 15-16)

Inputs (task type, monthly request volume, accuracy threshold, privacy
requirement) -> recommendation (which model, projected cost, accuracy,
ROI breakeven chart), driven entirely by results/benchmark_matrix.csv and
results/cost_per_request_matrix.csv — no model inference happens here. The decision logic itself lives in
src/decision_framework.py, shared with scripts/internal_validation.py.

Run locally:
    streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.decision_framework import (
    ROI_REFERENCE_MODEL,
    load_matrix,
    recommend,
    reference_usd_per_request,
    withdrawn,
)

TASKS = ["classification", "ner", "summarization", "financial_sentiment", "code_generation"]
TASK_LABELS = {
    "classification": "Text Classification",
    "ner": "Named Entity Recognition",
    "summarization": "Summarization",
    "financial_sentiment": "Financial Sentiment",
    "code_generation": "Code Generation",
}
VOLUME_OPTIONS = [1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000]


@st.cache_data
def load_benchmark() -> pd.DataFrame:
    return load_matrix()


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
    monthly_requests = st.select_slider(
        "Monthly request volume",
        options=VOLUME_OPTIONS,
        value=100_000,
        format_func=lambda v: f"{v:,}",
    )
    privacy_required = st.toggle("Data must stay on-premise (privacy requirement)")

task_df = df[df["task"] == task].copy()
withheld = withdrawn(df, task)
task_df = task_df[~task_df.index.isin(withheld.index)]
if privacy_required:
    task_df = task_df[task_df["privacy_risk"] == "low"]

recommended, eligible = recommend(df, task, monthly_requests, accuracy_threshold, privacy_required)

if not withheld.empty:
    st.info(
        f"**{len(withheld)} self-hosted {TASK_LABELS[task]} models are withheld from this "
        "recommendation.** Their scores came from a different measurement instrument than the "
        "API models were scored with, so the two are not comparable and the stored numbers do "
        "not represent extraction quality. The evaluation harness has been corrected; these "
        "cells await re-evaluation. Withheld: "
        + ", ".join(sorted(withheld["model"]))
        + "."
    )

if recommended is None:
    st.warning(
        f"No model reaches {accuracy_threshold:.0%} accuracy on "
        f"{TASK_LABELS[task]} under these constraints. Closest options:"
    )
    st.dataframe(
        eligible[["model", "method", "accuracy", "usd_per_request", "privacy_risk"]].reset_index(drop=True),
        use_container_width=True,
    )
else:
    st.success(f"**Recommended: {recommended['model']}** ({recommended['method']})")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{recommended['accuracy']:.1%}")
    m2.metric("Cost / 1k requests", f"${recommended['usd_per_request'] * 1000:.3f}")
    m3.metric("Projected monthly cost", f"${recommended['projected_monthly_cost']:,.2f}")
    m4.metric("Privacy risk", recommended["privacy_risk"])

st.subheader(f"All candidates for {TASK_LABELS[task]}")
display_cols = ["model", "method", "accuracy", "usd_per_request", "latency_ms", "privacy_risk"]
st.dataframe(
    task_df[display_cols].sort_values("accuracy", ascending=False).reset_index(drop=True),
    use_container_width=True,
)

# --- Cost projection chart: all task candidates at the selected volume ---
st.subheader("Projected monthly cost at your request volume")
chart_df = task_df.copy()
chart_df["projected_monthly_cost"] = chart_df["usd_per_request"] * monthly_requests
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
# Plotted against gpt-4o — the same reference API breakeven_requests_vs_reference
# was computed against — so the line intersection on the chart and the
# breakeven number in the caption agree with each other.
if recommended is not None and recommended["privacy_risk"] == "low":
    st.subheader(f"ROI breakeven: {recommended['model']} vs {ROI_REFERENCE_MODEL} (reference API)")
    ref_cost = reference_usd_per_request(df, task)
    fine_tune_cost = recommended["fine_tuning_cost_usd"]

    x_max = max(20_000, int(monthly_requests * 1.2))
    volumes = list(range(0, x_max, max(x_max // 100, 1)))
    slm_costs = [fine_tune_cost + recommended["usd_per_request"] * v for v in volumes]
    api_costs = [ref_cost * v for v in volumes]

    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(x=volumes, y=slm_costs, name=f"{recommended['model']} (fine-tuned)", mode="lines"))
    fig_roi.add_trace(go.Scatter(x=volumes, y=api_costs, name=f"{ROI_REFERENCE_MODEL} (API)", mode="lines"))
    fig_roi.add_vline(x=monthly_requests, line_dash="dot", annotation_text="Your monthly volume")
    fig_roi.update_layout(
        xaxis_title="Cumulative requests",
        yaxis_title="Cumulative cost (USD)",
    )
    st.plotly_chart(fig_roi, use_container_width=True)

    breakeven = recommended["breakeven_requests_vs_reference"]
    if pd.notna(breakeven):
        st.caption(
            f"Breakeven at ~{breakeven:,.0f} requests against {ROI_REFERENCE_MODEL} — "
            f"below that volume, the API is cheaper; above it, the fine-tuned SLM wins."
        )
    else:
        st.caption(
            f"At batch-1 serving, {recommended['model']} costs more per request than "
            f"{ROI_REFERENCE_MODEL} on this task, so it never recovers its training cost — "
            f"it's recommended here on accuracy and/or privacy grounds instead."
        )

st.divider()
st.caption(
    "Data source: results/benchmark_matrix.csv — 3 fine-tuned SLMs (Phi-4-mini/LoRA, "
    "Mistral-7B-v0.3/QLoRA, Llama-3.2-3B/LoRA) vs 3 frontier LLM APIs (GPT-4o, Claude Haiku 4.5, "
    "Gemini 2.5 Flash) across 5 enterprise task types."
)
