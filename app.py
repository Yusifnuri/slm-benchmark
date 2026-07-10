"""
SLM Decision Framework — Streamlit app.
Expose: Deliverable 1, "SLM Decision Framework Tool" (Weeks 15-16)

Inputs (task type, monthly token volume, accuracy threshold, privacy
requirement) -> recommendation (which model, projected cost, accuracy,
ROI breakeven chart), driven entirely by results/benchmark_matrix.csv —
no model inference happens here, this just reads the already-compiled
benchmark numbers and applies the decision logic tree described in the
expose (Weeks 13-14).

Run locally:
    streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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


def implied_fine_tuning_cost(row: pd.Series, llm_cost_per_1m: float) -> float:
    """
    Recovers the fine-tuning cost baked into an already-computed
    roi_breakeven_tokens (rather than storing it as a separate column):
    at volume = roi_breakeven_tokens, SLM total cost == LLM total cost, so
    fine_tuning_cost = (llm_cost - slm_cost) * roi_breakeven_tokens / 1e6.
    Returns 0.0 for API rows or an infinite/missing breakeven (never
    recovers its cost against this reference LLM within any realistic volume).
    """
    breakeven = row["roi_breakeven_tokens"]
    if pd.isna(breakeven) or breakeven in (float("inf"),) or row["privacy_risk"] != "low":
        return 0.0
    return (llm_cost_per_1m - row["cost_per_1m_tokens"]) / 1_000_000 * breakeven


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

eligible = task_df[task_df["accuracy"] >= accuracy_threshold].copy()

if eligible.empty:
    st.warning(
        f"No model reaches {accuracy_threshold:.0%} accuracy on "
        f"{TASK_LABELS[task]} under these constraints. Closest options:"
    )
    eligible = task_df.sort_values("accuracy", ascending=False).head(3)
else:
    eligible["projected_monthly_cost"] = (
        eligible["cost_per_1m_tokens"] / 1_000_000 * monthly_volume
    )
    recommended = eligible.sort_values("projected_monthly_cost").iloc[0]

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
if not eligible.empty and recommended["privacy_risk"] == "low":
    st.subheader(f"ROI breakeven: {recommended['model']} vs cheapest API")
    api_rows = task_df[task_df["privacy_risk"] == "high"]
    if not api_rows.empty:
        cheapest_api = api_rows.sort_values("cost_per_1m_tokens").iloc[0]
        fine_tune_cost = implied_fine_tuning_cost(recommended, cheapest_api["cost_per_1m_tokens"])

        volumes = [v for v in range(0, 20_000_000, 200_000)]
        slm_costs = [fine_tune_cost + recommended["cost_per_1m_tokens"] / 1_000_000 * v for v in volumes]
        api_costs = [cheapest_api["cost_per_1m_tokens"] / 1_000_000 * v for v in volumes]

        fig_roi = go.Figure()
        fig_roi.add_trace(go.Scatter(x=volumes, y=slm_costs, name=f"{recommended['model']} (fine-tuned)", mode="lines"))
        fig_roi.add_trace(go.Scatter(x=volumes, y=api_costs, name=f"{cheapest_api['model']} (API)", mode="lines"))
        fig_roi.add_vline(x=monthly_volume, line_dash="dot", annotation_text="Your monthly volume")
        fig_roi.update_layout(
            xaxis_title="Cumulative monthly tokens",
            yaxis_title="Cumulative cost (USD)",
        )
        st.plotly_chart(fig_roi, use_container_width=True)

        breakeven = recommended["roi_breakeven_tokens"]
        if pd.notna(breakeven) and breakeven != float("inf"):
            st.caption(
                f"Breakeven at ~{breakeven:,.0f} tokens/month against {cheapest_api['model']} — "
                f"below that volume, the API is cheaper; above it, the fine-tuned SLM wins."
            )
        else:
            st.caption(
                f"At this task's inference cost, {recommended['model']} never recovers its "
                f"training cost against {cheapest_api['model']} on a pure per-token basis — "
                f"it's recommended here on accuracy and/or privacy grounds instead."
            )

st.divider()
st.caption(
    "Data source: results/benchmark_matrix.csv — 3 fine-tuned SLMs (Phi-4-mini/LoRA, "
    "Mistral-7B-v0.3/QLoRA, Llama-3.2-3B/LoRA) vs 3 frontier LLM APIs (GPT-4o, Claude Haiku 4.5, "
    "Gemini 2.5 Flash) across 5 enterprise task types."
)
