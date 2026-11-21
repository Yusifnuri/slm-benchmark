"""
Per-request cost for every cell of the benchmark matrix.

cost_per_request.py makes the unit-consistent (per-request) comparison for the
two cells the thesis discussed in detail. This script extends the same cost
model to all 30 cells so the decision framework can rank on it instead of on
cost_per_1m_tokens, whose API and SLM columns use different token
denominators (see cost_per_request.py's docstring).

    API:  usd_per_request = (T_in * p_in + T_out * p_out) / 1e6
    SLM:  usd_per_request = latency_s * R_GPU / 3600   (batch-1 serving)

T_in/T_out are the provider-billed means from the Phase 1 logs where those
were persisted (classification, summarisation, financial sentiment, code).
The CoNLL log predates usage logging, so NER is estimated from prompt and
response characters across the same 3.8-4.6 chars-per-token band
cost_per_request.py uses; the 4.2 midpoint is the point estimate and the band
is reported as usd_per_request_lo/hi.

fine_tuning_cost_usd is training_hours (from the phase2_fine_tuning runs in
mlflow.db) x R_GPU, and breakeven_requests_vs_reference is the request volume
at which that one-off cost is recovered against ROI_REFERENCE_MODEL.

Output: results/cost_per_request_matrix.csv. Run from the repository root:
    python results/cost_per_request_matrix.py
"""

import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from evaluation.metrics import LLM_API_COSTS  # noqa: E402

R_GPU = 3.99  # USD/hr, same imputed rate as benchmark_matrix.py
ROI_REFERENCE_MODEL = "gpt-4o"
PROVIDER = {"gpt-4o": "openai", "claude-haiku-4-5": "anthropic", "gemini-2.5-flash": "gemini"}
CHARS_PER_TOKEN_BAND = (3.8, 4.2, 4.6)

# Verbatim from notebooks/03_baseline_conll.ipynb, which sent it with "Text: {text}".
NER_SYSTEM_PROMPT = (
    "You are a Named Entity Recognition system. "
    "Extract all named entities from the text. "
    "Return ONLY a JSON object with keys: PER (persons), ORG (organizations), LOC (locations), MISC (miscellaneous). "
    'Example: {"PER": ["John Smith"], "ORG": ["Apple"], "LOC": ["California"], "MISC": ["FIFA"]}'
)

OUTPUT_COLUMNS = [
    "model", "task", "method", "usd_per_request", "usd_per_request_lo", "usd_per_request_hi",
    "T_in", "T_out", "token_source", "fine_tuning_cost_usd", "breakeven_requests_vs_reference",
]


def api_cost(t_in: float, t_out: float, model: str) -> float:
    prices = LLM_API_COSTS[model]
    return (t_in * prices["input"] + t_out * prices["output"]) / 1e6


def slm_cost(latency_ms: float) -> float:
    return latency_ms / 1000 * R_GPU / 3600


def billed_tokens(log: pd.DataFrame, provider: str):
    """Mean provider-billed (T_in, T_out) over rows that recorded usage, else None."""
    pin, pout = f"{provider}_prompt_tokens", f"{provider}_completion_tokens"
    if pin not in log.columns:
        return None
    billed = log[log[pin].notna() & (log[pin] > 0)]
    if billed.empty:
        return None
    return float(billed[pin].mean()), float(billed[pout].mean())


def ner_char_counts(log: pd.DataFrame, provider: str):
    in_chars = (len(NER_SYSTEM_PROMPT) + len("Text: ") + log["text"].astype(str).str.len()).mean()
    out_chars = log[f"{provider}_response"].astype(str).str.len().mean()
    return float(in_chars), float(out_chars)


def training_hours(db_path: str) -> dict:
    """
    (model, task) -> training_hours from the latest phase2_fine_tuning run, the
    same selection benchmark_matrix.py's _lookup_training_info makes. Reads the
    MLflow store directly so this script needs no mlflow install.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = con.execute(
        """
        SELECT r.start_time, tm.value, tt.value, m.value
        FROM runs r
        JOIN tags tp ON tp.run_uuid = r.run_uuid AND tp.key = 'phase' AND tp.value = 'phase2_fine_tuning'
        JOIN tags tm ON tm.run_uuid = r.run_uuid AND tm.key = 'model'
        JOIN tags tt ON tt.run_uuid = r.run_uuid AND tt.key = 'task'
        JOIN latest_metrics m ON m.run_uuid = r.run_uuid AND m.key = 'training_hours'
        ORDER BY r.start_time
        """
    ).fetchall()
    con.close()
    return {(model.split("/")[-1], task): hours for _, model, task, hours in rows}


def compile_cost_matrix(
    matrix_path: str = "results/benchmark_matrix.csv",
    logs_dir: str = "logs",
    mlflow_db: str = "mlflow.db",
) -> pd.DataFrame:
    matrix = pd.read_csv(matrix_path)
    hours = training_hours(mlflow_db)
    rows = []

    for _, cell in matrix.iterrows():
        row = {"model": cell["model"], "task": cell["task"], "method": cell["method"]}
        if cell["method"] == "API":
            log = pd.read_csv(os.path.join(logs_dir, cell["source"]))
            provider = PROVIDER[cell["model"]]
            billed = billed_tokens(log, provider)
            if billed is not None:
                t_in, t_out = billed
                cost = api_cost(t_in, t_out, cell["model"])
                row.update(usd_per_request=cost, usd_per_request_lo=cost, usd_per_request_hi=cost,
                           T_in=t_in, T_out=t_out, token_source="billed")
            elif cell["task"] == "ner":
                in_c, out_c = ner_char_counts(log, provider)
                band = [api_cost(in_c / cpt, out_c / cpt, cell["model"]) for cpt in CHARS_PER_TOKEN_BAND]
                mid = CHARS_PER_TOKEN_BAND[1]
                row.update(usd_per_request=band[1], usd_per_request_lo=min(band),
                           usd_per_request_hi=max(band), T_in=in_c / mid, T_out=out_c / mid,
                           token_source=f"estimated_{mid}_chars_per_token")
            else:
                raise ValueError(f"No token counts for {cell['model']} / {cell['task']} in {cell['source']}")
            row.update(fine_tuning_cost_usd=0.0)
        else:
            cost = slm_cost(cell["latency_ms"])
            key = (cell["model"], cell["task"])
            if key not in hours:
                raise ValueError(f"No phase2_fine_tuning run with training_hours for {key} in {mlflow_db}")
            row.update(usd_per_request=cost, usd_per_request_lo=cost, usd_per_request_hi=cost,
                       T_in=np.nan, T_out=np.nan, token_source="latency",
                       fine_tuning_cost_usd=hours[key] * R_GPU)
        rows.append(row)

    df = pd.DataFrame(rows)
    reference = df[df["model"] == ROI_REFERENCE_MODEL].set_index("task")["usd_per_request"]
    df["breakeven_requests_vs_reference"] = [
        breakeven_requests(r["fine_tuning_cost_usd"], r["usd_per_request"], reference[r["task"]])
        if r["method"] != "API" else np.nan
        for _, r in df.iterrows()
    ]
    return df[OUTPUT_COLUMNS]


def breakeven_requests(fine_tuning_cost_usd: float, slm_usd_per_request: float,
                       api_usd_per_request: float) -> float:
    """Requests needed to recover the one-off training cost; NaN if the SLM never undercuts the API."""
    saving = api_usd_per_request - slm_usd_per_request
    return round(fine_tuning_cost_usd / saving) if saving > 0 else np.nan


if __name__ == "__main__":
    out = compile_cost_matrix()
    out.to_csv("results/cost_per_request_matrix.csv", index=False)
    print(out.assign(usd_per_1k_requests=out["usd_per_request"] * 1000)
             .drop(columns=["usd_per_request", "usd_per_request_lo", "usd_per_request_hi"])
             .to_string(index=False))
    print("\nSaved -> results/cost_per_request_matrix.csv")
