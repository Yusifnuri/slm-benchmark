"""
Per-request cost analysis — the unit-consistent economic comparison.

Why this exists: the benchmark matrix's cost_per_1m_tokens columns use two
different token denominators. c_api (blended) is per TOTAL (input+output)
token at an assumed 50/50 mix, while c_slm is per GENERATED token (and the
generated count itself is the task's max_new_tokens cap, an upper bound).
On input-heavy tasks — classification, NER, sentiment — a request is ~97%
input tokens, so the two columns are not in the same unit and their
difference is not a valid Δc. Comparing them produced a breakeven figure
("tokens/month") whose token unit does not exist.

The unit-consistent comparison is per REQUEST:
    c_api_req = (T_in * p_in + T_out * p_out) / 1e6
    c_slm_req = measured_latency_s * R_gpu / 3600
c_slm_req needs no token accounting at all — at batch-1, full-utilisation
serving, a request owns the GPU for its wall-clock duration, which is
exactly what was measured.

Token counts: exact billed counts were not persisted by the Phase 1
notebooks, so T_in/T_out are estimated from the *logged* per-request
character counts across a chars-per-token band (3.8-4.6 for English news
text, 3.0-3.5 for code) and every result is reported across the band.
The pending API re-runs should capture response.usage and replace these
estimates with billed counts.

Outputs: results/cost_per_request.csv + console summary.
"""

import numpy as np
import pandas as pd

R_GPU = 3.99  # USD/hr, H200 single-GPU on-demand (see benchmark_matrix.py)
PRICES = {  # USD per 1M input / output tokens, verified May 2026
    "gpt-4o": (5.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gemini-2.5-flash": (0.30, 2.50),
}
# measured single-request latency, seconds (benchmark_matrix.csv, phi-4-mini)
SLM_LATENCY_S = {"classification": 0.16177, "code_generation": 2.20008}
# compute-only fine-tuning cost at R_GPU (training_hours x rate)
C_FT = {"classification": 0.1544 * R_GPU, "code_generation": 0.012 * R_GPU}

AG_SYSTEM = (
    "You are a news classifier. Classify the text into EXACTLY one of these "
    "categories: World, Sports, Business, Sci/Tech. Reply with ONLY the "
    "category name. No punctuation, no explanation."
)


def classification_char_counts() -> tuple:
    ag = pd.read_csv("logs/ag_news_baseline.csv")
    in_chars = np.array(
        [len(AG_SYSTEM) + len("Text: ") + min(len(str(t)), 300) for t in ag["text"]]
    )
    out_chars = np.array([len(str(p)) for p in ag["openai_pred"]])
    return float(in_chars.mean()), float(out_chars.mean())


def code_char_counts() -> tuple:
    he = pd.read_csv("logs/humaneval_baseline.csv")
    out_chars = float(np.array([len(str(c)) for c in he["openai_code"]]).mean())
    # HumanEval problem prompts were not persisted in the log; 400-800 chars
    # covers the observed range of prompt lengths, + ~200 chars system prompt.
    # Replaced by exact billed counts once the 164-problem re-run captures
    # response.usage.
    in_chars_band = (600.0, 1000.0)
    return in_chars_band, out_chars


def analyse() -> pd.DataFrame:
    rows = []

    in_c, out_c = classification_char_counts()
    for cpt in (3.8, 4.2, 4.6):
        t_in, t_out = in_c / cpt, out_c / cpt
        c_slm = SLM_LATENCY_S["classification"] * R_GPU / 3600
        for api, (pi, po) in PRICES.items():
            c_api = (t_in * pi + t_out * po) / 1e6
            delta = c_api - c_slm
            rows.append({
                "task": "classification", "chars_per_token": cpt, "api": api,
                "T_in": round(t_in, 1), "T_out": round(t_out, 1),
                "api_usd_per_1k_req": round(c_api * 1000, 4),
                "slm_usd_per_1k_req": round(c_slm * 1000, 4),
                "slm_wins": delta > 0,
                "breakeven_requests_compute_only": (
                    round(C_FT["classification"] / delta) if delta > 0 else None
                ),
            })

    (in_lo, in_hi), out_c = code_char_counts()
    for cpt in (3.0, 3.5):
        for in_c2 in (in_lo, in_hi):
            t_in, t_out = in_c2 / cpt, out_c / cpt
            c_slm = SLM_LATENCY_S["code_generation"] * R_GPU / 3600
            for api, (pi, po) in PRICES.items():
                c_api = (t_in * pi + t_out * po) / 1e6
                delta = c_api - c_slm
                rows.append({
                    "task": "code_generation", "chars_per_token": cpt, "api": api,
                    "T_in": round(t_in, 1), "T_out": round(t_out, 1),
                    "api_usd_per_1k_req": round(c_api * 1000, 4),
                    "slm_usd_per_1k_req": round(c_slm * 1000, 4),
                    "slm_wins": delta > 0,
                    "breakeven_requests_compute_only": (
                        round(C_FT["code_generation"] / delta) if delta > 0 else None
                    ),
                })

    df = pd.DataFrame(rows)
    df.to_csv("results/cost_per_request.csv", index=False)
    print(df.to_string(index=False))

    # Batching flip factors for the classification cell (mid band): the
    # single-stream SLM per-request cost divided by k must fall below the
    # API's per-request cost for the comparison to flip.
    mid = df[(df.task == "classification") & (df.chars_per_token == 4.2)]
    c_slm = mid["slm_usd_per_1k_req"].iloc[0]
    print("\nBatching factor k at which the SLM undercuts each API (classification):")
    for _, r in mid.iterrows():
        k = c_slm / r["api_usd_per_1k_req"]
        print(f"  {r['api']:18s} k >= {k:.1f}" if k > 1 else
              f"  {r['api']:18s} already cheaper at k=1")
    return df


if __name__ == "__main__":
    analyse()
