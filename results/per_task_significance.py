"""
Per-task two-proportion tests: fine-tuned SLM vs each API baseline.

Why this exists alongside statistical_significance.py
-----------------------------------------------------
`statistical_significance.py` pairs on TASK: five per-task accuracies per
model, paired t-test, n=5 pairs. That answers "is there a consistent gap
across task types" and, at n=5, has almost no power — §4.7 of the thesis
says so and declines to use it as evidence in either direction.

It cannot answer the question §4.2.1 actually asks, which is about ONE task:
is the fine-tuned model's accuracy on topic classification distinguishable
from a baseline's? §4.2.1 answered it by observing that one arm's confidence
interval does not contain the other arm's point estimate. That is not an
inference rule — two intervals can each exclude the other's point estimate
while the difference is not significant, and they can overlap while it is.
This script runs the test that rule was standing in for.

Scope: only the tasks where BOTH arms report a proportion over a known
number of instances.
  - classification        accuracy over n instances, both arms       -> tested
  - financial_sentiment   accuracy over n instances, both arms       -> tested
  - ner                   self-hosted cells withdrawn (§4.2.2)       -> skipped
  - summarization         ROUGE-L is a mean of per-instance scores
                          in [0,1], not a proportion of successes    -> skipped
  - code_generation       API arm is greedy pass@1 (a proportion),
                          the fine-tuned arm is the unbiased pass@1
                          estimator over 10 samples per problem
                          (Chen et al., 2021) — not a binomial count,
                          so the two arms are not the same estimator  -> skipped

Caveat that belongs in the thesis text, not just here
-----------------------------------------------------
On classification the two samples are NESTED, not independent: the API arm's
100 instances are a subset of the fine-tuned arm's 200 (§3.2.3). A
two-proportion z-test assumes independence. Because correctness is positively
correlated across models on shared items, the independence assumption
OVERSTATES the standard error, so these p-values are conservative — a
correctly paired analysis (McNemar on the shared 100, plus the disjoint
remainder) would not be larger. That analysis needs per-instance predictions
for the fine-tuned arm, which the first sweep did not persist (§3.10); it
becomes possible once the revised harness writes results/predictions/.

Usage:
    python results/per_task_significance.py
Writes results/per_task_significance.csv and prints the table.
"""

import math

import pandas as pd

MATRIX = "results/benchmark_matrix.csv"
OUT = "results/per_task_significance.csv"

SLMS = ["phi-4-mini-instruct", "Mistral-7B-v0.3", "Llama-3.2-3B-Instruct"]
APIS = ["gpt-4o", "claude-haiku-4-5", "gemini-2.5-flash"]

# Instance counts per arm (§3.2.3). The API arm's n is carried in the matrix;
# the fine-tuned arm's is the evaluation cap in src/evaluation/evaluate.py
# (max_eval_samples=200) and is not stored per row.
SLM_N = {"classification": 200, "financial_sentiment": 200}
TESTABLE = list(SLM_N)

Z = 1.959963985  # two-sided 95%


def _phi(x):
    """Standard normal CDF, so the module has no SciPy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_proportion_z(x1, n1, x2, n2):
    """Pooled-variance two-proportion z-test; returns (z, two-sided p)."""
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (x1 / n1 - x2 / n2) / se
    return z, 2 * (1 - _phi(abs(z)))


def wilson(x, n):
    """Wilson score interval — the same interval §4.2.1 reports per arm."""
    p = x / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return centre - half, centre + half


def newcombe(x1, n1, x2, n2):
    """
    Newcombe's hybrid-score interval for the DIFFERENCE of two proportions.

    Reported instead of a Wald interval because the accuracies here sit near
    the boundary (0.92 at n=200), where Wald intervals are known to
    under-cover. Newcombe (1998), method 10.
    """
    l1, u1 = wilson(x1, n1)
    l2, u2 = wilson(x2, n2)
    p1, p2 = x1 / n1, x2 / n2
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


def holm(pvalues):
    """
    Holm–Bonferroni step-down, monotonicity enforced.

    The family is the nine comparisons WITHIN one task (3 SLMs x 3
    baselines), the same family shape §3.6 declares. Correcting within task
    rather than across all eighteen is the honest reading of that policy: the
    task is the experimental unit each hypothesis is about. Narrowing further
    — to just the three comparisons involving the model one wants to
    highlight — would inflate significance by choosing the family after
    seeing the result, so it is not done here.
    """
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, pvalues[idx] * (m - rank)))
        adjusted[idx] = running
    return adjusted


def run(matrix_path=MATRIX, output_path=OUT):
    df = pd.read_csv(matrix_path)
    rows = []
    for task in TESTABLE:
        sub = df[df.task == task].set_index("model")
        for slm in SLMS:
            n1 = SLM_N[task]
            p1 = float(sub.loc[slm, "accuracy"])
            # Accuracies are stored rounded; recover the success count by
            # rounding p*n, which is exact for every cell in the matrix.
            x1 = round(p1 * n1)
            for api in APIS:
                n2 = int(sub.loc[api, "n_instances"])
                p2 = float(sub.loc[api, "accuracy"])
                x2 = round(p2 * n2)
                z, p = two_proportion_z(x1, n1, x2, n2)
                lo, hi = newcombe(x1, n1, x2, n2)
                rows.append({
                    "task": task, "slm": slm, "llm_baseline": api,
                    "slm_accuracy": round(x1 / n1, 4), "slm_n": n1,
                    "llm_accuracy": round(x2 / n2, 4), "llm_n": n2,
                    "difference": round(x1 / n1 - x2 / n2, 4),
                    "z": round(z, 4), "p_value": round(p, 4),
                    "diff_ci_low": round(lo, 4), "diff_ci_high": round(hi, 4),
                })

    result = pd.DataFrame(rows)
    result["holm_adjusted_p"] = 0.0
    for task in TESTABLE:
        mask = result.task == task
        result.loc[mask, "holm_adjusted_p"] = [
            round(v, 4) for v in holm(list(result.loc[mask, "p_value"]))
        ]
    result["significant_after_holm"] = result.holm_adjusted_p < 0.05
    result.to_csv(output_path, index=False)

    print(f"{len(result)} comparisons, Holm-corrected within each of "
          f"{len(TESTABLE)} tasks (nested samples on classification — see "
          f"module docstring)\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    run()
