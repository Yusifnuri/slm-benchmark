"""
Paired t-tests: fine-tuned SLM accuracy vs LLM API baseline accuracy.
Expose: "Statistical significance: paired t-tests on accuracy differences
between SLMs and LLM baselines" (Week 11-12, Milestone M4)

Pairing unit: task. Each of the 5 tasks is evaluated by every model, so a
model's 5 per-task accuracy scores form one paired sample against another
model's 5 per-task accuracy scores — the standard framing when the
underlying per-item (per-test-example) predictions aren't stored, only the
aggregate accuracy per (model, task) in benchmark_matrix.csv.

Caveat (reported alongside every result, not hidden): n=5 pairs per test
is a small sample — this gives limited statistical power. A p-value here
answers "is there a consistent accuracy gap across the 5 task types",
not "would a single held-out example show a significant difference".
"""

import pandas as pd
from scipy import stats

SLMS = ["phi-4-mini-instruct", "Mistral-7B-v0.3", "Llama-3.2-3B-Instruct"]
LLMS = ["gpt-4o", "claude-haiku-4-5", "gemini-2.5-flash"]
TASKS = ["classification", "ner", "summarization", "financial_sentiment", "code_generation"]


def run_significance_tests(
    matrix_path: str = "results/benchmark_matrix.csv",
    output_path: str = "results/statistical_significance.csv",
) -> pd.DataFrame:
    df = pd.read_csv(matrix_path)

    # One accuracy value per (model, task) — pivot for easy paired lookup.
    pivot = df.pivot_table(index="model", columns="task", values="accuracy")
    pivot = pivot[TASKS]  # fixed task order so both sides of each pair line up

    rows = []
    for slm in SLMS:
        for llm in LLMS:
            slm_scores = pivot.loc[slm].values
            llm_scores = pivot.loc[llm].values
            t_stat, p_value = stats.ttest_rel(slm_scores, llm_scores)
            rows.append({
                "slm": slm,
                "llm_baseline": llm,
                "slm_mean_accuracy": round(slm_scores.mean(), 4),
                "llm_mean_accuracy": round(llm_scores.mean(), 4),
                "mean_difference": round(slm_scores.mean() - llm_scores.mean(), 4),
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_value, 4),
                "significant_at_0.05": p_value < 0.05,
                "n_pairs": len(TASKS),
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)
    print(f"Statistical significance results saved -> {output_path}")
    print(result_df.to_string(index=False))
    return result_df


if __name__ == "__main__":
    run_significance_tests()
