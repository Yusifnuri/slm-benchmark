"""
HumanEval code generation evaluation using pass@k metric.
Expose: "HumanEval/DS-1000 (code generation)" (Section 2, Phase 1)
Reference: Chen et al. 2021 - "Evaluating LLMs Trained on Code"

pass@k = probability that at least 1 of k generated solutions passes all unit tests
"""

import sys
import os
import time
import multiprocessing
import torch
import numpy as np
import mlflow
from typing import List, Tuple, Optional

CODE_EXEC_TIMEOUT_SECONDS = 5

# "fork" copies the current process (including functions defined at runtime,
# e.g. in a Jupyter cell) instead of re-importing __main__ in a fresh
# interpreter. macOS/Windows default to "spawn", which can't find a
# notebook-defined target function and crashes with
# "Can't get attribute '_exec_target' on <module '__main__'>". Linux already
# defaults to fork; this just makes it explicit everywhere.
_MP_CTX = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else multiprocessing.get_context()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator from Chen et al. 2021.

    Args:
        n: total number of generated samples per problem
        c: number of correct samples (passed all tests)
        k: k in pass@k

    Returns:
        pass@k probability for this problem
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(
        [(n - c - i) / (n - i) for i in range(k)]
    ))


def generate_solutions(
    model,
    tokenizer,
    prompt: str,
    n_samples: int = 10,
    max_new_tokens: int = 256,
) -> Tuple[List[str], float]:
    """
    Generate n candidate solutions for one HumanEval problem.

    Returns:
        (list of solution strings, avg latency ms per solution)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    solutions = []
    latencies = []

    for _ in range(n_samples):
        start = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        latencies.append((time.time() - start) * 1000)
        new_tokens = output[0][input_len:]
        solution = tokenizer.decode(new_tokens, skip_special_tokens=True)
        solutions.append(solution)

    return solutions, float(np.mean(latencies))


def _exec_target(prompt: str, solution: str, test_code: str, entry_point: str, result_queue) -> None:
    """Runs in a separate process so a hanging/unsafe generated solution can't
    block or damage the parent (evaluation) process."""
    try:
        exec_globals = {}
        exec(prompt + solution, exec_globals)
        exec(test_code, exec_globals)
        exec(f"check({entry_point})", exec_globals)
        result_queue.put(True)
    except Exception:
        result_queue.put(False)


def check_solution(prompt: str, solution: str, test_code: str, entry_point: str) -> bool:
    """
    Execute a generated solution against HumanEval unit tests in an isolated
    subprocess with a timeout, since model-generated code is untrusted and may
    hang (infinite loops) or misbehave. Returns True if all tests pass.
    """
    result_queue = _MP_CTX.Queue()
    proc = _MP_CTX.Process(
        target=_exec_target,
        args=(prompt, solution, test_code, entry_point, result_queue),
    )
    proc.start()
    proc.join(timeout=CODE_EXEC_TIMEOUT_SECONDS)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False

    return result_queue.get() if not result_queue.empty() else False


def evaluate_humaneval(
    base_model_name: str,
    adapter_path: Optional[str] = None,
    n_samples: int = 10,
    k_values: List[int] = [1, 10],
    max_problems: int = 164,
    mlflow_experiment: str = "phase2_evaluation",
) -> dict:
    """
    Full HumanEval evaluation pipeline.

    Args:
        base_model_name: HuggingFace model ID
        adapter_path: path to LoRA adapter (None = base model)
        n_samples: solutions generated per problem
        k_values: list of k for pass@k metric
        max_problems: number of HumanEval problems (max 164)

    Returns:
        dict with pass@k scores and avg latency
    """
    dataset = load_dataset("openai_humaneval", split="test")
    dataset = dataset.select(range(min(max_problems, len(dataset))))

    # trust_remote_code deliberately omitted — see build_lora_model in
    # train_lora.py for why (breaks phi-4-mini-instruct on current transformers;
    # unneeded for Mistral/Llama's native architectures).
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()

    problem_results = []
    all_latencies = []

    for i, problem in enumerate(dataset):
        print(f"Problem {i+1}/{len(dataset)}", end="\r")
        solutions, avg_lat = generate_solutions(
            model, tokenizer, problem["prompt"], n_samples
        )
        all_latencies.append(avg_lat)

        correct = sum(
            check_solution(
                problem["prompt"], sol,
                problem["test"], problem["entry_point"]
            )
            for sol in solutions
        )
        problem_results.append({"n": n_samples, "c": correct})

    # Calculate pass@k for each k value
    results = {}
    for k in k_values:
        scores = [pass_at_k(r["n"], r["c"], k) for r in problem_results]
        results[f"pass@{k}"] = round(float(np.mean(scores)), 4)

    results["avg_latency_ms"] = round(float(np.mean(all_latencies)), 2)

    # Log to MLflow
    with mlflow.start_run(run_name=f"humaneval_{base_model_name.split('/')[-1]}"):
        mlflow.set_experiment(mlflow_experiment)
        mlflow.set_tags({"model": base_model_name, "task": "code_generation"})
        mlflow.log_metrics(results)

    print(f"\n✅ HumanEval Results: {results}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--max_problems", type=int, default=164)
    args = parser.parse_args()

    evaluate_humaneval(
        base_model_name=args.model,
        adapter_path=args.adapter_path,
        max_problems=args.max_problems,
    )