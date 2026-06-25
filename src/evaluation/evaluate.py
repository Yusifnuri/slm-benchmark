"""
Evaluation script for fine-tuned SLMs.
Measures accuracy and latency for classification, NER,
summarization, and financial sentiment tasks.
Code generation is handled separately in evaluate_code.py.

Expose: "accuracy, latency (ms/request)" (Section 2, Phase 1 & 2)
"""

import sys
import os
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score
from rouge_score import rouge_scorer

from data.dataset_loader import SLMDataset, TASK_CONFIGS
from utils.mlflow_logger import BenchmarkLogger
from evaluation.metrics import (
    calculate_slm_cost_per_1m_tokens,
    calculate_roi_breakeven,
    get_privacy_risk,
    LLM_API_COSTS,
)


def load_model_for_inference(
    base_model_name: str,
    adapter_path: Optional[str] = None,
):
    """
    Load model for inference.
    If adapter_path provided: load fine-tuned SLM.
    If None: load base model (for baseline comparison).
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # merge adapter into base for faster inference

    model.eval()
    return model, tokenizer


def generate_prediction(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
) -> Tuple[str, float]:
    """
    Generate a single prediction and measure latency.

    Returns:
        (predicted_text, latency_ms)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=480)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy decoding for reproducibility
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.time() - start) * 1000

    # Decode only new tokens (strip prompt)
    input_len = inputs["input_ids"].shape[1]
    new_tokens = output[0][input_len:]
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return prediction, latency_ms


def evaluate_classification_or_sentiment(
    model,
    tokenizer,
    task: str,
    samples: List[Dict],
    config: Dict,
) -> Tuple[float, float]:
    """
    Evaluate classification or financial_sentiment task.
    Returns (accuracy, avg_latency_ms)
    """
    predictions, ground_truths, latencies = [], [], []
    label_map = config["label_map"]
    valid_labels = list(label_map.values())

    for sample in samples:
        pred, lat = generate_prediction(model, tokenizer, sample["prompt"])
        latencies.append(lat)

        # Normalize prediction to valid label
        pred_lower = pred.lower()
        matched = next(
            (lbl for lbl in valid_labels if lbl.lower() in pred_lower),
            valid_labels[0],  # default to first label if no match
        )
        predictions.append(matched)
        ground_truths.append(sample["completion"])

    accuracy = accuracy_score(ground_truths, predictions)
    avg_latency = np.mean(latencies)
    return round(accuracy, 4), round(avg_latency, 2)


def evaluate_summarization(
    model,
    tokenizer,
    samples: List[Dict],
) -> Tuple[float, float]:
    """
    Evaluate summarization task using ROUGE-L score.
    Returns (rouge_l, avg_latency_ms)
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores, latencies = [], []

    for sample in samples:
        pred, lat = generate_prediction(model, tokenizer, sample["prompt"], max_new_tokens=128)
        latencies.append(lat)
        score = scorer.score(sample["completion"], pred)
        rouge_scores.append(score["rougeL"].fmeasure)

    return round(np.mean(rouge_scores), 4), round(np.mean(latencies), 2)


def evaluate_ner(
    model,
    tokenizer,
    samples: List[Dict],
) -> Tuple[float, float]:
    """
    Evaluate NER task using token-level F1 score.
    Returns (f1, avg_latency_ms)
    """
    all_preds, all_true, latencies = [], [], []

    for sample in samples:
        pred, lat = generate_prediction(model, tokenizer, sample["prompt"], max_new_tokens=64)
        latencies.append(lat)
        # Simple token overlap F1
        pred_tokens = set(pred.lower().split())
        true_tokens = set(sample["completion"].lower().split())
        all_preds.append(pred_tokens)
        all_true.append(true_tokens)

    # Token-level F1
    f1_scores = []
    for pred_set, true_set in zip(all_preds, all_true):
        if not true_set:
            continue
        tp = len(pred_set & true_set)
        precision = tp / len(pred_set) if pred_set else 0
        recall = tp / len(true_set)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)
        f1_scores.append(f1)

    return round(np.mean(f1_scores), 4), round(np.mean(latencies), 2)


def run_full_evaluation(
    base_model_name: str,
    task: str,
    adapter_path: Optional[str] = None,
    financial_phrasebank_path: Optional[str] = None,
    gpu_cost_per_hour: float = 2.50,
    fine_tuning_cost_usd: float = 0.0,
    compare_against_llm: str = "gpt-4o",
    mlflow_experiment: str = "phase2_evaluation",
    max_eval_samples: int = 200,
):
    """
    Full evaluation pipeline for one model-task combination.
    Computes all 5 metrics required by the expose.
    Logs results to MLflow.
    """
    print(f"\n📊 Evaluating: {base_model_name.split('/')[-1]} | Task: {task}")

    model, tokenizer = load_model_for_inference(base_model_name, adapter_path)

    # Load evaluation dataset
    eval_ds = SLMDataset(
        task=task,
        split="test" if task == "code_generation" else "validation",
        tokenizer=tokenizer,
        max_length=512,
        max_samples=max_eval_samples,
        financial_phrasebank_path=financial_phrasebank_path,
    )
    config = TASK_CONFIGS[task]
    samples = eval_ds.samples  # list of {prompt, completion} dicts

    # --- Metric 1 & 2: Accuracy + Latency ---
    if task in ["classification", "financial_sentiment"]:
        accuracy, avg_latency_ms = evaluate_classification_or_sentiment(
            model, tokenizer, task, samples, config
        )
    elif task == "summarization":
        accuracy, avg_latency_ms = evaluate_summarization(model, tokenizer, samples)
    elif task == "ner":
        accuracy, avg_latency_ms = evaluate_ner(model, tokenizer, samples)
    else:
        raise ValueError(f"Use evaluate_code.py for task: {task}")

    # --- Metric 3: Cost per 1M tokens ---
    # Estimate tokens per second from latency
    avg_tokens_per_second = 32 / (avg_latency_ms / 1000)
    cost_per_1m = calculate_slm_cost_per_1m_tokens(gpu_cost_per_hour, avg_tokens_per_second)

    # --- Metric 4: Privacy risk ---
    privacy_risk = get_privacy_risk("on_premise")  # SLMs are on-premise

    # --- Metric 5: ROI breakeven ---
    api_cost = LLM_API_COSTS[compare_against_llm]["blended"]
    roi_breakeven = calculate_roi_breakeven(fine_tuning_cost_usd, api_cost, cost_per_1m)

    # --- Log to MLflow ---
    logger = BenchmarkLogger(mlflow_experiment)
    logger.log_benchmark_result(
        model_name=base_model_name,
        task=task,
        accuracy=accuracy,
        latency_ms=avg_latency_ms,
        cost_per_1m_tokens=cost_per_1m,
        privacy_risk=privacy_risk,
        roi_breakeven_tokens=roi_breakeven,
    )

    print(f"\n✅ Results:")
    print(f"   Accuracy/ROUGE-L/F1 : {accuracy}")
    print(f"   Avg Latency (ms)     : {avg_latency_ms}")
    print(f"   Cost per 1M tokens   : ${cost_per_1m}")
    print(f"   Privacy risk         : {privacy_risk}")
    print(f"   ROI breakeven tokens : {roi_breakeven:,.0f}")

    return {
        "accuracy": accuracy,
        "latency_ms": avg_latency_ms,
        "cost_per_1m_tokens": cost_per_1m,
        "privacy_risk": privacy_risk,
        "roi_breakeven_tokens": roi_breakeven,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--financial_path", default=None)
    parser.add_argument("--fine_tuning_cost", type=float, default=50.0)
    parser.add_argument("--compare_llm", default="gpt-4o")
    args = parser.parse_args()

    run_full_evaluation(
        base_model_name=args.model,
        task=args.task,
        adapter_path=args.adapter_path,
        financial_phrasebank_path=args.financial_path,
        fine_tuning_cost_usd=args.fine_tuning_cost,
        compare_against_llm=args.compare_llm,
    )