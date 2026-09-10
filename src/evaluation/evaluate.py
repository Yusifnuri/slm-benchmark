"""
Evaluation script for fine-tuned SLMs.
Measures accuracy and latency for classification, NER,
summarization, and financial sentiment tasks.
Code generation is handled separately in evaluate_code.py.

Expose: "accuracy, latency (ms/request)" (Section 2, Phase 1 & 2)
"""

import csv
import re
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


CLASSIFICATION_MAX_NEW_TOKENS = 32
NER_MAX_NEW_TOKENS = 64
SUMMARIZATION_MAX_NEW_TOKENS = 128

# Per-instance predictions land here, one CSV per (model, task), so that
# instance-level statistics (paired bootstrap CIs, per-class breakdowns,
# error analysis) can be computed after the fact instead of requiring yet
# another GPU re-run. This was the single biggest analysis gap of the first
# evaluation sweep: only aggregate scores were logged for the fine-tuned
# arm, which restricted the thesis's statistical comparison to task level.
PREDICTIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "predictions",
)


def save_predictions(model_name: str, task: str, rows: List[Dict]) -> str:
    """Write per-instance prediction rows to results/predictions/."""
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    short = model_name.split("/")[-1]
    path = os.path.join(PREDICTIONS_DIR, f"{short}__{task}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-instance predictions saved -> {path}")
    return path


# CoNLL-2003 integer tag ids -> BIO labels (mapping documented in
# TASK_CONFIGS["ner"]; the fine-tuned models emit these ids as text because
# str(ner_tags) is their training target).
NER_ID2TAG = {
    0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC",
}


def _tokens_from_ner_prompt(prompt: str) -> List[str]:
    """Recover the input token list from the NER prompt template."""
    m = re.search(r"from this text:\n(.*)\nEntities:", prompt, re.S)
    return m.group(1).split() if m else []


def _decode_tag_string_to_entities(tag_string: str, tokens: List[str]) -> Tuple[set, bool]:
    """
    '[5, 0, 6, ...]' + input tokens -> set of lowercased entity surface
    forms (types flattened), via BIO decoding. Mirrors the reduction the
    API baseline notebooks apply to their JSON responses (flatten entity
    types, lowercase, exact surface match) so both arms are scored with
    the same instrument. Returns (entity_set, parse_ok).
    """
    ints = re.findall(r"\d+", tag_string)
    if not ints:
        return set(), False
    tags = [NER_ID2TAG.get(int(t), "O") for t in ints]
    entities, current = [], []
    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current:
                entities.append(" ".join(current))
            current = [tok]
        elif tag.startswith("I-") and current:
            current.append(tok)
        else:
            if current:
                entities.append(" ".join(current))
            current = []
    if current:
        entities.append(" ".join(current))
    return set(e.lower().strip() for e in entities), True


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
    predictions, ground_truths, latencies, rows = [], [], [], []
    label_map = config["label_map"]
    valid_labels = list(label_map.values())
    parse_failures = 0

    for i, sample in enumerate(samples):
        pred, lat = generate_prediction(
            model, tokenizer, sample["prompt"], max_new_tokens=CLASSIFICATION_MAX_NEW_TOKENS
        )
        latencies.append(lat)

        # Normalize prediction to valid label. An output containing no valid
        # label is scored INCORRECT (previously it silently defaulted to the
        # first label, which could score an unparseable output as correct
        # whenever the gold label happened to be that first label — a bias in
        # an unknown direction). Unparseable outputs are counted separately so
        # "cannot do the task" and "cannot emit the format" stay
        # distinguishable (same diagnostic the API notebooks keep).
        pred_lower = pred.lower()
        matched = next(
            (lbl for lbl in valid_labels if lbl.lower() in pred_lower), None
        )
        if matched is None:
            parse_failures += 1
        gold = sample["completion"]
        predictions.append(matched if matched is not None else "<unparseable>")
        ground_truths.append(gold)
        rows.append({
            "idx": i,
            "gold": gold,
            "pred": matched if matched is not None else "",
            "raw_prediction": pred[:200],
            "correct": matched == gold,
            "parse_ok": matched is not None,
            "latency_ms": round(lat, 2),
        })

    accuracy = accuracy_score(ground_truths, predictions)
    avg_latency = np.mean(latencies)
    extras = {"parse_failure_rate": round(parse_failures / len(samples), 4)}
    return round(accuracy, 4), round(avg_latency, 2), rows, extras


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
    rouge_scores, latencies, rows = [], [], []

    for i, sample in enumerate(samples):
        pred, lat = generate_prediction(
            model, tokenizer, sample["prompt"], max_new_tokens=SUMMARIZATION_MAX_NEW_TOKENS
        )
        latencies.append(lat)
        score = scorer.score(sample["completion"], pred)
        rouge_scores.append(score["rougeL"].fmeasure)
        rows.append({
            "idx": i,
            "rougeL": round(score["rougeL"].fmeasure, 4),
            "prediction": pred[:500],
            "latency_ms": round(lat, 2),
        })

    return round(np.mean(rouge_scores), 4), round(np.mean(latencies), 2), rows, {}


def evaluate_ner(
    model,
    tokenizer,
    samples: List[Dict],
) -> Tuple[float, float, List[Dict], Dict]:
    """
    Entity-level F1, aligned with the instrument the API baselines use
    (notebooks/03): both prediction and reference are reduced to sets of
    lowercased entity surface forms (entity types flattened), scored per
    sentence and averaged; corpus-level micro P/R/F1 is computed alongside.

    The previous version of this function scored whitespace-token overlap
    between the generated string and str(ner_tags) — overlap between two
    integer-list strings, dominated by the ubiquitous 'O' tag id (0). That
    number tracked output-format imitation, not entity extraction, and was
    not comparable with the API arm's entity-set F1. The fine-tuned models
    emit integer tag lists (their training target); here those are
    BIO-decoded over the input tokens into entity surface forms first.

    Returns (sentence_avg_f1, avg_latency_ms, rows, extras)
    """
    latencies, f1_scores, rows = [], [], []
    micro_tp = micro_fp = micro_fn = 0
    parse_failures = 0

    for i, sample in enumerate(samples):
        pred, lat = generate_prediction(
            model, tokenizer, sample["prompt"], max_new_tokens=NER_MAX_NEW_TOKENS
        )
        latencies.append(lat)
        tokens = _tokens_from_ner_prompt(sample["prompt"])
        gold_set, _ = _decode_tag_string_to_entities(sample["completion"], tokens)
        pred_set, parse_ok = _decode_tag_string_to_entities(pred, tokens)
        if not parse_ok:
            parse_failures += 1

        if not gold_set and not pred_set:
            precision = recall = f1 = 1.0
            tp = fp = fn = 0
        else:
            tp = len(gold_set & pred_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) else 0.0)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        f1_scores.append(f1)
        rows.append({
            "idx": i,
            "gold_entities": "|".join(sorted(gold_set)),
            "pred_entities": "|".join(sorted(pred_set)),
            "raw_prediction": pred[:300],
            "f1": round(f1, 4),
            "parse_ok": parse_ok,
            "latency_ms": round(lat, 2),
        })

    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                if (micro_p + micro_r) else 0.0)
    extras = {
        "micro_f1": round(micro_f1, 4),
        "parse_failure_rate": round(parse_failures / len(samples), 4),
    }
    return round(float(np.mean(f1_scores)), 4), round(float(np.mean(latencies)), 2), rows, extras


def run_full_evaluation(
    base_model_name: str,
    task: str,
    adapter_path: Optional[str] = None,
    financial_phrasebank_path: Optional[str] = None,
    gpu_cost_per_hour: float = 3.99,
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
    if task == "code_generation":
        raise ValueError("Use evaluate_code.py for task: code_generation")

    print(f"\n📊 Evaluating: {base_model_name.split('/')[-1]} | Task: {task}")

    model, tokenizer = load_model_for_inference(base_model_name, adapter_path)

    # Load the held-out test split — never seen during training or during
    # Trainer's periodic "validation" checks — for the numbers that actually
    # get reported in the benchmark matrix.
    eval_ds = SLMDataset(
        task=task,
        split="test",
        tokenizer=tokenizer,
        max_length=512,
        max_samples=max_eval_samples,
        financial_phrasebank_path=financial_phrasebank_path,
    )
    config = TASK_CONFIGS[task]
    samples = eval_ds.samples  # list of {prompt, completion} dicts

    # --- Metric 1 & 2: Accuracy + Latency (plus per-instance rows) ---
    if task in ["classification", "financial_sentiment"]:
        accuracy, avg_latency_ms, pred_rows, extras = evaluate_classification_or_sentiment(
            model, tokenizer, task, samples, config
        )
        generated_tokens = CLASSIFICATION_MAX_NEW_TOKENS
    elif task == "summarization":
        accuracy, avg_latency_ms, pred_rows, extras = evaluate_summarization(
            model, tokenizer, samples
        )
        generated_tokens = SUMMARIZATION_MAX_NEW_TOKENS
    elif task == "ner":
        accuracy, avg_latency_ms, pred_rows, extras = evaluate_ner(
            model, tokenizer, samples
        )
        generated_tokens = NER_MAX_NEW_TOKENS
    else:
        raise ValueError(f"Use evaluate_code.py for task: {task}")

    save_predictions(base_model_name, task, pred_rows)

    # --- Metric 3: Cost per 1M tokens ---
    # Estimate tokens per second from latency using the actual max_new_tokens
    # generated for this task (must match the value passed to generate_prediction above).
    avg_tokens_per_second = generated_tokens / (avg_latency_ms / 1000)
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
        extra_metrics=extras or None,
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