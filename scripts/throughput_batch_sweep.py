"""
Batched-throughput sweep — turns the thesis's batching claim into a
measurement.

The economic sections (§3.5.3, §4.4) currently rely on the literature claim
that continuous batching raises per-GPU throughput by up to an order of
magnitude (Yu et al., OSDI 2022; Kwon et al., SOSP 2023). This script
measures the actual factor for the fine-tuned models on this benchmark's own
classification prompts, at naive static batching — a conservative lower
bound on what a continuous-batching server (vLLM et al.) would achieve.

Batching gain is highly sensitive to request length (a classification
request is ~66 total tokens, a code request thousands), so a single global
factor would mislead — the sweep runs PER TASK, and results are reported in
requests/sec (never tokens/sec: the cost model is per-request, and quoting
throughput per token would reintroduce the denominator problem §3.5.3
exists to kill).

Run on the GPU box (~30 min for all tasks of one model):
    python scripts/throughput_batch_sweep.py \
        --model microsoft/phi-4-mini-instruct \
        --task classification --adapter_path models/phi4_classification
    # repeat per task/adapter, or --task all with a shared adapter layout

Output: results/throughput_batch_sweep.csv — one row per (task, batch size)
with requests/sec, per-request cost, and the speedup factor over batch 1.
Quote the per-task factor in §4.4 in place of the literature bound.
"""

import argparse
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd
import torch

from evaluation.evaluate import (
    load_model_for_inference,
    CLASSIFICATION_MAX_NEW_TOKENS,
    NER_MAX_NEW_TOKENS,
    SUMMARIZATION_MAX_NEW_TOKENS,
)
from data.dataset_loader import SLMDataset

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
N_REQUESTS = 64  # per batch size; small but stable at these latencies
WARMUP_BATCHES = 2

# max_new_tokens must match what evaluation used, or the factor won't
# transfer to the measured per-request costs. code_generation uses MBPP
# test prompts (SLMDataset's code split) at evaluate_code.py's 256 cap.
TASK_TOKENS = {
    "classification": CLASSIFICATION_MAX_NEW_TOKENS,
    "financial_sentiment": CLASSIFICATION_MAX_NEW_TOKENS,
    "ner": NER_MAX_NEW_TOKENS,
    "summarization": SUMMARIZATION_MAX_NEW_TOKENS,
    "code_generation": 256,
}


def sweep(model_name: str, adapter_path: str | None, task: str) -> pd.DataFrame:
    max_new = TASK_TOKENS[task]
    model, tokenizer = load_model_for_inference(model_name, adapter_path)
    tokenizer.padding_side = "left"  # decoder-only batching needs left padding

    ds = SLMDataset(task=task, split="test", tokenizer=tokenizer,
                    max_length=512, max_samples=N_REQUESTS)
    prompts = [s["prompt"] for s in ds.samples]

    rows = []
    base_rps = None
    for bs in BATCH_SIZES:
        batches = [prompts[i:i + bs] for i in range(0, len(prompts), bs)]
        # warm-up (compilation/caches) — excluded from timing
        for b in batches[:WARMUP_BATCHES]:
            enc = tokenizer(b, return_tensors="pt", padding=True,
                            truncation=True, max_length=480).to(model.device)
            with torch.no_grad():
                model.generate(**enc, max_new_tokens=max_new,
                               do_sample=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        start = time.time()
        n_done = 0
        for b in batches:
            enc = tokenizer(b, return_tensors="pt", padding=True,
                            truncation=True, max_length=480).to(model.device)
            with torch.no_grad():
                model.generate(**enc, max_new_tokens=max_new,
                               do_sample=False, pad_token_id=tokenizer.eos_token_id)
            n_done += len(b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        rps = n_done / elapsed
        if base_rps is None:
            base_rps = rps
        rows.append({
            "task": task,
            "batch_size": bs,
            "requests_per_sec": round(rps, 3),
            "speedup_vs_batch1": round(rps / base_rps, 2),
            "usd_per_1k_requests": round(3.99 / 3600 / rps * 1000, 4),
        })
        print(rows[-1])

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out = "results/throughput_batch_sweep.csv"
    # append across tasks/models so one CSV accumulates the whole sweep
    if os.path.exists(out):
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True, choices=list(TASK_TOKENS))
    p.add_argument("--adapter_path", default=None)
    a = p.parse_args()
    sweep(a.model, a.adapter_path, a.task)
