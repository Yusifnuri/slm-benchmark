"""
One-off diagnostic for Mistral-7B-v0.3's pass@1 ~= 0 on code_generation.

Round 1 (merge_and_unload() removed in evaluate_code.py) didn't fix it —
pass@1 went from 0.0 to 0.0006 (noise) while avg_latency_ms roughly doubled,
confirming the change took effect without changing the outcome.

Round 2: compares the current sampling config (temperature=0.8, top_p=0.95 —
what evaluate_code.py actually uses) against greedy decoding for the same
problems, to test whether high-temperature sampling interacting with 4-bit
QLoRA's quantization noise is what's producing garbled solutions.

Usage:
    python scripts/debug_mistral_codegen.py
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.evaluation.evaluate_code import generate_solutions, check_solution

N_PROBLEMS = 3
N_SAMPLES = 2

dataset = load_dataset("openai/openai_humaneval", split="test").select(range(N_PROBLEMS))

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.3",
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "./adapters/Mistral-7B-v0.3_code_generation")
model.eval()


def generate_greedy(prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True)


for problem in dataset:
    print("=" * 70)
    print(f"Problem: {problem['entry_point']}")

    print("\n--- Current config: temperature=0.8, top_p=0.95, sampled ---")
    solutions, _ = generate_solutions(model, tokenizer, problem["prompt"], n_samples=N_SAMPLES)
    for i, sol in enumerate(solutions):
        passed = check_solution(problem["prompt"], sol, problem["test"], problem["entry_point"])
        print(f"  Sample {i+1} (passed={passed}):\n{sol}\n")

    print("--- Greedy (do_sample=False) ---")
    greedy_sol = generate_greedy(problem["prompt"])
    from src.evaluation.evaluate_code import truncate_solution
    greedy_sol = truncate_solution(greedy_sol)
    passed = check_solution(problem["prompt"], greedy_sol, problem["test"], problem["entry_point"])
    print(f"  Greedy (passed={passed}):\n{greedy_sol}\n")
