"""
One-off diagnostic for Mistral-7B-v0.3's pass@1 = pass@10 = 0.0 on
code_generation — prints a few raw generated solutions instead of just the
aggregate pass@k score, to tell apart a real fine-tuning failure from an
evaluation-side bug (e.g. a prompt/chat-template mismatch specific to
Mistral). Not meant to be run as part of the regular sweep.

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
model = model.merge_and_unload()
model.eval()

for problem in dataset:
    print("=" * 70)
    print(f"Problem: {problem['entry_point']}")
    print(f"--- Prompt ---\n{problem['prompt']}")

    solutions, _ = generate_solutions(model, tokenizer, problem["prompt"], n_samples=N_SAMPLES)
    for i, sol in enumerate(solutions):
        passed = check_solution(problem["prompt"], sol, problem["test"], problem["entry_point"])
        print(f"--- Solution {i+1} (passed={passed}) ---")
        print(sol)
        print()
