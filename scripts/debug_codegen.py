"""
One-off diagnostic for inspecting raw HumanEval generations for any
fine-tuned model — prints the current sampled config side by side with
greedy decoding, for a handful of problems (not the full 164x10 sweep).

Originally written to debug Mistral-7B-v0.3 (QLoRA)'s pass@1 ~= 0. Neither
removing merge_and_unload() nor (per prior findings) quantization alone
fully explained it — dataset_loader.py's SLMDataset never appends an EOS
token after the completion (same code path for all 3 models), so none of
them get an explicit "stop here" training signal. phi-4-mini-instruct and
Llama-3.2-3B-Instruct are already instruction-tuned before our LoRA
fine-tune and may fall back on that prior behavior; Mistral-7B-v0.3 is a
base (non-instruct) checkpoint with nothing to fall back on. This script
lets the same raw-output check run against any of the 3 models to see
whether phi4/Llama also ramble past the correct answer (just less
catastrophically) or genuinely stop cleanly.

Usage:
    python scripts/debug_codegen.py --model microsoft/phi-4-mini-instruct --adapter_path ./adapters/phi-4-mini-instruct_code_generation
    python scripts/debug_codegen.py --model meta-llama/Llama-3.2-3B-Instruct --adapter_path ./adapters/Llama-3.2-3B-Instruct_code_generation
    python scripts/debug_codegen.py --model mistralai/Mistral-7B-v0.3 --adapter_path ./adapters/Mistral-7B-v0.3_code_generation
"""

import sys
import os
import argparse
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.evaluation.evaluate_code import generate_solutions, check_solution, truncate_solution

N_PROBLEMS = 3
N_SAMPLES = 2

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--adapter_path", required=True)
args = parser.parse_args()

dataset = load_dataset("openai/openai_humaneval", split="test").select(range(N_PROBLEMS))

tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, args.adapter_path)
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
    print(f"Model: {args.model} | Problem: {problem['entry_point']}")

    print("\n--- Current config: temperature=0.8, top_p=0.95, sampled ---")
    solutions, _ = generate_solutions(model, tokenizer, problem["prompt"], n_samples=N_SAMPLES)
    for i, sol in enumerate(solutions):
        passed = check_solution(problem["prompt"], sol, problem["test"], problem["entry_point"])
        print(f"  Sample {i+1} (passed={passed}):\n{sol}\n")

    print("--- Greedy (do_sample=False) ---")
    greedy_sol = truncate_solution(generate_greedy(problem["prompt"]))
    passed = check_solution(problem["prompt"], greedy_sol, problem["test"], problem["entry_point"])
    print(f"  Greedy (passed={passed}):\n{greedy_sol}\n")
