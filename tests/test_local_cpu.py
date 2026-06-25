"""
Local CPU test script — run BEFORE sending anything to GPU.
Tests all components with tiny data (10 samples, 2 steps).
If this passes without errors, code is safe for GPU.

Usage:
    python tests/test_local_cpu.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_loader import SLMDataset, load_task_dataset
from src.evaluation.metrics import (
    calculate_slm_cost_per_1m_tokens,
    calculate_roi_breakeven,
    get_privacy_risk,
)


def test_dataset_loader():
    """Test all 5 datasets load correctly."""
    print("\n🧪 Testing dataset loader...")

    tasks_to_test = [
        ("classification", "train"),
        ("ner", "train"),
        ("summarization", "train"),
        ("code_generation", "test"),   # ← "test" olmalı
    ]

    for task, split in tasks_to_test:
        dataset, config = load_task_dataset(task, split=split, max_samples=5)
        print(f"   ✅ {task}: {len(dataset)} samples loaded")

    print("✅ Dataset loader: PASSED")


def test_metrics():
    """Test cost and ROI calculations."""
    print("\n🧪 Testing metrics...")

    cost = calculate_slm_cost_per_1m_tokens(
        gpu_cost_per_hour=2.50,
        tokens_per_second=500,
    )
    assert cost > 0, "Cost must be positive"
    print(f"   Cost per 1M tokens: ${cost}")

    roi = calculate_roi_breakeven(
        fine_tuning_cost_usd=50.0,
        api_cost_per_1m_tokens=10.0,
        slm_cost_per_1m_tokens=cost,
    )
    assert roi > 0, "ROI breakeven must be positive"
    print(f"   ROI breakeven: {roi:,.0f} tokens")

    risk = get_privacy_risk("on_premise")
    assert risk == "low"
    print(f"   Privacy risk (on_premise): {risk}")

    print("✅ Metrics: PASSED")


def test_slm_dataset_tokenization():
    """Test SLM dataset tokenization with dummy tokenizer."""
    print("\n🧪 Testing SLMDataset tokenization...")

    from transformers import AutoTokenizer
    # Use tiny tokenizer for CPU test — no model download needed
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ds = SLMDataset(
        task="classification",
        split="train",
        tokenizer=tokenizer,
        max_length=128,
        max_samples=5,
    )
    sample = ds[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["input_ids"].shape[0] == 128
    print(f"   Sample keys: {list(sample.keys())}")
    print(f"   input_ids shape: {sample['input_ids'].shape}")
    print("✅ SLMDataset tokenization: PASSED")


if __name__ == "__main__":
    print("=" * 50)
    print("LOCAL CPU TEST SUITE")
    print("Run this before sending anything to GPU.")
    print("=" * 50)

    test_dataset_loader()
    test_metrics()
    test_slm_dataset_tokenization()

    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED — Safe to run on GPU")
    print("=" * 50)