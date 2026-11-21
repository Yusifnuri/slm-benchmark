"""
Tests for src/training/run_utils.py. Pure Python: no torch, transformers or GPU needed.

Usage:
    python -m pytest tests/test_run_utils.py
    python tests/test_run_utils.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.training.run_utils import (
    adapter_dir,
    check_lora_targets,
    run_name,
    seed_from_adapter_path,
    unmatched_targets,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Attention/MLP projection names as defined in transformers' modeling_phi3.py
# and modeling_llama.py (Mistral uses the same names as Llama).
PHI3_PROJ = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
LLAMA_PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def module_names(proj_names, n_layers=2):
    """Qualified module names shaped like a decoder-only HF model."""
    names = ["", "model", "model.embed_tokens", "lm_head"]
    for i in range(n_layers):
        names += [f"model.layers.{i}", f"model.layers.{i}.self_attn", f"model.layers.{i}.mlp"]
        for p in proj_names:
            parent = "mlp" if p in ("gate_up_proj", "gate_proj", "up_proj", "down_proj") else "self_attn"
            names.append(f"model.layers.{i}.{parent}.{p}")
    return names


class FakeModel:
    def __init__(self, names):
        self._names = names

    def named_modules(self):
        return ((n, None) for n in self._names)


def config_targets(path):
    with open(os.path.join(REPO, path)) as f:
        cfg = yaml.safe_load(f)
    return (cfg.get("lora") or cfg.get("qlora"))["target_modules"]


def test_thesis_phi_targets_are_caught():
    # The thesis config: only o_proj exists in Phi-3, the other three matched nothing.
    missing = unmatched_targets(module_names(PHI3_PROJ), ["q_proj", "v_proj", "k_proj", "o_proj"])
    assert missing == ["q_proj", "v_proj", "k_proj"]
    try:
        check_lora_targets(FakeModel(module_names(PHI3_PROJ)), ["q_proj", "v_proj", "k_proj", "o_proj"])
    except ValueError as e:
        assert "qkv_proj" in str(e)  # the error points at the layer to use instead
    else:
        raise AssertionError("expected ValueError for unmatched Phi-3 targets")


def test_repo_configs_match_their_architectures():
    for path, proj in [
        ("configs/phi4_lora.yaml", PHI3_PROJ),
        ("configs/llama_lora.yaml", LLAMA_PROJ),
        ("configs/mistral_qlora.yaml", LLAMA_PROJ),
    ]:
        targets = config_targets(path)
        check_lora_targets(FakeModel(module_names(proj)), targets)  # raises on a mismatch


def test_suffix_match_does_not_overmatch():
    # "proj" alone must not match "o_proj": PEFT matches whole name segments.
    assert unmatched_targets(module_names(LLAMA_PROJ), ["proj"]) == ["proj"]


def test_run_names_carry_the_seed():
    assert run_name("microsoft/phi-4-mini-instruct", "ner", 43) == "phi-4-mini-instruct_ner_seed43"
    assert adapter_dir("meta-llama/Llama-3.2-3B-Instruct", "classification", 42) == (
        "./adapters/Llama-3.2-3B-Instruct_classification_seed42"
    )


def test_seed_from_adapter_path():
    assert seed_from_adapter_path("./adapters/phi-4-mini-instruct_ner_seed43") == 43
    assert seed_from_adapter_path("./adapters/phi-4-mini-instruct_ner_seed43/") == 43
    assert seed_from_adapter_path("./adapters/phi-4-mini-instruct_ner") is None  # thesis adapter
    assert seed_from_adapter_path(None) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {name}")
