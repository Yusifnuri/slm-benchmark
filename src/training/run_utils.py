"""
Helpers shared by train_lora.py, train_qlora.py and scripts/run_all_experiments.py.

Kept free of torch/transformers imports so the checks below can be unit-tested
on a machine without a GPU stack (tests/test_run_utils.py).
"""

import os
import re
from typing import Iterable, List, Optional

DEFAULT_SEED = 42  # TrainingArguments' default seed, i.e. what the thesis runs used


def run_name(model_name: str, task: str, seed: int) -> str:
    """One name per (model, task, seed) run, used for outputs/, adapters/ and MLflow.

    The seed is always part of the name, so multi-seed runs never overwrite each
    other and never collide with the unsuffixed adapters released with the thesis.
    """
    return f"{model_name.split('/')[-1]}_{task}_seed{seed}"


def adapter_dir(model_name: str, task: str, seed: int) -> str:
    return f"./adapters/{run_name(model_name, task, seed)}"


def seed_from_adapter_path(adapter_path: Optional[str]) -> Optional[int]:
    """Seed encoded in an adapter directory name, or None (thesis adapters, base models, APIs).

    Lets evaluation keep each seed's predictions and MLflow run apart without a new
    argument, while thesis-era paths keep their original output names.
    """
    if not adapter_path:
        return None
    match = re.search(r"_seed(\d+)$", os.path.basename(os.path.normpath(adapter_path)))
    return int(match.group(1)) if match else None


def unmatched_targets(module_names: Iterable[str], target_modules: List[str]) -> List[str]:
    """Return the configured LoRA targets that match no module in the model.

    Uses PEFT's rule for a list of targets: a module matches a target when its
    qualified name equals the target or ends with "." + target.
    """
    names = list(module_names)
    return [
        target for target in target_modules
        if not any(name == target or name.endswith("." + target) for name in names)
    ]


def check_lora_targets(model, target_modules: List[str]) -> None:
    """Fail before training if any configured LoRA target is missing from the model.

    PEFT only errors when *none* of the targets match. If some match, the rest are
    skipped silently, yet adapter_config.json still lists every configured name.
    That is how the thesis Phi-4-mini adapters ended up on o_proj alone: Phi-3
    fuses q, k and v into a single qkv_proj, so q_proj/k_proj/v_proj matched nothing.
    """
    missing = unmatched_targets((name for name, _ in model.named_modules()), target_modules)
    if missing:
        leaf_names = sorted({name.rsplit(".", 1)[-1] for name, _ in model.named_modules() if name})
        proj_names = [n for n in leaf_names if n.endswith("_proj")]
        raise ValueError(
            f"LoRA target_modules {missing} match no layer in this model. "
            f"Projection layers it does have: {proj_names}. Fix target_modules in the config."
        )
