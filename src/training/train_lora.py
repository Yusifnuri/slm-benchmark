"""
LoRA fine-tuning script for Phi-4-mini and Llama-3.2-3B.
Expose: "Phi-4-mini with LoRA (Weeks 5-6), Llama-3.2-3B with LoRA (Weeks 9-10)"

Usage:
    # Phi-4-mini
    python src/training/train_lora.py \
        --config configs/phi4_lora.yaml \
        --task classification

    # Llama
    python src/training/train_lora.py \
        --config configs/llama_lora.yaml \
        --task ner \
        --financial_path /path/to/phrasebank.csv
"""

import sys
import os
import yaml
import torch
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

from data.dataset_loader import SLMDataset
from utils.mlflow_logger import BenchmarkLogger


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_lora_model(model_name: str, lora_cfg: dict):
    """Load base model in float16 and attach LoRA adapters."""
    print(f"\n📦 Loading model: {model_name}")

    # trust_remote_code deliberately omitted: transformers >= 4.51 has native
    # Phi3/Mistral/Llama support, and enabling it for phi-4-mini-instruct pulls
    # Microsoft's stale custom modeling_phi3.py, which imports `LossKwargs`
    # from transformers.utils — removed in current transformers and fails
    # with ImportError. Native classes work for all 3 models we fine-tune.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Gradient checkpointing trades compute for memory — needed headroom to fit
    # larger batch sizes on a single GPU. enable_input_require_grads() is required
    # alongside it for PEFT models, otherwise no gradients flow to the LoRA adapters.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    return model, tokenizer


def train(config_path: str, task: str, financial_path: str = None):
    cfg = load_config(config_path)
    model_short = cfg["model"]["name"].split("/")[-1]
    logger = BenchmarkLogger(
        cfg["mlflow"]["experiment_name"],
        cfg["mlflow"]["tracking_uri"],
    )

    print(f"\n🚀 LoRA Training | Model: {model_short} | Task: {task}")

    model, tokenizer = build_lora_model(cfg["model"]["name"], cfg["lora"])

    train_ds = SLMDataset(
        task=task, split="train", tokenizer=tokenizer,
        max_length=cfg["model"]["max_length"],
        max_samples=cfg["training"]["max_samples_train"],
        financial_phrasebank_path=financial_path,
    )
    eval_ds = SLMDataset(
        # "validation" here is for Trainer's periodic loss checks /
        # load_best_model_at_end checkpoint selection only. Final reported
        # benchmark numbers come from the untouched "test" split in evaluate.py.
        task=task,
        split="validation",
        tokenizer=tokenizer,
        max_length=cfg["model"]["max_length"],
        max_samples=cfg["training"]["max_samples_eval"],
        financial_phrasebank_path=financial_path,
    )

    output_dir = f"./outputs/{model_short}_{task}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["training"]["num_epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        lr_scheduler_type=cfg["training"]["lr_scheduler"],
        fp16=cfg["training"]["fp16"],
        save_steps=cfg["training"]["save_steps"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        report_to="mlflow",
        run_name=f"{model_short}_{task}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
    )

    start = time.time()
    result = trainer.train()
    training_hours = (time.time() - start) / 3600

    logger.log_training_run(
        model_name=cfg["model"]["name"],
        task=task,
        method="LoRA",
        params={
            "lora_r": cfg["lora"]["r"],
            "lora_alpha": cfg["lora"]["alpha"],
            "lr": cfg["training"]["learning_rate"],
            "epochs": cfg["training"]["num_epochs"],
            "batch_size": cfg["training"]["batch_size"],
        },
        metrics={
            "train_loss": result.training_loss,
            "training_hours": training_hours,
        },
        artifacts_path=output_dir,
    )

    adapter_path = f"./adapters/{model_short}_{task}"
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"✅ Adapter saved → {adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True, choices=[
        "classification", "ner", "summarization",
        "financial_sentiment", "code_generation",
    ])
    parser.add_argument("--financial_path", default=None)
    args = parser.parse_args()
    train(args.config, args.task, args.financial_path)