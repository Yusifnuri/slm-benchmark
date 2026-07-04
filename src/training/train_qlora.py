"""
QLoRA fine-tuning script for Mistral-7B-v0.3.
Expose: "Mistral-7B-v0.3 with QLoRA (Weeks 7-8)"

QLoRA (Dettmers et al. 2023) loads the model in 4-bit NF4 quantization,
reducing GPU RAM from ~80GB to ~14GB, enabling fine-tuning on A6000.

Usage:
    python src/training/train_qlora.py \
        --config configs/mistral_qlora.yaml \
        --task classification
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
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

from data.dataset_loader import SLMDataset
from utils.mlflow_logger import BenchmarkLogger


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_qlora_model(model_name: str, qlora_cfg: dict):
    """
    Load Mistral-7B in 4-bit NF4 quantization and attach LoRA adapters.

    Steps:
    1. BitsAndBytesConfig → tells HuggingFace to load weights as INT4
    2. prepare_model_for_kbit_training → enables gradient computation on INT4
    3. LoraConfig + get_peft_model → attach trainable adapter layers
    """
    print(f"\n📦 Loading model in 4-bit QLoRA: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # QLoRA quantization config (Dettmers et al. 2023)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=qlora_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qlora_cfg["bnb_4bit_use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Required for QLoRA: prepares quantized layers for gradient flow
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=qlora_cfg["r"],
        lora_alpha=qlora_cfg["alpha"],
        lora_dropout=qlora_cfg["dropout"],
        target_modules=qlora_cfg["target_modules"],
        bias=qlora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Gradient checkpointing trades compute for memory. Combined with 4-bit
    # quantization this is what keeps Mistral-7B fine-tuning inside a single
    # consumer/workstation GPU's VRAM budget. enable_input_require_grads() is
    # required alongside it for PEFT models, otherwise no gradients flow to
    # the LoRA adapters when the base model is frozen.
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

    print(f"\n🚀 QLoRA Training | Model: {model_short} | Task: {task}")

    model, tokenizer = build_qlora_model(cfg["model"]["name"], cfg["qlora"])

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
        method="QLoRA",
        params={
            "lora_r": cfg["qlora"]["r"],
            "lora_alpha": cfg["qlora"]["alpha"],
            "lr": cfg["training"]["learning_rate"],
            "epochs": cfg["training"]["num_epochs"],
            "batch_size": cfg["training"]["batch_size"],
            "quantization": "4bit_nf4",
            "double_quant": cfg["qlora"]["bnb_4bit_use_double_quant"],
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
    print(f" QLoRA Adapter saved → {adapter_path}")


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