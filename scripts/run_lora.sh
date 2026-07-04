#!/bin/bash
# Run LoRA fine-tuning for Phi-4-mini and Llama-3.2 across all 5 tasks
# Usage: bash scripts/run_lora.sh phi4   OR   bash scripts/run_lora.sh llama

MODEL=${1:-phi4}
FINANCIAL_PATH="data/financial_phrasebank.csv"  # run scripts/prepare_financial_data.py first

if [ "$MODEL" = "phi4" ]; then
    CONFIG="configs/phi4_lora.yaml"
elif [ "$MODEL" = "llama" ]; then
    CONFIG="configs/llama_lora.yaml"
else
    echo "Usage: bash scripts/run_lora.sh [phi4|llama]"
    exit 1
fi

TASKS=("classification" "ner" "summarization" "financial_sentiment" "code_generation")

for TASK in "${TASKS[@]}"; do
    echo "=========================================="
    echo "Training: $MODEL | Task: $TASK"
    echo "=========================================="
    if [ "$TASK" = "financial_sentiment" ]; then
        python src/training/train_lora.py \
            --config $CONFIG \
            --task $TASK \
            --financial_path $FINANCIAL_PATH
    else
        python src/training/train_lora.py \
            --config $CONFIG \
            --task $TASK
    fi
done

echo "✅ All LoRA training complete for $MODEL"