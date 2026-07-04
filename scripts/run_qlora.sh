#!/bin/bash
# Run QLoRA fine-tuning for Mistral-7B across all 5 tasks
# Usage: bash scripts/run_qlora.sh

CONFIG="configs/mistral_qlora.yaml"
FINANCIAL_PATH="data/financial_phrasebank.csv"  # run scripts/prepare_financial_data.py first

TASKS=("classification" "ner" "summarization" "financial_sentiment" "code_generation")

for TASK in "${TASKS[@]}"; do
    echo "=========================================="
    echo "QLoRA Training: Mistral-7B | Task: $TASK"
    echo "=========================================="
    if [ "$TASK" = "financial_sentiment" ]; then
        python src/training/train_qlora.py \
            --config $CONFIG \
            --task $TASK \
            --financial_path $FINANCIAL_PATH
    else
        python src/training/train_qlora.py \
            --config $CONFIG \
            --task $TASK
    fi
done

echo "✅ All QLoRA training complete for Mistral-7B"