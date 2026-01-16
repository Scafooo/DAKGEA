#!/bin/bash
# Ablation Study: BART without fine-tuning
# Tests the impact of fine-tuning on augmentation quality
# Reduction: 0.1, Augmentation: 1.0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Ablation Study: No Fine-tuning"
echo "=========================================="

CONFIGS=(
    "config/experiments/massive/ablation_no_finetune/BBC_DB_01_10.yaml"
    "config/experiments/massive/ablation_no_finetune/D_W_15K_V1_01_10.yaml"
    "config/experiments/massive/ablation_no_finetune/D_W_15K_V2_01_10.yaml"
    "config/experiments/massive/ablation_no_finetune/ICEW_WIKI_01_10.yaml"
    "config/experiments/massive/ablation_no_finetune/ICEW_YAGO_01_10.yaml"
)

for config in "${CONFIGS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running: $config"
    echo "=========================================="
    python -m src.main --config "$config"
done

echo ""
echo "=========================================="
echo "Ablation Study Complete!"
echo "=========================================="
