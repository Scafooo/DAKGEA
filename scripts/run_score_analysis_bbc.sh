#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/../.venv/bin/activate"

python scripts/run_score_analysis.py \
    --dataset   openea/BBC_DB \
    --red-ratio 0.1 \
    --aug-ratio 0.2 \
    --seed      11037 \
    --topk      10 \
    --csv       analysis/bbc_01_02_scores.csv
