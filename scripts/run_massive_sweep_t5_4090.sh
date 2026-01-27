#!/bin/bash

# Script per avviare lo Sweep Massivo con T5 su RTX 4090
# Posizione: scripts/run_massive_sweep_t5_4090.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "--------------------------------------------------------------------------------"
echo "  DAKGEA: STARTING T5 MIX-UP SWEEP (OPTIMIZED FOR RTX 4090)"
echo "  Project Root: $PROJECT_ROOT"
echo "  Date: $(date)"
echo "--------------------------------------------------------------------------------"

if [ -d ".venv" ]; then
    PYTHON_EXEC="./.venv/bin/python"
else
    PYTHON_EXEC="python"
fi

export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES=0 

LOG_FILE="massive_t5_4090_$(date +%Y%m%d_%H%M%S).log"

echo "Running T5 sweep (Fine-tuning -> Optimization -> Report)..."
echo "Log: $LOG_FILE"
echo ""

$PYTHON_EXEC -u tests/massive_mixup_t5_sweep.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "--------------------------------------------------------------------------------"
echo "  T5 SWEEP COMPLETED."
echo "  Check massive_t5_report.txt for the final results."
echo "--------------------------------------------------------------------------------"
