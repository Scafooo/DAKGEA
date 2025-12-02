#!/usr/bin/env bash
# Helper script called by parallel to run a single experiment
# Usage: _run_single_experiment.sh CONFIG_FILE

CONFIG_FILE="$1"

if [[ -z "$CONFIG_FILE" ]]; then
    echo "ERROR: No config file specified"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# PROJECT_ROOT, GPU_ID, and PYTHONPATH should be exported by parent script
if [[ -z "$PROJECT_ROOT" ]]; then
    echo "ERROR: PROJECT_ROOT not set"
    exit 1
fi

# Run the experiment
CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${PROJECT_ROOT}/experiments/runner/run.py" "${CONFIG_FILE}"
