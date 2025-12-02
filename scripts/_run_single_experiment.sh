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

if [[ -z "$PROJECT_ROOT" ]]; then
    echo "ERROR: PROJECT_ROOT not set"
    exit 1
fi

if [[ -z "$GPU_ID" ]]; then
    echo "ERROR: GPU_ID not set"
    exit 1
fi

if [[ -z "$PYTHONPATH" ]]; then
    echo "WARNING: PYTHONPATH not set"
fi

echo "--------------------------------------------------"
echo "Running experiment:"
echo "CONFIG_FILE: $CONFIG_FILE"
echo "GPU_ID: $GPU_ID"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "--------------------------------------------------"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${PROJECT_ROOT}/experiments/runner/run.py" "${CONFIG_FILE}"
EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "ERROR: Experiment failed for config $CONFIG_FILE with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "Experiment completed successfully for config $CONFIG_FILE"
