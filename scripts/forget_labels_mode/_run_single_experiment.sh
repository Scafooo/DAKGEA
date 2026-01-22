#!/usr/bin/env bash
# Helper script called by parallel to run a single experiment in Forget Labels mode
# Usage: _run_single_experiment.sh CONFIG_FILE

set -euo pipefail

CONFIG_FILE="$1"

if [[ -z "$CONFIG_FILE" ]]; then
    echo "ERROR: No config file specified"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# PROJECT_ROOT should be exported by parent script
if [[ -z "${PROJECT_ROOT:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
    export PROJECT_ROOT
fi

# Activate virtual environment if available
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# Run the experiment using the CUSTOM runner
# Use GPU_ID if set, otherwise default to 0
CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" python "${PROJECT_ROOT}/scripts/forget_labels_mode/run.py" "${CONFIG_FILE}"
