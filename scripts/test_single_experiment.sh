#!/usr/bin/env bash
# Simple test script to run a single experiment with parallel

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Find parallel
if [ -f "${PROJECT_ROOT}/.local/bin/parallel" ]; then
    PARALLEL_BIN="${PROJECT_ROOT}/.local/bin/parallel"
    echo "Using local parallel: ${PARALLEL_BIN}"
elif command -v parallel &> /dev/null; then
    PARALLEL_BIN="parallel"
    echo "Using system parallel"
else
    echo "ERROR: parallel not found"
    exit 1
fi

# Setup
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}"
export GPU_ID=0

# Test config
TEST_CONFIG="${PROJECT_ROOT}/config/experiments/massive/bert_int_only_red/BBC_DB_01_00.yaml"

if [ ! -f "${TEST_CONFIG}" ]; then
    echo "ERROR: Test config not found: ${TEST_CONFIG}"
    exit 1
fi

echo "Test config: ${TEST_CONFIG}"
echo "Project root: ${PROJECT_ROOT}"
echo "GPU ID: ${GPU_ID}"
echo ""

# Define and export function
run_experiment() {
    local config_file="$1"
    echo "=== Running experiment ==="
    echo "Config: ${config_file}"
    echo "Command: CUDA_VISIBLE_DEVICES=${GPU_ID} python ${PROJECT_ROOT}/experiments/runner/run.py ${config_file}"
    echo ""
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${PROJECT_ROOT}/experiments/runner/run.py" "${config_file}"
}
export -f run_experiment

# Run with parallel
echo "Starting parallel execution..."
echo "${TEST_CONFIG}" | "${PARALLEL_BIN}" --will-cite --jobs 1 --verbose run_experiment {}

echo ""
echo "Done!"
