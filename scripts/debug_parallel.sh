#!/usr/bin/env bash
# Debug script to diagnose parallel execution issues

set -x  # Enable debug output

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

echo "=== Debug Info ==="
echo "Script dir: ${SCRIPT_DIR}"
echo "Project root: ${PROJECT_ROOT}"
echo "Bash version: ${BASH_VERSION}"
echo ""

# Find parallel
PARALLEL_BIN=""
if [ -f "${PROJECT_ROOT}/.local/bin/parallel" ]; then
    PARALLEL_BIN="${PROJECT_ROOT}/.local/bin/parallel"
    echo "Using local parallel: ${PARALLEL_BIN}"
elif command -v parallel &> /dev/null; then
    PARALLEL_BIN=$(which parallel)
    echo "Using system parallel: ${PARALLEL_BIN}"
else
    echo "ERROR: parallel not found"
    exit 1
fi

echo "Parallel version:"
"${PARALLEL_BIN}" --version | head -1
echo ""

# Create a simple test config
TEST_CONFIG="${PROJECT_ROOT}/config/experiments/massive/bert_int_only_red/BBC_DB_01_00.yaml"
if [ ! -f "${TEST_CONFIG}" ]; then
    echo "ERROR: Test config not found: ${TEST_CONFIG}"
    exit 1
fi

echo "Test config: ${TEST_CONFIG}"
echo ""

# Test simple parallel execution
echo "=== Test 1: Simple echo ==="
echo "${TEST_CONFIG}" | "${PARALLEL_BIN}" --will-cite --jobs 1 echo "Config: {}"
echo ""

# Test with python --version (quick)
echo "=== Test 2: Python version ==="
echo "${TEST_CONFIG}" | "${PARALLEL_BIN}" --will-cite --jobs 1 \
    "python3 --version && echo 'Config: {}'"
echo ""

# Test with actual run.py help
echo "=== Test 3: Run.py help ==="
echo "${TEST_CONFIG}" | "${PARALLEL_BIN}" --will-cite --jobs 1 \
    "python3 ${PROJECT_ROOT}/experiments/runner/run.py --help | head -5"
echo ""

# Test with actual config (dry run check)
echo "=== Test 4: Actual command (file existence check) ==="
export PYTHONPATH="${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=0

echo "${TEST_CONFIG}" | "${PARALLEL_BIN}" --will-cite --jobs 1 \
    "test -f {} && echo 'Config exists: {}' || echo 'Config missing: {}'"
echo ""

echo "=== All debug tests completed ==="
