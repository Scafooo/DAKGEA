#!/usr/bin/env bash
# ============================================================
#  Quick smoke-test for the EDA augmentation method
#  Dataset : BBC_DB
#  Reduction ratio : 0.1
#  Augmentation ratio : 0.1
# ============================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

export PYTHONPATH="${PROJECT_ROOT}"
export PYTHONHASHSEED=0

# Activate virtual environment
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

CONFIG="${PROJECT_ROOT}/config/experiments/test_eda_quick.yaml"

BRANCH=$(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no-git")
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "CPU only")

echo "============================================================"
echo "  EDA Augmentation — Quick Smoke Test"
echo "============================================================"
echo "  Config  : ${CONFIG}"
echo "  Branch  : ${BRANCH}"
echo "  Hardware: ${GPU}"
echo "  Started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

python -m experiments.runner "${CONFIG}" --overwrite-existing

EXIT=$?
echo ""
echo "============================================================"
if [ $EXIT -eq 0 ]; then
    echo "  ✓ Completed successfully"
else
    echo "  ✗ Failed with exit code ${EXIT}"
fi
echo "============================================================"
exit $EXIT
