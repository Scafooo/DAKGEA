#!/usr/bin/env bash
set -euo pipefail

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}"
export PYTHONHASHSEED=0

# ── venv ───────────────────────────────────────────────────────────────────
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# ── parameters ─────────────────────────────────────────────────────────────
DATASET="openea/BBC_DB"
RED_RATIO="0.1"
AUG_RATIO="0.2"
SEED="11037"
CSV_OUT="${PROJECT_ROOT}/analysis/bbc_01_02_scores.csv"

NO_AUG_CONFIG="/tmp/score_analysis_no_aug.yaml"
AUG_CONFIG="/tmp/score_analysis_aug.yaml"

# ── generate configs ────────────────────────────────────────────────────────
cat > "${NO_AUG_CONFIG}" << EOF
experiment:
  name: score_analysis_no_aug
  dataset:
    name: ${DATASET}
    writer: bert_int
  reduction:
    method: forget_labels
    ratio: ${RED_RATIO}
    writer: bert_int
    eval: true
    save_dataset: false
    save_model: false
  model: bert_int
  seed: ${SEED}
  clear: true
  overwrite_existing: true
EOF

cat > "${AUG_CONFIG}" << EOF
experiment:
  name: score_analysis_aug
  dataset:
    name: ${DATASET}
    writer: bert_int
  reduction:
    method: forget_labels
    ratio: ${RED_RATIO}
    writer: bert_int
    eval: false
    save_dataset: false
    save_model: false
  augmentation:
    method: plm
    ratio: ${AUG_RATIO}
    writer: bert_int
    eval: true
    save_dataset: false
    save_model: false
  model: bert_int
  seed: ${SEED}
  clear: false
  overwrite_existing: true
EOF

# ── run experiments ─────────────────────────────────────────────────────────
echo "============================================================"
echo "  Running: no-aug  (reduction=${RED_RATIO})"
echo "============================================================"
python "${PROJECT_ROOT}/experiments/runner/run.py" "${NO_AUG_CONFIG}"

echo "============================================================"
echo "  Running: aug  (reduction=${RED_RATIO}, augmentation=${AUG_RATIO})"
echo "============================================================"
python "${PROJECT_ROOT}/experiments/runner/run.py" "${AUG_CONFIG}"

# ── find score files ────────────────────────────────────────────────────────
NO_AUG_SCORES=$(find "${PROJECT_ROOT}/results/score_analysis_no_aug" \
    -name "score_distributions.json" -path "*/bert_int/reduced/*" | head -1)
AUG_SCORES=$(find "${PROJECT_ROOT}/results/score_analysis_aug" \
    -name "score_distributions.json" -path "*/bert_int/plm/*" | head -1)

if [[ -z "${NO_AUG_SCORES}" ]]; then
    echo "ERROR: score_distributions.json not found for no-aug run"
    exit 1
fi
if [[ -z "${AUG_SCORES}" ]]; then
    echo "ERROR: score_distributions.json not found for aug run"
    exit 1
fi

echo ""
echo "no-aug scores : ${NO_AUG_SCORES}"
echo "aug scores    : ${AUG_SCORES}"

# ── analyse ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Score analysis"
echo "============================================================"
python "${PROJECT_ROOT}/scripts/tools/analyze_score_distributions.py" \
    --no-aug "${NO_AUG_SCORES}" \
    --aug    "${AUG_SCORES}"    \
    --topk   50                 \
    --csv    "${CSV_OUT}"
