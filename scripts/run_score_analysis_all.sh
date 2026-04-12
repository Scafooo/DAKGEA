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

# ── best configurations (flan_t5, max delta H@1) ──────────────────────────
# Dataset          openea_name         red  aug  seed
declare -a DATASETS=(
    "BBC_DB      openea/BBC_DB      0.1  0.9  11037"
    "D_W_15K_V1  openea/D_W_15K_V1  0.1  0.7  11039"
    "D_W_15K_V2  openea/D_W_15K_V2  0.7  0.6  11041"
    "ICEW_WIKI   openea/ICEW_WIKI   0.2  0.8  11041"
    "ICEW_YAGO   openea/ICEW_YAGO   0.1  0.4  11041"
)

for entry in "${DATASETS[@]}"; do
    read -r DS_NAME DATASET RED_RATIO AUG_RATIO SEED <<< "${entry}"

    echo "============================================================"
    echo "  Dataset: ${DS_NAME}  (red=${RED_RATIO}, aug=${AUG_RATIO}, seed=${SEED})"
    echo "============================================================"

    NO_AUG_CONFIG="/tmp/score_analysis_no_aug_${DS_NAME}.yaml"
    AUG_CONFIG="/tmp/score_analysis_aug_${DS_NAME}.yaml"
    CSV_OUT="${PROJECT_ROOT}/analysis/${DS_NAME}_scores.csv"

    # ── generate configs ────────────────────────────────────────────────
    cat > "${NO_AUG_CONFIG}" << EOF
experiment:
  name: score_analysis_no_aug_${DS_NAME}
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
  name: score_analysis_aug_${DS_NAME}
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
    method: plm_mixup
    backbone: "flan-t5-xl"
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

    # ── run experiments ──────────────────────────────────────────────────
    echo "  Running: no-aug  (reduction=${RED_RATIO})"
    python "${PROJECT_ROOT}/experiments/runner/run.py" "${NO_AUG_CONFIG}"

    echo "  Running: aug  (reduction=${RED_RATIO}, augmentation=${AUG_RATIO})"
    python "${PROJECT_ROOT}/experiments/runner/run.py" "${AUG_CONFIG}"

    # ── find score files ─────────────────────────────────────────────────
    NO_AUG_SCORES=$(find "${PROJECT_ROOT}/results/score_analysis_no_aug_${DS_NAME}" \
        -name "score_distributions.json" -path "*/bert_int/reduced/*" | head -1)
    AUG_SCORES=$(find "${PROJECT_ROOT}/results/score_analysis_aug_${DS_NAME}" \
        -name "score_distributions.json" -path "*/bert_int/plm/*" | head -1)

    if [[ -z "${NO_AUG_SCORES}" ]]; then
        echo "ERROR: score_distributions.json not found for no-aug run (${DS_NAME})"
        exit 1
    fi
    if [[ -z "${AUG_SCORES}" ]]; then
        echo "ERROR: score_distributions.json not found for aug run (${DS_NAME})"
        exit 1
    fi

    echo "  no-aug scores : ${NO_AUG_SCORES}"
    echo "  aug scores    : ${AUG_SCORES}"

    # ── analyse ──────────────────────────────────────────────────────────
    python "${PROJECT_ROOT}/scripts/tools/analyze_score_distributions.py" \
        --no-aug "${NO_AUG_SCORES}" \
        --aug    "${AUG_SCORES}"    \
        --topk   1000               \
        --csv    "${CSV_OUT}"

    echo "  CSV saved to: ${CSV_OUT}"
    echo ""
done

echo "All done."
