#!/usr/bin/env bash
# ============================================================
#  SDEA Test Suite
#  forget_labels @ 1%–10% sweep — reduction evaluation only, no augmentation
#  Generates config YAMLs and runs them via run_experiment.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

OUTDIR="${PROJECT_ROOT}/config/experiments/generated/sdea_tests"
mkdir -p "${OUTDIR}"

# ---- Datasets ---------------------------------------------------------------
DATASETS=(
    "openea/BBC_DB"
    "openea/D_W_15K_V1"
    "openea/D_W_15K_V2"
    "openea/ICEW_WIKI"
    "openea/ICEW_YAGO"
)

# ---- Ratios (1% → 10%) ------------------------------------------------------
RATIOS=("0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.10")


SEED="11037"

# ---- Generate YAMLs ---------------------------------------------------------
generated=()

for dataset in "${DATASETS[@]}"; do
    ds_slug="${dataset##*/}"
    for ratio in "${RATIOS[@]}"; do
        ratio_tag="${ratio/./}"   # 0.01 → 001, 0.10 → 010
        exp_name="sdea_${ds_slug}_fl${ratio_tag}"
        outfile="${OUTDIR}/${exp_name}.yaml"

        cat > "${outfile}" <<YAML
experiment:
  name: ${exp_name}
  suite: sdea_ratio_sweep
  dataset:
    name: ${dataset}
    writer: bert_int
  reduction:
    method: forget_labels
    ratio: ${ratio}
    writer: bert_int
    save: false
    eval: true
  augmentation:
    method: stub
    writer: bert_int
    save: false
    eval: false
  model: sdea
  seed: ${SEED}
  clear: true
  overwrite_existing: true
YAML
        generated+=("${outfile}")
    done
done

echo "Generated ${#generated[@]} configs in ${OUTDIR}:"
for f in "${generated[@]}"; do
    echo "  $(basename "$f")"
done
echo ""

# ---- Run via run_experiment.sh ----------------------------------------------
exec "${SCRIPT_DIR}/run_experiment.sh" "${OUTDIR}"
