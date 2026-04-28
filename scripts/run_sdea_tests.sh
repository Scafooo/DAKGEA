#!/usr/bin/env bash
# ============================================================
#  SDEA Test Suite
#  forget_labels @ 10% — reduction evaluation only, no augmentation
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

RATIO="0.01"
SEED="11037"

# ---- Generate YAMLs ---------------------------------------------------------
generated=()

for dataset in "${DATASETS[@]}"; do
    ds_slug="${dataset##*/}"
    exp_name="sdea_${ds_slug}_fl01"
    outfile="${OUTDIR}/${exp_name}.yaml"

    cat > "${outfile}" <<YAML
experiment:
  name: ${exp_name}
  dataset:
    name: ${dataset}
    writer: bert_int
  reduction:
    method: forget_labels
    ratio: ${RATIO}
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

echo "Generated ${#generated[@]} configs in ${OUTDIR}:"
for f in "${generated[@]}"; do
    echo "  $(basename "$f")"
done
echo ""

# ---- Run via run_experiment.sh ----------------------------------------------
exec "${SCRIPT_DIR}/run_experiment.sh" "${OUTDIR}"
