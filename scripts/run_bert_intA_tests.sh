#!/usr/bin/env bash
# ============================================================
#  BERT-INT-A Test Suite
#  forget_labels @ 10% / 20% / 30% — reduction evaluation only, no augmentation
#  Generates config YAMLs and runs them via run_experiment.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

OUTDIR="${PROJECT_ROOT}/config/experiments/generated/bert_intA_tests"
mkdir -p "${OUTDIR}"

# ---- Datasets ---------------------------------------------------------------
DATASETS=(
    "openea/BBC_DB"
    "openea/D_W_15K_V1"
    "openea/D_W_15K_V2"
    "openea/ICEW_WIKI"
    "openea/ICEW_YAGO"
)

# ---- Reduction ratios -------------------------------------------------------
RATIOS=("0.1" "0.2" "0.3")

SEED="11037"
SUITE="bert_intA_tests"

# ---- Generate YAMLs ---------------------------------------------------------
generated=()

for dataset in "${DATASETS[@]}"; do
    ds_slug="${dataset##*/}"
    for ratio in "${RATIOS[@]}"; do
        ratio_tag="${ratio/./}"   # 0.1 → 01, 0.2 → 02, 0.3 → 03
        exp_name="${ds_slug}_r${ratio_tag}"
        outfile="${OUTDIR}/${exp_name}.yaml"

        cat > "${outfile}" <<YAML
experiment:
  suite: ${SUITE}
  name: ${exp_name}
  dataset:
    name: ${dataset}
    writer: bert_int
  reduction:
    method: forget_labels
    ratio: ${ratio}
    writer: bert_int
    eval: true
    save_dataset: false
    save_model: false
  augmentation:
    method: stub
    writer: bert_int
    eval: false
    save_dataset: false
    save_model: false
  model: bert_intA
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
