#!/usr/bin/env bash
# ============================================================
#  PLM Mixup Flan-T5 Test Suite
#  forget_labels @ 10%/20%/30% + plm_mixup (flan-t5-xl) augmentation
#  Generates config YAMLs and runs them via run_experiment.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

OUTDIR="${PROJECT_ROOT}/config/experiments/generated/sdea_plm_mixup_flan_t5_test"
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
REDUCTION_RATIOS=("0.1" "0.2" "0.3")

# ---- Augmentation ratios (0.1 → 1.0) ----------------------------------------
AUG_RATIOS=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")

SEED="11037"
SUITE="sdea_plm_mixup_flan_t5_test"

# ---- Generate YAMLs ---------------------------------------------------------
generated=()

for dataset in "${DATASETS[@]}"; do
    ds_slug="${dataset##*/}"
    for r in "${REDUCTION_RATIOS[@]}"; do
        r_tag="${r/./}"   # 0.1 → 01, 0.2 → 02, 0.3 → 03
        for a in "${AUG_RATIOS[@]}"; do
            a_tag="${a/./}"   # 0.1 → 01, 1.0 → 10
            exp_name="${ds_slug}_r${r_tag}_a${a_tag}"
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
    ratio: ${r}
    writer: bert_int
    eval: true
    save_dataset: false
    save_model: false
  augmentation:
    method: plm_mixup
    ratio: ${a}
    backbone: flan-t5-xl
    pretrained_model_dir: models/pretrained_plm/${ds_slug}
    writer:
      type: bert_int
      augmented_only_train: true
    eval: true
    save_dataset: false
    save_model: false
  model: sdea
  seed: ${SEED}
  clear: true
  overwrite_existing: true
YAML
            generated+=("${outfile}")
        done
    done
done

echo "Generated ${#generated[@]} configs in ${OUTDIR}:"
for f in "${generated[@]}"; do
    echo "  $(basename "$f")"
done
echo ""

# ---- Run via run_experiment.sh ----------------------------------------------
exec "${SCRIPT_DIR}/run_experiment.sh" "${OUTDIR}"
