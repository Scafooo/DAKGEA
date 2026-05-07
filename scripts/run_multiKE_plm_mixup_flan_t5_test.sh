#!/usr/bin/env bash
# ============================================================
#  MultiKE PLM Mixup Flan-T5 Test Suite
#  forget_labels @ 10%/20%/30% + plm_mixup (flan-t5-xl) augmentation
#  Generates config YAMLs and runs them via run_experiment.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

OUTDIR="${PROJECT_ROOT}/config/experiments/generated/multiKE_plm_mixup_flan_t5_test"
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
SUITE="multiKE_plm_mixup_flan_t5_test"

# ---- Word2Vec embeddings -----------------------------------------------------
W2V_DIR="${PROJECT_ROOT}/data/embeddings"
W2V_VEC="${W2V_DIR}/wiki-news-300d-1M.vec"
W2V_ZIP="${W2V_DIR}/wiki-news-300d-1M.vec.zip"
W2V_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"

mkdir -p "${W2V_DIR}"

if [ ! -f "${W2V_VEC}" ]; then
    echo "word2vec embeddings not found. Downloading (~650 MB)..."
    if [ ! -f "${W2V_ZIP}" ]; then
        wget -q --show-progress -O "${W2V_ZIP}" "${W2V_URL}"
    fi
    echo "Extracting ${W2V_ZIP}..."
    unzip -q "${W2V_ZIP}" -d "${W2V_DIR}"
    echo "word2vec embeddings ready at ${W2V_VEC}"
else
    echo "word2vec embeddings already present at ${W2V_VEC}"
fi

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
  model: multiKE
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
