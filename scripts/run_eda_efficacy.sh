#!/usr/bin/env bash
# ============================================================
#  EDA augmentation — efficacy sweep
#  5 datasets × 10 reduction ratios (0.1 … 1.0) × aug=1.0
#  = 50 experiments
#  Both reduction AND augmentation are evaluated so that the
#  baseline (reduced, no aug) is directly comparable to EDA.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

export PYTHONPATH="${PROJECT_ROOT}"
export PYTHONHASHSEED=0

if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

GEN_DIR="${PROJECT_ROOT}/config/experiments/generated/eda_efficacy"
mkdir -p "${GEN_DIR}"

BRANCH=$(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no-git")
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "CPU only")

echo "============================================================"
echo "  EDA Augmentation — Efficacy Sweep (50 experiments)"
echo "============================================================"
echo "  Branch   : ${BRANCH}"
echo "  Hardware : ${GPU}"
echo "  Started  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# ── Generate all 50 YAML configs ────────────────────────────────
PROJECT_ROOT="${PROJECT_ROOT}" python - <<'PYEOF'
import os, pathlib

root    = pathlib.Path(os.environ["PROJECT_ROOT"])
gen_dir = root / "config/experiments/generated/eda_efficacy"
gen_dir.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "BBC_DB":      "openea/BBC_DB",
    "D_W_15K_V1":  "openea/D_W_15K_V1",
    "D_W_15K_V2":  "openea/D_W_15K_V2",
    "ICEWS_WIKI":  "openea/ICEW_WIKI",
    "ICEWS_YAGO":  "openea/ICEW_YAGO",
}
SEED   = 11037
RATIOS = [round(i * 0.1, 1) for i in range(1, 11)]   # 0.1 … 1.0

generated = 0
for ds_key, ds_name in DATASETS.items():
    for red in RATIOS:
        red_tag = str(int(red * 10)).zfill(2)
        name    = f"eda_efficacy_{ds_key}_{red_tag}"
        yaml    = f"""\
experiment:
  suite: eda_efficacy
  name: {name}
  dataset:
    name: {ds_name}
    writer: bert_int
  reduction:
    method: forget_labels
    ratio: {red}
    writer: bert_int
    eval: false
    save_dataset: false
    save_model: false
  augmentation:
    method: eda
    ratio: 1.0
    alpha_sr: 0.1
    alpha_ri: 0.1
    alpha_rs: 0.1
    alpha_rd: 0.1
    writer:
      type: bert_int
      augmented_only_train: true
    eval: true
    save_dataset: false
    save_model: false
  model: bert_int
  seed: {SEED}
  clear: true
  overwrite_existing: false
"""
        (gen_dir / f"{name}.yaml").write_text(yaml)
        generated += 1

print(f"Generated {generated} configs in {gen_dir}")
PYEOF

echo ""
echo "============================================================"
echo "  Running 50 experiments"
echo "============================================================"
echo ""

TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

for CONFIG in "${GEN_DIR}"/eda_efficacy_*.yaml; do
    TOTAL=$((TOTAL + 1))
    NAME=$(basename "${CONFIG}" .yaml)

    echo "  [${TOTAL}/50] ${NAME}"

    if python -m experiments.runner "${CONFIG}"; then
        PASSED=$((PASSED + 1))
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 2 ]; then
            SKIPPED=$((SKIPPED + 1))
            echo "    → skipped (already exists)"
        else
            FAILED=$((FAILED + 1))
            echo "    ✗ FAILED (exit ${EXIT_CODE}) — continuing"
        fi
    fi
done

echo ""
echo "============================================================"
echo "  Done  : ${TOTAL} total"
echo "  OK    : ${PASSED}"
echo "  Skip  : ${SKIPPED}"
echo "  Failed: ${FAILED}"
echo "  Ended : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

[ "${FAILED}" -eq 0 ]
