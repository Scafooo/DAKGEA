#!/usr/bin/env bash
# ============================================================
#  Run EDA augmentation on the best (red, aug) configs
#  from analysis/best_configs_flan_t5.json
# ============================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

export PYTHONPATH="${PROJECT_ROOT}"
export PYTHONHASHSEED=0

if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

BEST_JSON="${PROJECT_ROOT}/analysis/best_configs_flan_t5.json"
GEN_DIR="${PROJECT_ROOT}/config/experiments/generated"
RESULTS_DIR="${PROJECT_ROOT}/results/eda_best"

mkdir -p "${GEN_DIR}" "${RESULTS_DIR}"

BRANCH=$(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no-git")
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "CPU only")

echo "============================================================"
echo "  EDA Augmentation — Best Configs Run"
echo "============================================================"
echo "  Best configs : ${BEST_JSON}"
echo "  Branch       : ${BRANCH}"
echo "  Hardware     : ${GPU}"
echo "  Started      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

# Generate one YAML config per entry in the JSON
python - <<'PYEOF'
import json, pathlib, sys

root      = pathlib.Path(sys.argv[0]).resolve().parents[0] if False else pathlib.Path(".")
best_json = pathlib.Path("analysis/best_configs_flan_t5.json")
gen_dir   = pathlib.Path("config/experiments/generated")

with open(best_json) as f:
    best = json.load(f)

for ds_key, cfg in best.items():
    name = f"eda_best_{cfg['folder']}"
    yaml_content = f"""\
experiment:
  name: {name}
  dataset:
    name: {cfg['dataset']}
    writer: bert_int
  reduction:
    method: random_entities
    ratio: {cfg['red']}
    writer: bert_int
    save: false
    eval: false
  augmentation:
    method: eda
    ratio: {cfg['aug']}
    alpha_sr: 0.1
    alpha_ri: 0.1
    alpha_rs: 0.1
    alpha_rd: 0.1
    writer: bert_int
    save: false
    eval: true
  model: bert_int
  seed: {cfg['seed']}
  clear: false
  overwrite_existing: true
"""
    out = gen_dir / f"{name}.yaml"
    out.write_text(yaml_content)
    print(out)
PYEOF

echo ""
echo "============================================================"
echo "  Running experiments"
echo "============================================================"
echo ""

TOTAL=0
PASSED=0
FAILED=0

for CONFIG in "${GEN_DIR}"/eda_best_*.yaml; do
    TOTAL=$((TOTAL + 1))
    NAME=$(basename "${CONFIG}" .yaml)
    echo "------------------------------------------------------------"
    echo "  [${TOTAL}] ${NAME}"
    echo "  Config: ${CONFIG}"
    echo "------------------------------------------------------------"

    if python -m experiments.runner "${CONFIG}" --overwrite-existing; then
        PASSED=$((PASSED + 1))
        echo "  ✓ ${NAME} — OK"
    else
        FAILED=$((FAILED + 1))
        echo "  ✗ ${NAME} — FAILED (continuing)"
    fi
    echo ""
done

echo "============================================================"
echo "  Done: ${PASSED}/${TOTAL} passed, ${FAILED} failed"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

[ "${FAILED}" -eq 0 ]
