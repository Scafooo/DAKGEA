#!/usr/bin/env bash
# ============================================================
#  run_baseline_reduction.sh
#  Baseline experiments: reduction only (no augmentation)
#  5 seeds × 10 ratios × 5 datasets
#  Saves per-run results + mean in results/baseline_reduction/
# ============================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}"
export PYTHONHASHSEED=0

# ---------- Activate virtual environment ----------
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# ---------- Parameters ----------
DATASETS=("BBC_DB" "D_W_15K_V1" "D_W_15K_V2" "ICEW_WIKI" "ICEW_YAGO")
RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
SEEDS=(11037 42 123 456 789)

CONFIG_DIR="${PROJECT_ROOT}/config/experiments/generated/baseline_reduction"
RESULTS_DIR="${PROJECT_ROOT}/results/baseline_reduction"

mkdir -p "$CONFIG_DIR"
mkdir -p "$RESULTS_DIR"

# ---------- Banner ----------
echo "=============================================="
echo "  Baseline Reduction Experiments"
echo "  Datasets : ${DATASETS[*]}"
echo "  Ratios   : ${RATIOS[*]}"
echo "  Seeds    : ${SEEDS[*]}"
echo "  Total    : $((${#DATASETS[@]} * ${#RATIOS[@]} * ${#SEEDS[@]})) runs"
echo "  Started  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

FAILED=()

for DATASET in "${DATASETS[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        for SEED in "${SEEDS[@]}"; do

            EXP_NAME="baseline_red_${DATASET}_r${RATIO}_s${SEED}"
            CONFIG_FILE="${CONFIG_DIR}/${EXP_NAME}.yaml"

            # Generate config (no augmentation section = fully skipped by runner)
            cat > "$CONFIG_FILE" <<EOF
experiment:
  name: ${EXP_NAME}
  suite: baseline_reduction
  dataset:
    name: openea/${DATASET}
    writer: bert_int
  reduction:
    method: random_entities
    ratio: ${RATIO}
    writer: bert_int
    eval: true
    save_dataset: false
    save_model: false
  model: bert_int
  seed: ${SEED}
  clear: true
  overwrite_existing: true
EOF

            echo ""
            echo "[$(date +%H:%M:%S)] ${EXP_NAME}"
            python "${PROJECT_ROOT}/experiments/runner/run.py" "$CONFIG_FILE" && \
                echo "  ✓ done" || {
                echo "  ✗ FAILED: ${EXP_NAME}"
                FAILED+=("$EXP_NAME")
            }

        done
    done
done

# ---------- Summary ----------
echo ""
echo "=============================================="
echo "  Computing means..."
echo "=============================================="

python3 - <<PYEOF
import json, os
from pathlib import Path
from collections import defaultdict

root    = Path("${PROJECT_ROOT}")
res_dir = root / "results"
out_dir = root / "results" / "baseline_reduction"
out_dir.mkdir(parents=True, exist_ok=True)

datasets = ["BBC_DB", "D_W_15K_V1", "D_W_15K_V2", "ICEW_WIKI", "ICEW_YAGO"]
ratios   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
seeds    = [11037, 42, 123, 456, 789]
metrics  = ["hits@1", "hits@10", "mrr"]

summary = {}

for dataset in datasets:
    summary[dataset] = {}
    for ratio in ratios:
        all_runs = []
        for seed in seeds:
            exp_name    = f"baseline_red_{dataset}_r{ratio}_s{seed}"
            result_path = res_dir / "baseline_reduction" / exp_name / "reduction" / "results.json"
            if not result_path.exists():
                print(f"  MISSING: {result_path}")
                continue
            with open(result_path) as f:
                data = json.load(f)
            bert = data.get("bert_int", data)
            run  = {m: bert[m] for m in metrics if m in bert}
            if run:
                all_runs.append({"seed": seed, **run})

        if all_runs:
            means = {
                m: round(sum(r[m] for r in all_runs if m in r) / len(all_runs), 4)
                for m in metrics
            }
            summary[dataset][str(ratio)] = {
                "n":    len(all_runs),
                "mean": means,
                "runs": all_runs,
            }
            print(f"  {dataset:<15} r={ratio}  {means}  (n={len(all_runs)})")
        else:
            print(f"  {dataset:<15} r={ratio}  no results")

out_file = out_dir / "summary.json"
with open(out_file, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {out_file}")
PYEOF

echo ""
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "✓ All runs completed."
else
    echo "✗ ${#FAILED[@]} run(s) failed:"
    for f in "${FAILED[@]}"; do echo "    - $f"; done
fi
echo "Results: ${RESULTS_DIR}/summary.json"
echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
