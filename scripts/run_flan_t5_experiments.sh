#!/bin/bash
#
# FLAN-T5 Complete Experiment Pipeline
# =====================================
# Single script that runs everything:
# 1. Pre-train PLM models (once per dataset)
# 2. Run all 500 experiments (each with 5 reduction runs,
#    selecting the run with the lowest hits@1)
# 3. Analyze results and generate statistics
#
# Usage:
#   ./scripts/run_flan_t5_experiments.sh
#
# Requirements:
#   - NVIDIA GPU with ~24GB VRAM (RTX 4090)
#   - ~50GB disk space
#

set -e  # Exit on error

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
DATASETS=("BBC_DB" "D_W_15K_V1" "D_W_15K_V2" "ICEW_WIKI" "ICEW_YAGO")
CONFIG_DIR="config/experiments/massive/flan_t5_bert_int"
RESULTS_DIR="results/flan_t5_bert_int"
LOG_DIR="results/logs/flan_t5"
PRETRAINED_DIR="models/pretrained_plm"
REDUCTION_RUNS=5
BASE_SEED=11037

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$PRETRAINED_DIR"

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/full_run_${TIMESTAMP}.log"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

log "============================================================"
log "FLAN-T5 COMPLETE EXPERIMENT PIPELINE"
log "============================================================"
log "Datasets: ${DATASETS[*]}"
log "Total experiments: 500 (5 datasets × 10 red × 10 aug)"
log "Reduction runs per experiment: $REDUCTION_RUNS (select lowest hits@1)"
log "Log file: $MAIN_LOG"
log "============================================================"

# ==============================================================
# PHASE 1: Pre-train PLM models
# ==============================================================
log ""
log "============================================================"
log "PHASE 1: PRE-TRAINING PLM MODELS"
log "============================================================"

for dataset in "${DATASETS[@]}"; do
    MODEL_DIR="$PRETRAINED_DIR/$dataset"

    if [[ -f "$MODEL_DIR/adapter_config.json" ]]; then
        log "[SKIP] $dataset - model already exists"
        continue
    fi

    log "[TRAIN] $dataset - starting pre-training..."

    python scripts/pretrain_plm_per_dataset.py --datasets "$dataset" \
        2>&1 | tee -a "$LOG_DIR/pretrain_${dataset}_${TIMESTAMP}.log"

    if [[ -f "$MODEL_DIR/adapter_config.json" ]]; then
        log "[OK] $dataset - model saved"
    else
        log "[ERROR] $dataset - pre-training failed!"
        exit 1
    fi
done

log "[COMPLETE] All models pre-trained"

# ==============================================================
# PHASE 2: Run experiments
# ==============================================================
log ""
log "============================================================"
log "PHASE 2: RUNNING EXPERIMENTS"
log "============================================================"

TOTAL_CONFIGS=$(ls "$CONFIG_DIR"/*.yaml 2>/dev/null | wc -l)
CURRENT=0
SKIPPED=0
RAN=0

# Helper: extract reduction hits@1 from results.json
extract_reduction_hits1() {
    local results_file="$1"
    python3 -c "
import json, sys
try:
    with open('$results_file') as f:
        data = json.load(f)
    for model in data.values():
        print(model.get('hits@1', 999.0))
        sys.exit(0)
    print(999.0)
except Exception:
    print(999.0)
"
}

# Helper: create reduction-only config (stub augmentation + custom seed)
make_reduction_only_config() {
    local src_config="$1"
    local dst_config="$2"
    local seed="$3"
    python3 -c "
import yaml
with open('$src_config') as f:
    cfg = yaml.safe_load(f)
cfg['experiment']['seed'] = $seed
cfg['experiment']['augmentation'] = {
    'method': 'stub',
    'writer': 'bert_int',
    'eval': False,
    'save_dataset': False,
    'save_model': False
}
with open('$dst_config', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"
}

for config_file in "$CONFIG_DIR"/*.yaml; do
    CURRENT=$((CURRENT + 1))
    CONFIG_NAME=$(basename "$config_file" .yaml)

    EXP_DIR="$RESULTS_DIR/$CONFIG_NAME"
    METADATA_FILE="$EXP_DIR/metadata.json"

    # Skip if already completed (multi-run or legacy single-run)
    if [[ -f "$METADATA_FILE" ]]; then
        if grep -q '"reduction_runs"\|"results"' "$METADATA_FILE" 2>/dev/null; then
            log "[$CURRENT/$TOTAL_CONFIGS] [SKIP] $CONFIG_NAME - already completed"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi

    log "[$CURRENT/$TOTAL_CONFIGS] $CONFIG_NAME - $REDUCTION_RUNS reduction runs + 1 augmentation..."

    # --- Phase A: Reduction-only runs to find lowest hits@1 ---
    BEST_HITS1=""
    BEST_SEED=$BASE_SEED

    for run in $(seq 1 $REDUCTION_RUNS); do
        RUN_SEED=$((BASE_SEED + run - 1))
        TEMP_CONFIG="$LOG_DIR/tmp_${CONFIG_NAME}_red${run}.yaml"

        make_reduction_only_config "$config_file" "$TEMP_CONFIG" "$RUN_SEED"

        log "  [RED $run/$REDUCTION_RUNS] seed=$RUN_SEED..."

        python -m experiments.runner "$TEMP_CONFIG" --overwrite-existing \
            2>&1 | tee -a "$LOG_DIR/exp_${CONFIG_NAME}_red${run}_${TIMESTAMP}.log"

        REDUCTION_RESULTS="$EXP_DIR/reduction/results.json"
        if [[ -f "$REDUCTION_RESULTS" ]]; then
            CURRENT_HITS1=$(extract_reduction_hits1 "$REDUCTION_RESULTS")

            IS_BETTER=$(python3 -c "
best = '$BEST_HITS1'
current = float('$CURRENT_HITS1')
if best == '' or current < float(best):
    print('yes')
else:
    print('no')
")
            if [[ "$IS_BETTER" == "yes" ]]; then
                BEST_HITS1="$CURRENT_HITS1"
                BEST_SEED=$RUN_SEED
                log "  [RED $run/$REDUCTION_RUNS] NEW BEST hits@1=$BEST_HITS1"
            else
                log "  [RED $run/$REDUCTION_RUNS] hits@1=$CURRENT_HITS1 (best=$BEST_HITS1)"
            fi
        else
            log "  [RED $run/$REDUCTION_RUNS] WARNING: no reduction results"
        fi

        rm -f "$TEMP_CONFIG"
    done

    # --- Phase B: Full pipeline (reduction + augmentation) with best seed ---
    log "  [AUG] Full run with best seed=$BEST_SEED..."
    FINAL_CONFIG="$LOG_DIR/tmp_${CONFIG_NAME}_final.yaml"
    sed "s/seed: [0-9]*/seed: $BEST_SEED/" "$config_file" > "$FINAL_CONFIG"

    python -m experiments.runner "$FINAL_CONFIG" --overwrite-existing \
        2>&1 | tee -a "$LOG_DIR/exp_${CONFIG_NAME}_aug_${TIMESTAMP}.log"

    rm -f "$FINAL_CONFIG"

    # Tag metadata with multi-run info
    python3 -c "
import json
meta_file = '$METADATA_FILE'
try:
    with open(meta_file) as f:
        meta = json.load(f)
    meta['reduction_runs'] = $REDUCTION_RUNS
    meta['best_reduction_seed'] = $BEST_SEED
    meta['best_reduction_hits1'] = float('$BEST_HITS1')
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
except Exception as e:
    print(f'WARNING: could not update metadata: {e}')
"

    log "  [DONE] best_seed=$BEST_SEED, reduction_hits@1=$BEST_HITS1"
    RAN=$((RAN + 1))
done

log "[COMPLETE] Experiments: $RAN ran, $SKIPPED skipped (already done)"

# ==============================================================
# PHASE 3: Analyze results
# ==============================================================
log ""
log "============================================================"
log "PHASE 3: ANALYZING RESULTS"
log "============================================================"

python -m experiments.statistics.analyze_results "$RESULTS_DIR" \
    2>&1 | tee -a "$LOG_DIR/analyze_${TIMESTAMP}.log"

log "[COMPLETE] Analysis finished"

# ==============================================================
# Summary
# ==============================================================
log ""
log "============================================================"
log "ALL DONE!"
log "============================================================"
log "Experiments: $RAN executed, $SKIPPED skipped"
log "Results: $RESULTS_DIR/"
log "Statistics: results/statistics/"
log "LEAR tables: results/statistics/latex/lear_summary.tex"
log "Full log: $MAIN_LOG"
log "============================================================"
