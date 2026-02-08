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

# Helper: create reduction-only config (stub augmentation + custom seed + trial name)
make_reduction_only_config() {
    local src_config="$1"
    local dst_config="$2"
    local seed="$3"
    local trial_name="$4"
    python3 -c "
import yaml
with open('$src_config') as f:
    cfg = yaml.safe_load(f)
cfg['experiment']['seed'] = $seed
cfg['experiment']['name'] = '$trial_name'
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

# Helper: create config with reduction.eval=false + custom seed
make_noeval_config() {
    local src_config="$1"
    local dst_config="$2"
    local seed="$3"
    python3 -c "
import yaml
with open('$src_config') as f:
    cfg = yaml.safe_load(f)
cfg['experiment']['seed'] = $seed
cfg['experiment']['reduction']['eval'] = False
with open('$dst_config', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"
}

# ---- Step 0: Snapshot which configs already have results ----
declare -A HAD_RESULTS  # key=CONFIG_NAME -> "yes" if results existed before we start
for config_file in "$CONFIG_DIR"/*.yaml; do
    CONFIG_NAME=$(basename "$config_file" .yaml)
    META="$RESULTS_DIR/$CONFIG_NAME/metadata.json"
    if [[ -f "$META" ]] && grep -q '"results"\|"evaluations"' "$META" 2>/dev/null; then
        HAD_RESULTS[$CONFIG_NAME]="yes"
    fi
done

# ---- Collect groups: {DATASET}_{RED} -> list of config files ----
declare -A REDUCTION_GROUPS  # key="DATASET_RED" -> first config (template)
declare -A GROUP_CONFIGS     # key="DATASET_RED" -> space-separated config files

for config_file in "$CONFIG_DIR"/*.yaml; do
    CONFIG_NAME=$(basename "$config_file" .yaml)
    GROUP_KEY="${CONFIG_NAME%_*}"
    if [[ -z "${REDUCTION_GROUPS[$GROUP_KEY]+x}" ]]; then
        REDUCTION_GROUPS[$GROUP_KEY]="$config_file"
    fi
    GROUP_CONFIGS[$GROUP_KEY]="${GROUP_CONFIGS[$GROUP_KEY]:-} $config_file"
done

# ---- Per-group loop: reduction selection + immediate execution ----
# For each (dataset, ratio) group:
#   1. Select best reduction seed (5 runs) or recover from existing
#   2. Immediately run all pending configs with that seed

TOTAL_GROUPS=${#REDUCTION_GROUPS[@]}
GROUP_CURRENT=0

for GROUP_KEY in $(echo "${!REDUCTION_GROUPS[@]}" | tr ' ' '\n' | sort); do
    GROUP_CURRENT=$((GROUP_CURRENT + 1))
    TEMPLATE_CONFIG="${REDUCTION_GROUPS[$GROUP_KEY]}"

    log ""
    log "--- [$GROUP_CURRENT/$TOTAL_GROUPS] Group: $GROUP_KEY ---"

    # Check if ANY config in this group already has reduction_runs
    RECOVERED_SEED=""
    RECOVERED_HITS1=""
    EXISTING_REDUCTION_RESULTS=""
    GROUP_ALL_DONE=true
    for cfg in ${GROUP_CONFIGS[$GROUP_KEY]}; do
        CFG_NAME=$(basename "$cfg" .yaml)
        CFG_META="$RESULTS_DIR/$CFG_NAME/metadata.json"
        if [[ -f "$CFG_META" ]] && grep -q '"reduction_runs"' "$CFG_META" 2>/dev/null; then
            if [[ -z "$RECOVERED_SEED" ]]; then
                RECOVERED_SEED=$(python3 -c "
import json
with open('$CFG_META') as f:
    meta = json.load(f)
print(meta.get('best_reduction_seed', $BASE_SEED))
")
                RECOVERED_HITS1=$(python3 -c "
import json
with open('$CFG_META') as f:
    meta = json.load(f)
print(meta.get('best_reduction_hits1', 0))
")
                # Remember where the reduction results.json is
                EXISTING_REDUCTION_RESULTS="$RESULTS_DIR/$CFG_NAME/reduction/results.json"
            fi
        else
            GROUP_ALL_DONE=false
        fi
    done

    # Case 1: All configs in group already completed
    if [[ "$GROUP_ALL_DONE" == "true" ]]; then
        log "  [SKIP] all configs already completed"
        for cfg in ${GROUP_CONFIGS[$GROUP_KEY]}; do
            CURRENT=$((CURRENT + 1))
            SKIPPED=$((SKIPPED + 1))
        done
        continue
    fi

    # Case 2: Some configs done, recover seed
    if [[ -n "$RECOVERED_SEED" ]]; then
        BEST_SEED=$RECOVERED_SEED
        BEST_HITS1=${RECOVERED_HITS1:-0}
        log "  [REUSE] recovered seed=$BEST_SEED, hits@1=$BEST_HITS1"

    else
        # Case 3: No configs done, run 5 reduction trials
        log "  [REDUCTION] $REDUCTION_RUNS runs to select best seed..."

        TRIAL_NAME="${GROUP_KEY}_red_trial"
        TRIAL_EXP_DIR="$RESULTS_DIR/$TRIAL_NAME"
        BEST_REDUCTION_FILE="$LOG_DIR/tmp_${GROUP_KEY}_best_reduction.json"

        BEST_HITS1=""
        BEST_SEED=$BASE_SEED

        for run in $(seq 1 $REDUCTION_RUNS); do
            RUN_SEED=$((BASE_SEED + run - 1))
            TEMP_CONFIG="$LOG_DIR/tmp_${GROUP_KEY}_red${run}.yaml"

            make_reduction_only_config "$TEMPLATE_CONFIG" "$TEMP_CONFIG" "$RUN_SEED" "$TRIAL_NAME"

            log "    [RED $run/$REDUCTION_RUNS] seed=$RUN_SEED..."

            python -m experiments.runner "$TEMP_CONFIG" --overwrite-existing \
                2>&1 | tee -a "$LOG_DIR/exp_${GROUP_KEY}_red${run}_${TIMESTAMP}.log"

            REDUCTION_RESULTS="$TRIAL_EXP_DIR/reduction/results.json"
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
                    cp "$REDUCTION_RESULTS" "$BEST_REDUCTION_FILE"
                    log "    [RED $run/$REDUCTION_RUNS] NEW BEST hits@1=$BEST_HITS1"
                else
                    log "    [RED $run/$REDUCTION_RUNS] hits@1=$CURRENT_HITS1 (best=$BEST_HITS1)"
                fi
            else
                log "    [RED $run/$REDUCTION_RUNS] WARNING: no reduction results"
            fi

            rm -f "$TEMP_CONFIG"
        done

        log "  [SELECTED] seed=$BEST_SEED, hits@1=$BEST_HITS1"

        # Save best reduction results.json path for later
        EXISTING_REDUCTION_RESULTS="$BEST_REDUCTION_FILE"

        # Update reduction results for configs that already had results
        for cfg in ${GROUP_CONFIGS[$GROUP_KEY]}; do
            CFG_NAME=$(basename "$cfg" .yaml)
            if [[ "${HAD_RESULTS[$CFG_NAME]:-no}" != "yes" ]]; then
                continue
            fi

            CFG_EXP_DIR="$RESULTS_DIR/$CFG_NAME"
            CFG_RED_DIR="$CFG_EXP_DIR/reduction"
            CFG_META="$CFG_EXP_DIR/metadata.json"

            mkdir -p "$CFG_RED_DIR"
            cp "$BEST_REDUCTION_FILE" "$CFG_RED_DIR/results.json"
            log "  [UPDATE] $CFG_NAME - reduction results updated"

            if [[ -f "$CFG_META" ]]; then
                python3 -c "
import json
try:
    with open('$CFG_META') as f:
        meta = json.load(f)
    meta['reduction_runs'] = $REDUCTION_RUNS
    meta['best_reduction_seed'] = $BEST_SEED
    meta['best_reduction_hits1'] = float('$BEST_HITS1')
    with open('$CFG_META', 'w') as f:
        json.dump(meta, f, indent=2)
except Exception as e:
    print(f'WARNING: could not update metadata: {e}')
"
            fi
        done

        # Clean up trial directory
        rm -rf "$TRIAL_EXP_DIR"
    fi

    # ---- Run pending configs in this group ----
    for cfg in ${GROUP_CONFIGS[$GROUP_KEY]}; do
        CFG_NAME=$(basename "$cfg" .yaml)
        CURRENT=$((CURRENT + 1))

        if [[ "${HAD_RESULTS[$CFG_NAME]:-no}" == "yes" ]]; then
            log "  [$CURRENT/$TOTAL_CONFIGS] [SKIP] $CFG_NAME - already had results"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        log "  [$CURRENT/$TOTAL_CONFIGS] [RUN] $CFG_NAME (seed=$BEST_SEED)..."

        # Run with reduction.eval=false (reduction is fast, no BERT-INT eval)
        FINAL_CONFIG="$LOG_DIR/tmp_${CFG_NAME}_final.yaml"
        make_noeval_config "$cfg" "$FINAL_CONFIG" "$BEST_SEED"

        python -m experiments.runner "$FINAL_CONFIG" --overwrite-existing \
            2>&1 | tee -a "$LOG_DIR/exp_${CFG_NAME}_${TIMESTAMP}.log"

        rm -f "$FINAL_CONFIG"

        # Copy the correct reduction results.json from existing config
        CFG_RED_DIR="$RESULTS_DIR/$CFG_NAME/reduction"
        if [[ -f "$EXISTING_REDUCTION_RESULTS" ]] && [[ -d "$CFG_RED_DIR" ]]; then
            cp "$EXISTING_REDUCTION_RESULTS" "$CFG_RED_DIR/results.json"
        fi

        # Tag metadata
        METADATA_FILE="$RESULTS_DIR/$CFG_NAME/metadata.json"
        if [[ -f "$METADATA_FILE" ]]; then
            python3 -c "
import json
try:
    with open('$METADATA_FILE') as f:
        meta = json.load(f)
    meta['reduction_runs'] = $REDUCTION_RUNS
    meta['best_reduction_seed'] = $BEST_SEED
    meta['best_reduction_hits1'] = float('${BEST_HITS1:-0}')
    with open('$METADATA_FILE', 'w') as f:
        json.dump(meta, f, indent=2)
except Exception as e:
    print(f'WARNING: could not update metadata: {e}')
"
        fi

        RAN=$((RAN + 1))
    done

    # Clean up temp files
    rm -f "$LOG_DIR/tmp_${GROUP_KEY}_best_reduction.json"
done

log ""
log "[COMPLETE] Experiments: $RAN ran, $SKIPPED skipped"

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
