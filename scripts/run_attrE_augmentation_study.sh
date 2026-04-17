#!/bin/bash
#
# AttrE Augmentation Study
# ========================
# Mirrors run_flan_t5_experiments.sh for AttrE.
# Runs 15 experiments: 5 datasets × 3 reduction ratios (0.1/0.3/0.5) × aug=0.3
#
# Each (dataset, reduction_ratio) group:
#   1. 5 reduction-only trials → select seed with lowest hits@1
#   2. Run augmentation experiment with best seed
#
# Usage:
#   bash scripts/run_attrE_augmentation_study.sh
#
# Requirements:
#   - NVIDIA GPU (RTX 4090 recommended)
#   - Pre-trained PLM models in models/pretrained_plm/<DATASET>/
#     (same models used by flan_t5_bert_int — re-used if already present)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Configuration
DATASETS=("BBC_DB" "D_W_15K_V1" "D_W_15K_V2" "ICEWS_WIKI" "ICEWS_YAGO")
CONFIG_DIR="config/experiments/massive/flan_t5_attrE"
RESULTS_DIR="results/flan_t5_attrE"
LOG_DIR="results/logs/attrE_augmentation_study"
PRETRAINED_DIR="models/pretrained_plm"
REDUCTION_RUNS=5
BASE_SEED=11037

# Only the 15 configs for this study (red=0.1/0.3/0.5, aug=0.3)
STUDY_PATTERN="*_0[135]_03.yaml"

mkdir -p "$LOG_DIR"
mkdir -p "$PRETRAINED_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/full_run_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

log "============================================================"
log "AttrE AUGMENTATION STUDY"
log "============================================================"
log "Datasets:    ${DATASETS[*]}"
log "Configs:     15  (5 datasets × red=0.1/0.3/0.5 × aug=0.3)"
log "Red trials:  $REDUCTION_RUNS per group (select lowest hits@1)"
log "Log file:    $MAIN_LOG"
log "============================================================"

# ==============================================================
# PHASE 1: Pre-train PLM models (skip if already present)
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

log "[COMPLETE] All models ready"

# ==============================================================
# PHASE 2: Run experiments
# ==============================================================
log ""
log "============================================================"
log "PHASE 2: RUNNING EXPERIMENTS"
log "============================================================"

# Helpers ----------------------------------------------------------

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
    'writer': {'type': 'openea', 'augmented_only_train': True},
    'eval': False,
    'save_dataset': False,
    'save_model': False,
}
with open('$dst_config', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"
}

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

# ------------------------------------------------------------------

mapfile -t ALL_CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name "$STUDY_PATTERN" | sort)

if [[ ${#ALL_CONFIGS[@]} -eq 0 ]]; then
    log "[ERROR] No configs found matching '$STUDY_PATTERN' in $CONFIG_DIR"
    exit 1
fi

log "Selected ${#ALL_CONFIGS[@]} configs:"
for cfg in "${ALL_CONFIGS[@]}"; do log "  - $(basename "$cfg")"; done

TOTAL_CONFIGS=${#ALL_CONFIGS[@]}
CURRENT=0
SKIPPED=0
RAN=0

# Snapshot which configs already have results
declare -A HAD_RESULTS
for config_file in "${ALL_CONFIGS[@]}"; do
    CONFIG_NAME=$(basename "$config_file" .yaml)
    META="$RESULTS_DIR/$CONFIG_NAME/metadata.json"
    if [[ -f "$META" ]] && grep -q '"results"\|"evaluations"' "$META" 2>/dev/null; then
        HAD_RESULTS[$CONFIG_NAME]="yes"
    fi
done

# Group configs by {DATASET}_{RED} (e.g. BBC_DB_01)
declare -A REDUCTION_GROUPS
declare -A GROUP_CONFIGS

for config_file in "${ALL_CONFIGS[@]}"; do
    CONFIG_NAME=$(basename "$config_file" .yaml)
    GROUP_KEY="${CONFIG_NAME%_*}"   # strip last _XX  →  BBC_DB_01
    if [[ -z "${REDUCTION_GROUPS[$GROUP_KEY]+x}" ]]; then
        REDUCTION_GROUPS[$GROUP_KEY]="$config_file"
    fi
    GROUP_CONFIGS[$GROUP_KEY]="${GROUP_CONFIGS[$GROUP_KEY]:-} $config_file"
done

TOTAL_GROUPS=${#REDUCTION_GROUPS[@]}
GROUP_CURRENT=0

for GROUP_KEY in $(echo "${!REDUCTION_GROUPS[@]}" | tr ' ' '\n' | sort); do
    GROUP_CURRENT=$((GROUP_CURRENT + 1))
    TEMPLATE_CONFIG="${REDUCTION_GROUPS[$GROUP_KEY]}"

    log ""
    log "--- [$GROUP_CURRENT/$TOTAL_GROUPS] Group: $GROUP_KEY ---"

    # Try to recover seed from existing metadata
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
                EXISTING_REDUCTION_RESULTS="$RESULTS_DIR/$CFG_NAME/reduction/results.json"
            fi
        else
            GROUP_ALL_DONE=false
        fi
    done

    if [[ "$GROUP_ALL_DONE" == "true" ]]; then
        log "  [SKIP] all configs already completed"
        for cfg in ${GROUP_CONFIGS[$GROUP_KEY]}; do
            CURRENT=$((CURRENT + 1))
            SKIPPED=$((SKIPPED + 1))
        done
        continue
    fi

    if [[ -n "$RECOVERED_SEED" ]]; then
        BEST_SEED=$RECOVERED_SEED
        BEST_HITS1=${RECOVERED_HITS1:-0}
        log "  [REUSE] recovered seed=$BEST_SEED, hits@1=$BEST_HITS1"

    else
        log "  [REDUCTION] $REDUCTION_RUNS trials to select best seed..."

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
        EXISTING_REDUCTION_RESULTS="$BEST_REDUCTION_FILE"

        rm -rf "$TRIAL_EXP_DIR"
    fi

    # Run pending configs in this group
    for cfg in ${GROUP_CONFIGS[$GROUP_KEY]}; do
        CFG_NAME=$(basename "$cfg" .yaml)
        CURRENT=$((CURRENT + 1))

        if [[ "${HAD_RESULTS[$CFG_NAME]:-no}" == "yes" ]]; then
            log "  [$CURRENT/$TOTAL_CONFIGS] [SKIP] $CFG_NAME - already done"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        log "  [$CURRENT/$TOTAL_CONFIGS] [RUN] $CFG_NAME (seed=$BEST_SEED)..."

        FINAL_CONFIG="$LOG_DIR/tmp_${CFG_NAME}_final.yaml"
        make_noeval_config "$cfg" "$FINAL_CONFIG" "$BEST_SEED"

        python -m experiments.runner "$FINAL_CONFIG" --overwrite-existing \
            2>&1 | tee -a "$LOG_DIR/exp_${CFG_NAME}_${TIMESTAMP}.log"

        rm -f "$FINAL_CONFIG"

        CFG_RED_DIR="$RESULTS_DIR/$CFG_NAME/reduction"
        if [[ -f "$EXISTING_REDUCTION_RESULTS" ]] && [[ -d "$CFG_RED_DIR" ]]; then
            cp "$EXISTING_REDUCTION_RESULTS" "$CFG_RED_DIR/results.json"
        fi

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
log "Results:     $RESULTS_DIR/"
log "Full log:    $MAIN_LOG"
log "============================================================"
