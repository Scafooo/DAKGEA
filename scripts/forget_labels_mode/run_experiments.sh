#!/usr/bin/env bash
# ============================================================
#  Forget Labels Experiment Launcher
#  Mirror behavior of scripts/run_experiment.sh but for custom mode
# ============================================================

set -euo pipefail

# ---------- Configuration ----------
DEFAULT_JOBS=2
DEFAULT_GPU_ID=0
TIMEOUT=7200

# ---------- Helpers ----------
term_width() { tput cols 2>/dev/null || echo 80; }
full_line() { printf '%*s\n' "$(term_width)" '' | tr ' ' "$1"; }

# ---------- Setup paths ----------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}"

# Default experiment target if none provided
FORGET_LABELS_ROOT="config/experiments/massive/forget_labels"
FILE_NAME="${1:-${FORGET_LABELS_ROOT}}"

# ---------- Resolve configuration target (file or directory) ----------
resolve_target_path() {
    local candidate="$1"
    local search_paths=(
        "$candidate"
        "${PROJECT_ROOT}/${candidate}"
        "${PROJECT_ROOT}/config/experiments/massive/${candidate}"
    )
    for p in "${search_paths[@]}"; do
        if [[ -f "$p" || -d "$p" ]]; then
            echo "$p"
            return 0
        fi
        if [[ -f "$p.yaml" ]]; then
            echo "$p.yaml"
            return 0
        fi
    done
    return 1
}

if ! TARGET_PATH="$(resolve_target_path "$FILE_NAME")"; then
    echo "❌ Configuration target not found: ${FILE_NAME}"
    exit 1
fi

if [[ -d "$TARGET_PATH" ]]; then
    MODE="batch"
    mapfile -t CONFIG_SET < <(find "$TARGET_PATH" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) | sort)
    if [[ ${#CONFIG_SET[@]} -eq 0 ]]; then
        echo "❌ No YAML configuration files found in directory: ${TARGET_PATH}"
        exit 1
    fi
else
    MODE="single"
    CONFIG_SET=("$TARGET_PATH")
fi

# ---------- Banner ----------
clear
full_line '-'
printf "%*s\n" $((($(term_width) + 40) / 2)) "Forget Labels Experiment Runner"
full_line '-'
echo "📂 Project root : ${PROJECT_ROOT}"
if [[ "$MODE" == "single" ]]; then
    echo "📘 Config file  : ${CONFIG_SET[0]}"
else
    echo "📁 Config dir   : ${TARGET_PATH}"
    echo "🧪 Config count : ${#CONFIG_SET[@]}"
fi
full_line '-'

# ---------- Confirmation ----------
if [[ "$MODE" == "batch" ]]; then
    read -r -p "Run all ${#CONFIG_SET[@]} configurations in parallel? [y/N] " CONFIRM
    case "${CONFIRM,,}" in
        y|yes) ;; 
        *) echo "ℹ️ Aborted."; exit 0 ;; 
    esac
fi

# ---------- Parallel Setup ----------
if command -v parallel &> /dev/null; then
    PARALLEL_BIN="parallel"
elif [ -f "${PROJECT_ROOT}/.local/bin/parallel" ]; then
    PARALLEL_BIN="${PROJECT_ROOT}/.local/bin/parallel"
else
    echo "⚠️ GNU Parallel not found. Running sequentially."
    PARALLEL_BIN=""
fi

export GPU_ID="${DEFAULT_GPU_ID}"
LOG_DIR="${PROJECT_ROOT}/results/logs/forget_labels_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
JOBLOG="${LOG_DIR}/joblog.txt"

# ---------- Run ----------
if [[ -n "${PARALLEL_BIN}" && "$MODE" == "batch" ]]; then
    echo "▶️ Starting parallel execution (Jobs: ${DEFAULT_JOBS}, GPU: ${DEFAULT_GPU_ID})..."
    "${PARALLEL_BIN}" --will-cite --jobs "${DEFAULT_JOBS}" \
        --joblog "${JOBLOG}" --timeout "${TIMEOUT}" --progress \
        --results "${LOG_DIR}" \
        bash "${SCRIPT_DIR}/_run_single_experiment.sh" ::: "${CONFIG_SET[@]}"
else
    for CONFIG_FILE in "${CONFIG_SET[@]}"; do
        echo ""
        full_line '='
        echo "▶️ Running configuration: ${CONFIG_FILE}"
        bash "${SCRIPT_DIR}/_run_single_experiment.sh" "${CONFIG_FILE}"
        full_line '='
    done
fi

echo ""
full_line '-'
echo "✅ Done! Logs available in: ${LOG_DIR}"
full_line '-'