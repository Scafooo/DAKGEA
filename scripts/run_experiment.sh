#!/usr/bin/env bash
# ============================================================
#  DAKGEA Launcher
#  Data Augmentation for Knowledge Graph Entity Resolution
#  Runs an experiment with the correct Python path and config
# ============================================================

# ============================================================
#  EXPERIMENT SELECTION
#  Modify this line to select a different experiment
#  can be also a directory containing multiple experiments
# ============================================================
EXPERIMENT="${EXPERIMENT:-/massive/bert_int_aug_red/}"

# ---------- Helpers ----------
term_width() { tput cols 2>/dev/null || echo 80; }
full_line() { printf '%*s\n' "$(term_width)" '' | tr ' ' "$1"; }

# ---------- Clear screen + banner ----------
clear
full_line '-'
printf "%*s\n" $((($(term_width) + 40) / 2)) "Data Augmentation for Knowledge Graphs Entity Resolution"
full_line '-'

# ---------- Setup paths ----------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}"
FILE_NAME="${1:-${RUN_CONFIG:-${EXPERIMENT}}}"

# ---------- Resolve configuration target (file or directory) ----------
resolve_target_path() {
    local candidate="$1"
    local search_paths=(
        "$candidate"
        "${PROJECT_ROOT}/config/experiments/${candidate}"
        "${PROJECT_ROOT}/${candidate}"
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
        if [[ -f "$p.yml" ]]; then
            echo "$p.yml"
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
    echo "Found the following configurations in ${TARGET_PATH}:"
    for cfg in "${CONFIG_SET[@]}"; do
        echo "  - ${cfg}"
    done
    echo ""
    read -r -p "Run all ${#CONFIG_SET[@]} configurations? [y/N] " CONFIRM
    case "${CONFIRM,,}" in
        y|yes)
            ;;
        *)
            echo "ℹ️  Aborted by user."
            exit 0
            ;;
    esac
else
    MODE="single"
    CONFIG_SET=("$TARGET_PATH")
fi

# ---------- Activate virtual environment ----------
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
elif [ -f "${PROJECT_ROOT}/.venv/Scripts/activate" ]; then  # Windows
    source "${PROJECT_ROOT}/.venv/Scripts/activate"
fi

# ---------- Runtime info ----------
BRANCH=$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no-git")
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
    if [[ -z "$GPU" || "$GPU" == ERROR* ]]; then
        GPU=$(nvidia-smi -L 2>/dev/null | head -n 1 | sed 's/^GPU [0-9]\+: //')
    fi
    [[ -z "$GPU" ]] && GPU="Unknown GPU"
else
    GPU="CPU only"
fi

full_line '-'
echo "📂 Project root : ${PROJECT_ROOT}"
echo "🐍 Python path  : ${PYTHONPATH}"
echo "💼 Environment  : ${VIRTUAL_ENV:-system Python}"
if [[ "$MODE" == "single" ]]; then
    echo "📘 Config file  : ${CONFIG_SET[0]}"
else
    echo "📁 Config dir   : ${TARGET_PATH}"
    echo "🧪 Config count : ${#CONFIG_SET[@]}"
fi
echo "🌿 Git branch   : ${BRANCH}"
echo "💻 Hardware     : ${GPU}"
echo "🕓 Started at   : $(date '+%Y-%m-%d %H:%M:%S %Z')"
full_line '-'

# ---------- Run experiment(s) ----------
OVERALL_EXIT=0
FAILED_RUNS=()

for CONFIG_FILE in "${CONFIG_SET[@]}"; do
    echo ""
    full_line '='
    echo "▶️  Running configuration: ${CONFIG_FILE}"
    python "${PROJECT_ROOT}/experiments/runner/run.py" "${CONFIG_FILE}"
    RUN_EXIT=$?
    if [[ $RUN_EXIT -eq 0 ]]; then
        echo "✅ Completed: ${CONFIG_FILE}"
    else
        echo "❌ Failed (${RUN_EXIT}): ${CONFIG_FILE}"
        FAILED_RUNS+=("${CONFIG_FILE} [exit ${RUN_EXIT}]")
        OVERALL_EXIT=$RUN_EXIT
    fi
    full_line '='
done

echo ""
if [[ ${#FAILED_RUNS[@]} -eq 0 ]]; then
    echo "✅ All experiments completed successfully!"
else
    echo "❌ ${#FAILED_RUNS[@]} experiment(s) failed:"
    for entry in "${FAILED_RUNS[@]}"; do
        echo "   - ${entry}"
    done
fi

full_line '-'
exit $OVERALL_EXIT
