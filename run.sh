#!/usr/bin/env bash
# ============================================================
#  DAKGEA Launcher
#  Data Augmentation for Knowledge Graph Entity Resolution
#  Runs an experiment with the correct Python path and config
# ============================================================

# ============================================================
#  EXPERIMENT SELECTION
#  Modify this line to select a different experiment
# ============================================================
EXPERIMENT="${EXPERIMENT:-exp_8.yaml}"

# ---------- Helpers ----------
term_width() { tput cols 2>/dev/null || echo 80; }
full_line() { printf '%*s\n' "$(term_width)" '' | tr ' ' "$1"; }

# ---------- Clear screen + banner ----------
clear
full_line '-'
printf "%*s\n" $((($(term_width) + 40) / 2)) "Data Augmentation for Knowledge Graphs Entity Resolution"
full_line '-'

# ---------- Setup paths ----------
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${PROJECT_ROOT}"
FILE_NAME="${1:-${RUN_CONFIG:-${EXPERIMENT}}}"

# ---------- Resolve configuration ----------
resolve_config_path() {
    local candidate="$1"
    local search_paths=(
        "$candidate"
        "${PROJECT_ROOT}/config/experiments/${candidate}"
        "${PROJECT_ROOT}/${candidate}"
    )
    for p in "${search_paths[@]}"; do
        [[ -f "$p" ]] && { echo "$p"; return 0; }
        [[ -f "$p.yaml" ]] && { echo "$p.yaml"; return 0; }
        [[ -f "$p.yml" ]] && { echo "$p.yml"; return 0; }
    done
    return 1
}

if ! CONFIG_FILE="$(resolve_config_path "$FILE_NAME")"; then
    echo "❌ Configuration file not found: ${FILE_NAME}"
    exit 1
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
echo "📘 Config file  : ${CONFIG_FILE}"
echo "🌿 Git branch   : ${BRANCH}"
echo "💻 Hardware     : ${GPU}"
echo "🕓 Started at   : $(date '+%Y-%m-%d %H:%M:%S %Z')"
full_line '-'

# ---------- Run experiment ----------
python "${PROJECT_ROOT}/experiments/run.py" "${CONFIG_FILE}"
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Experiment completed successfully!"
else
    echo "❌ Experiment failed with exit code ${EXIT_CODE}"
fi

full_line '-'
exit $EXIT_CODE
