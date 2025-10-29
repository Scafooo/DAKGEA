#!/usr/bin/env bash
# ============================================================
#  DAKGEA Launcher
#  Runs an experiment with the correct Python path and config
#  Works on Linux, macOS, and Windows (Git Bash / WSL)
# ============================================================

# Resolve project root (directory where this script lives)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Ensure that src/ is in PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}"

# Allow overriding config via CLI argument or RUN_CONFIG environment variable
FILE_NAME="${1:-${RUN_CONFIG:-exp_4.yaml}}"

# Resolve configuration path (supports names without extension)
resolve_config_path() {
    local candidate="$1"
    local search_paths=()

    if [[ "$candidate" == /* ]]; then
        search_paths+=("$candidate")
    else
        if [[ "$candidate" == */* ]]; then
            search_paths+=("${PROJECT_ROOT}/${candidate}")
        else
            search_paths+=("${PROJECT_ROOT}/config/experiments/${candidate}")
            search_paths+=("${PROJECT_ROOT}/${candidate}")
        fi
    fi

    for path in "${search_paths[@]}"; do
        if [[ -f "$path" ]]; then
            echo "$path"
            return 0
        fi

        case "$path" in
            *.yaml|*.yml) ;;
            *)
                for ext in yaml yml; do
                    if [[ -f "${path}.${ext}" ]]; then
                        echo "${path}.${ext}"
                        return 0
                    fi
                done
                ;;
        esac
    done

    return 1
}

if ! CONFIG_FILE="$(resolve_config_path "$FILE_NAME")"; then
    if [[ "$FILE_NAME" == /* ]]; then
        echo "❌ Configuration file not found: ${FILE_NAME}"
    else
        echo "❌ Configuration file not found: ${PROJECT_ROOT}/config/experiments/${FILE_NAME}"
    fi
    exit 1
fi

# Activate virtual environment automatically (if it exists)
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
elif [ -f "${PROJECT_ROOT}/.venv/Scripts/activate" ]; then  # Windows
    source "${PROJECT_ROOT}/.venv/Scripts/activate"
fi

# Show debug info
echo "-----------------------------------------------"
echo "🧠 DAKGEA experiment launcher"
echo "📂 Project root : ${PROJECT_ROOT}"
echo "🐍 Python path  : ${PYTHONPATH}"
echo "💼 Environment  : ${VIRTUAL_ENV:-system Python}"
echo "📘 Config file  : ${CONFIG_FILE}"
echo "-----------------------------------------------"

# Verify config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ Configuration file not found: ${CONFIG_FILE}"
    exit 1
fi

# Run the experiment
python "${PROJECT_ROOT}/experiments/run.py" "${CONFIG_FILE}"

# Forward exit code
exit $?
