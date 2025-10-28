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

FILE_NAME="exp_3.yaml"

# Default config file (you can change this)
CONFIG_FILE="${PROJECT_ROOT}/config/experiments/${FILE_NAME}"

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
