#!/usr/bin/env bash
# ============================================================
#  DAKGEA Parallel Launcher
#  Data Augmentation for Knowledge Graph Entity Resolution
#  Runs multiple experiments in parallel using GNU Parallel
# ============================================================

set -euo pipefail

# ============================================================
#  DEFAULT CONFIGURATION
# ============================================================
DEFAULT_JOBS=4          # Number of parallel jobs (good for RTX 4090)
DEFAULT_RETRY=1         # Number of retries for failed jobs
DEFAULT_TIMEOUT=7200    # Timeout per job in seconds (2 hours)

# ============================================================
#  ARGUMENT PARSING
# ============================================================
usage() {
    cat <<EOF
Usage: $0 [OPTIONS] --dir EXPERIMENT_DIR

Run multiple DAKGEA experiments in parallel using GNU Parallel.

OPTIONS:
    --dir DIR               Directory containing YAML experiment configs (required)
    --jobs N                Number of parallel jobs (default: ${DEFAULT_JOBS})
    --retry N               Number of retries for failed jobs (default: ${DEFAULT_RETRY})
    --timeout SECONDS       Timeout per job in seconds (default: ${DEFAULT_TIMEOUT})
    --resume                Resume from previous interrupted run
    --dry-run               Show what would be executed without running
    --pattern PATTERN       Glob pattern for YAML files (default: "*.yaml")
    --gpu-id ID             GPU device ID to use (default: 0)
    --verbose               Show live output from experiments (disables --results logging)
    --help                  Show this help message

EXAMPLES:
    # Run all experiments in a directory with 4 parallel jobs
    $0 --dir config/experiments/massive/bert_int_aug_red --jobs 4

    # Dry run to see what will be executed
    $0 --dir config/experiments/massive/bert_int_aug_red --dry-run

    # Resume interrupted run
    $0 --dir config/experiments/massive/bert_int_aug_red --resume

    # Run with 6 parallel jobs on GPU 0
    $0 --dir config/experiments/massive/bert_int_aug_red --jobs 6 --gpu-id 0

EOF
    exit 0
}

# Parse arguments
DIR=""
JOBS="${DEFAULT_JOBS}"
RETRY="${DEFAULT_RETRY}"
TIMEOUT="${DEFAULT_TIMEOUT}"
RESUME=false
DRY_RUN=false
PATTERN="*.yaml"
GPU_ID=0
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dir)
            DIR="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --retry)
            RETRY="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --pattern)
            PATTERN="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [[ -z "$DIR" ]]; then
    echo "Error: --dir is required"
    usage
fi

# ============================================================
#  SETUP
# ============================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}"
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Check if GNU Parallel is installed (local first, then system)
PARALLEL_BIN=""
if [ -f "${PROJECT_ROOT}/.local/bin/parallel" ]; then
    PARALLEL_BIN="${PROJECT_ROOT}/.local/bin/parallel"
    echo "Using local GNU Parallel: ${PARALLEL_BIN}"
elif command -v parallel &> /dev/null; then
    PARALLEL_BIN="parallel"
    echo "Using system GNU Parallel"
else
    echo "Error: GNU Parallel is not installed"
    echo ""
    echo "Install locally (no sudo required):"
    echo "  bash scripts/install_parallel_local.sh"
    echo ""
    echo "Or install system-wide:"
    echo "  sudo dnf install parallel"
    exit 1
fi

# Resolve experiment directory
resolve_target_path() {
    local candidate="$1"
    local search_paths=(
        "$candidate"
        "${PROJECT_ROOT}/config/experiments/${candidate}"
        "${PROJECT_ROOT}/${candidate}"
    )
    for p in "${search_paths[@]}"; do
        if [[ -d "$p" ]]; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

if ! TARGET_DIR="$(resolve_target_path "$DIR")"; then
    echo "Error: Directory not found: ${DIR}"
    exit 1
fi

# Find all YAML files
mapfile -t ALL_CONFIG_FILES < <(find "$TARGET_DIR" -maxdepth 1 -type f -name "${PATTERN}" | sort)

if [[ ${#ALL_CONFIG_FILES[@]} -eq 0 ]]; then
    echo "Error: No YAML files found in ${TARGET_DIR} matching pattern '${PATTERN}'"
    exit 1
fi

# Note: The runner now does early completion checks internally
# Each experiment will exit immediately if already complete (when overwrite_existing=false)
# This is much faster than pre-checking because it happens in parallel
CONFIG_FILES=("${ALL_CONFIG_FILES[@]}")

# Setup log directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/results/logs/parallel_run_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

JOBLOG_FILE="${LOG_DIR}/joblog.txt"
SUMMARY_FILE="${LOG_DIR}/summary.txt"

# Activate virtual environment
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
elif [ -f "${PROJECT_ROOT}/.venv/Scripts/activate" ]; then
    source "${PROJECT_ROOT}/.venv/Scripts/activate"
fi

# ============================================================
#  BANNER
# ============================================================
term_width() { tput cols 2>/dev/null || echo 80; }
full_line() { printf '%*s\n' "$(term_width)" '' | tr ' ' "$1"; }

clear
full_line '='
printf "%*s\n" $((($(term_width) + 50) / 2)) "DAKGEA Parallel Experiment Runner"
full_line '='
echo ""
echo "Configuration:"
echo "  Experiment dir    : ${TARGET_DIR}"
echo "  Total configs     : ${#CONFIG_FILES[@]}"
echo "  Parallel jobs     : ${JOBS}"
echo "  Retry attempts    : ${RETRY}"
echo "  Timeout per job   : ${TIMEOUT}s"
echo "  GPU device        : ${GPU_ID}"
echo "  Log directory     : ${LOG_DIR}"
echo "  Resume mode       : ${RESUME}"
echo "  Verbose output    : ${VERBOSE}"
echo "  Dry run           : ${DRY_RUN}"
echo ""

# GPU Info
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i ${GPU_ID} 2>/dev/null || echo "Unknown")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader -i ${GPU_ID} 2>/dev/null || echo "Unknown")
    echo "GPU Information:"
    echo "  Device ${GPU_ID}       : ${GPU_NAME}"
    echo "  Total memory    : ${GPU_MEM}"
    echo ""
fi

full_line '-'

# ============================================================
#  DRY RUN
# ============================================================
if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN - Would execute the following experiments:"
    echo ""
    for cfg in "${CONFIG_FILES[@]}"; do
        echo "  - $(basename "$cfg")"
    done
    echo ""
    echo "Parallel command structure:"
    echo "  ${PARALLEL_BIN} --will-cite --jobs ${JOBS} --bar --joblog ${JOBLOG_FILE} \\"
    echo "    --retry ${RETRY} --timeout ${TIMEOUT} --results ${LOG_DIR} \\"
    echo "    run_experiment {} ::: <config_files>"
    echo ""
    echo "Where run_experiment executes:"
    echo "  CUDA_VISIBLE_DEVICES=${GPU_ID} python experiments/runner/run.py <config_file>"
    echo ""
    exit 0
fi

# ============================================================
#  CONFIRMATION
# ============================================================
read -r -p "Run ${#CONFIG_FILES[@]} experiments with ${JOBS} parallel jobs? [y/N] " CONFIRM
case "${CONFIRM,,}" in
    y|yes)
        ;;
    *)
        echo "Aborted by user."
        exit 0
        ;;
esac

echo ""
full_line '='
echo "Starting parallel execution..."
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
full_line '='
echo ""

# ============================================================
#  PARALLEL EXECUTION
# ============================================================
START_TIME=$(date +%s)

# Build parallel command arguments (without the binary itself)
PARALLEL_ARGS=(
    --will-cite
    --jobs "${JOBS}"
    --joblog "${JOBLOG_FILE}"
    --retry "${RETRY}"
    --timeout "${TIMEOUT}"
)

# Add progress bar only if not verbose
if [[ "$VERBOSE" == "false" ]]; then
    PARALLEL_ARGS+=(--bar)
    PARALLEL_ARGS+=(--results "${LOG_DIR}")
else
    # In verbose mode, show output directly
    echo "Note: Verbose mode enabled - output will be shown directly (not saved to --results)"
fi

if [[ "$RESUME" == "true" ]]; then
    PARALLEL_ARGS+=(--resume)
fi

# Export required variables for parallel
export PROJECT_ROOT
export PYTHONPATH
export CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED
export GPU_ID

# Define function to run experiment (will be called by parallel)
run_experiment() {
    local config_file="$1"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${PROJECT_ROOT}/experiments/runner/run.py" "${config_file}"
}
export -f run_experiment

# Run parallel execution
printf '%s\n' "${CONFIG_FILES[@]}" | "${PARALLEL_BIN}" "${PARALLEL_ARGS[@]}" run_experiment {}

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ============================================================
#  SUMMARY
# ============================================================
echo ""
full_line '='
echo "Parallel execution completed!"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total time: ${ELAPSED}s ($(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))))"
full_line '='
echo ""

# Parse joblog to get statistics
if [[ -f "${JOBLOG_FILE}" ]]; then
    TOTAL=$(tail -n +2 "${JOBLOG_FILE}" | wc -l)
    COMPLETED=$(tail -n +2 "${JOBLOG_FILE}" | awk '$7 == 0' | wc -l)
    FAILED=$(tail -n +2 "${JOBLOG_FILE}" | awk '$7 != 0' | wc -l)

    echo "Results:"
    echo "  Total experiments : ${TOTAL}"
    echo "  Completed         : ${COMPLETED}"
    echo "  Failed            : ${FAILED}"
    echo ""

    if [[ ${FAILED} -gt 0 ]]; then
        echo "Failed experiments:"
        tail -n +2 "${JOBLOG_FILE}" | awk '$7 != 0 {print "  - " $10 " (exit code: " $7 ")"}' | head -20
        if [[ ${FAILED} -gt 20 ]]; then
            echo "  ... and $((FAILED - 20)) more"
        fi
        echo ""
    fi

    # Write summary file
    {
        echo "DAKGEA Parallel Experiment Run Summary"
        echo "======================================="
        echo ""
        echo "Run timestamp: ${TIMESTAMP}"
        echo "Experiment directory: ${TARGET_DIR}"
        echo "Total experiments: ${TOTAL}"
        echo "Completed: ${COMPLETED}"
        echo "Failed: ${FAILED}"
        echo "Parallel jobs: ${JOBS}"
        echo "Total time: ${ELAPSED}s"
        echo ""
        if [[ ${FAILED} -gt 0 ]]; then
            echo "Failed experiments:"
            tail -n +2 "${JOBLOG_FILE}" | awk '$7 != 0 {print "  " $10 " (exit code: " $7 ")"}'
        fi
    } > "${SUMMARY_FILE}"

    echo "Detailed logs: ${LOG_DIR}"
    echo "Job log: ${JOBLOG_FILE}"
    echo "Summary: ${SUMMARY_FILE}"
fi

full_line '='

exit ${EXIT_CODE}
