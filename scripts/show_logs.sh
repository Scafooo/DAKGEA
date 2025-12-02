#!/usr/bin/env bash
# Helper script to view logs from parallel runs

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
LOGS_DIR="${PROJECT_ROOT}/results/logs"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [LOG_DIR]

View logs from parallel experiment runs.

OPTIONS:
    --list              List all available run logs
    --latest            Use latest run (default if no LOG_DIR specified)
    --failed            Show only failed experiments
    --errors            Show recent errors from stderr
    --summary           Show summary
    --joblog            Show job log
    --tail N            Tail last N lines from each stderr (default: 20)
    --help              Show this help

ARGUMENTS:
    LOG_DIR            Specific log directory to view (optional)

EXAMPLES:
    # Show logs from latest run
    $0 --latest

    # List all runs
    $0 --list

    # Show only failed experiments from latest
    $0 --latest --failed

    # Show errors from specific run
    $0 results/logs/parallel_run_20251202_143022 --errors

    # Tail last 50 lines of errors
    $0 --latest --errors --tail 50

EOF
    exit 0
}

list_runs() {
    echo "Available parallel runs:"
    echo ""
    ls -lt "${LOGS_DIR}" | grep parallel_run | head -10 | while read -r line; do
        dir=$(echo "$line" | awk '{print $NF}')
        date=$(echo "$line" | awk '{print $6, $7, $8}')
        echo "  ${dir} (${date})"
    done
    exit 0
}

# Parse arguments
LIST=false
LATEST=false
FAILED=false
ERRORS=false
SUMMARY=false
JOBLOG_ONLY=false
TAIL_LINES=20
LOG_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)
            LIST=true
            shift
            ;;
        --latest)
            LATEST=true
            shift
            ;;
        --failed)
            FAILED=true
            shift
            ;;
        --errors)
            ERRORS=true
            shift
            ;;
        --summary)
            SUMMARY=true
            shift
            ;;
        --joblog)
            JOBLOG_ONLY=true
            shift
            ;;
        --tail)
            TAIL_LINES="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            if [[ -z "$LOG_DIR" ]]; then
                LOG_DIR="$1"
            else
                echo "Unknown option: $1"
                usage
            fi
            shift
            ;;
    esac
done

# Handle --list
if [[ "$LIST" == "true" ]]; then
    list_runs
fi

# Find log directory
if [[ -z "$LOG_DIR" ]] || [[ "$LATEST" == "true" ]]; then
    LOG_DIR=$(ls -t "${LOGS_DIR}" | grep parallel_run | head -1)
    if [[ -z "$LOG_DIR" ]]; then
        echo "Error: No parallel run logs found in ${LOGS_DIR}"
        exit 1
    fi
    LOG_DIR="${LOGS_DIR}/${LOG_DIR}"
    echo "Using latest run: ${LOG_DIR}"
    echo ""
fi

# Validate directory
if [[ ! -d "$LOG_DIR" ]]; then
    # Try relative to logs dir
    if [[ -d "${LOGS_DIR}/${LOG_DIR}" ]]; then
        LOG_DIR="${LOGS_DIR}/${LOG_DIR}"
    else
        echo "Error: Log directory not found: ${LOG_DIR}"
        exit 1
    fi
fi

JOBLOG="${LOG_DIR}/joblog.txt"
SUMMARY="${LOG_DIR}/summary.txt"

if [[ ! -f "$JOBLOG" ]]; then
    echo "Error: No joblog found in ${LOG_DIR}"
    exit 1
fi

# Show summary if requested
if [[ "$SUMMARY" == "true" ]]; then
    if [[ -f "$SUMMARY" ]]; then
        cat "$SUMMARY"
    else
        echo "No summary file found"
    fi
    exit 0
fi

# Show joblog if requested
if [[ "$JOBLOG_ONLY" == "true" ]]; then
    cat "$JOBLOG"
    exit 0
fi

# Parse joblog
TOTAL=$(tail -n +2 "$JOBLOG" | wc -l)
COMPLETED=$(tail -n +2 "$JOBLOG" | awk '$7 == 0' | wc -l)
FAILED=$(tail -n +2 "$JOBLOG" | awk '$7 != 0 && $7 != ""' | wc -l)

echo "=== Experiment Run Summary ==="
echo "Total experiments: ${TOTAL}"
echo "Completed: ${COMPLETED}"
echo "Failed: ${FAILED}"
echo ""

# Show failed experiments
if [[ "$FAILED" == "true" ]] || [[ $FAILED -gt 0 ]]; then
    echo "=== Failed Experiments ==="
    tail -n +2 "$JOBLOG" | awk '$7 != 0 && $7 != "" {print $1, $10, "exit:", $7}' | while read -r seq cmd exit_code rest; do
        config=$(basename "$cmd")
        echo "  #${seq}: ${config} (${exit_code} ${rest})"
    done
    echo ""
fi

# Show errors if requested
if [[ "$ERRORS" == "true" ]]; then
    echo "=== Recent Errors (last ${TAIL_LINES} lines from each stderr) ==="
    echo ""

    for stderr in "${LOG_DIR}"/*/stderr; do
        if [[ -f "$stderr" ]] && [[ -s "$stderr" ]]; then
            exp_num=$(basename "$(dirname "$stderr")")
            echo "--- Experiment #${exp_num} ---"
            tail -${TAIL_LINES} "$stderr" | sed 's/^/  /'
            echo ""
        fi
    done
fi

# Default: show overview
if [[ "$FAILED" == "false" ]] && [[ "$ERRORS" == "false" ]]; then
    echo "Use --failed to see failed experiments"
    echo "Use --errors to see error messages"
    echo "Use --summary to see full summary"
    echo "Use --joblog to see full job log"
fi
