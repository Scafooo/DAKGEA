#!/usr/bin/env bash
# ============================================================
#  DAKGEA Parallel Experiment Monitor
#  Real-time monitoring of parallel experiment execution
# ============================================================

set -euo pipefail

# ============================================================
#  ARGUMENT PARSING
# ============================================================
usage() {
    cat <<EOF
Usage: $0 [LOG_DIR] [OPTIONS]

Monitor a running parallel DAKGEA experiment execution.

ARGUMENTS:
    LOG_DIR                 Log directory to monitor (optional, will use latest if not provided)

OPTIONS:
    --refresh SECONDS       Refresh interval in seconds (default: 5)
    --help                  Show this help message

EXAMPLES:
    # Monitor latest run
    $0

    # Monitor specific run
    $0 results/logs/parallel_run_20231201_143022

    # Monitor with 2 second refresh
    $0 --refresh 2

EOF
    exit 0
}

LOG_DIR=""
REFRESH=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --refresh)
            REFRESH="$2"
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

# ============================================================
#  SETUP
# ============================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Find log directory
if [[ -z "$LOG_DIR" ]]; then
    # Find latest parallel run
    LATEST=$(find "${PROJECT_ROOT}/results/logs" -maxdepth 1 -type d -name "parallel_run_*" 2>/dev/null | sort -r | head -1)
    if [[ -z "$LATEST" ]]; then
        echo "Error: No parallel run logs found in ${PROJECT_ROOT}/results/logs"
        echo "Run a parallel experiment first or specify a log directory"
        exit 1
    fi
    LOG_DIR="$LATEST"
fi

# Validate log directory
if [[ ! -d "$LOG_DIR" ]]; then
    # Try relative to project root
    if [[ -d "${PROJECT_ROOT}/${LOG_DIR}" ]]; then
        LOG_DIR="${PROJECT_ROOT}/${LOG_DIR}"
    else
        echo "Error: Log directory not found: ${LOG_DIR}"
        exit 1
    fi
fi

JOBLOG_FILE="${LOG_DIR}/joblog.txt"

if [[ ! -f "$JOBLOG_FILE" ]]; then
    echo "Error: Job log not found: ${JOBLOG_FILE}"
    echo "This might not be a valid parallel run log directory"
    exit 1
fi

# ============================================================
#  HELPER FUNCTIONS
# ============================================================
term_width() { tput cols 2>/dev/null || echo 80; }
term_height() { tput lines 2>/dev/null || echo 24; }
full_line() { printf '%*s\n' "$(term_width)" '' | tr ' ' "$1"; }

format_time() {
    local seconds=$1
    printf '%02d:%02d:%02d' $((seconds/3600)) $((seconds%3600/60)) $((seconds%60))
}

get_gpu_info() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null || echo ""
    fi
}

# ============================================================
#  MONITORING LOOP
# ============================================================
echo "Monitoring: ${LOG_DIR}"
echo "Press Ctrl+C to exit"
echo ""

trap 'echo ""; echo "Monitoring stopped."; exit 0' INT TERM

while true; do
    clear

    # Header
    full_line '='
    printf "%*s\n" $((($(term_width) + 40) / 2)) "DAKGEA Parallel Experiment Monitor"
    full_line '='
    echo ""
    echo "Log directory: ${LOG_DIR}"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Parse joblog
    if [[ -f "$JOBLOG_FILE" ]] && [[ $(wc -l < "$JOBLOG_FILE") -gt 1 ]]; then
        TOTAL=$(tail -n +2 "$JOBLOG_FILE" | wc -l)
        COMPLETED=$(tail -n +2 "$JOBLOG_FILE" | awk '$7 != "" && $7 == 0' | wc -l)
        FAILED=$(tail -n +2 "$JOBLOG_FILE" | awk '$7 != "" && $7 != 0' | wc -l)
        RUNNING=$((TOTAL - COMPLETED - FAILED))

        # Calculate progress percentage
        if [[ $TOTAL -gt 0 ]]; then
            PERCENT=$((COMPLETED * 100 / TOTAL))
        else
            PERCENT=0
        fi

        # Time statistics
        FIRST_START=$(tail -n +2 "$JOBLOG_FILE" | awk '{print $4}' | sort -n | head -1)
        if [[ -n "$FIRST_START" && "$FIRST_START" != "" ]]; then
            NOW=$(date +%s)
            ELAPSED=$((NOW - ${FIRST_START%.*}))

            if [[ $COMPLETED -gt 0 ]]; then
                AVG_TIME=$((ELAPSED / COMPLETED))
                REMAINING=$((RUNNING * AVG_TIME + (TOTAL - COMPLETED - RUNNING) * AVG_TIME))
            else
                AVG_TIME=0
                REMAINING=0
            fi
        else
            ELAPSED=0
            AVG_TIME=0
            REMAINING=0
        fi

        # Progress bar
        BAR_WIDTH=50
        FILLED=$((PERCENT * BAR_WIDTH / 100))
        BAR=$(printf '%*s' "$FILLED" '' | tr ' ' '█')
        EMPTY=$(printf '%*s' "$((BAR_WIDTH - FILLED))" '' | tr ' ' '░')

        full_line '-'
        echo "Progress:"
        echo "  [${BAR}${EMPTY}] ${PERCENT}%"
        echo ""
        echo "  Total experiments : ${TOTAL}"
        echo "  Completed         : ${COMPLETED}"
        echo "  Running           : ${RUNNING}"
        echo "  Failed            : ${FAILED}"
        echo ""
        echo "Time:"
        echo "  Elapsed           : $(format_time $ELAPSED)"
        if [[ $AVG_TIME -gt 0 ]]; then
            echo "  Avg per experiment: $(format_time $AVG_TIME)"
            echo "  ETA               : $(format_time $REMAINING)"
        fi
        echo ""

        # Currently running jobs
        if [[ $RUNNING -gt 0 ]]; then
            full_line '-'
            echo "Currently running experiments:"
            tail -n +2 "$JOBLOG_FILE" | awk '$7 == "" {print "  - " $10}' | tail -10
            echo ""
        fi

        # Recent failures
        if [[ $FAILED -gt 0 ]]; then
            full_line '-'
            echo "Recently failed experiments (last 5):"
            tail -n +2 "$JOBLOG_FILE" | awk '$7 != "" && $7 != 0 {print "  - " $10 " (exit: " $7 ")"}' | tail -5
            echo ""
        fi
    else
        echo "Waiting for job log data..."
        echo ""
    fi

    # GPU information
    GPU_INFO=$(get_gpu_info)
    if [[ -n "$GPU_INFO" ]]; then
        full_line '-'
        echo "GPU Status:"
        echo ""
        echo "  ID  Name                        Usage  Memory          Temp"
        echo "$GPU_INFO" | while IFS=',' read -r id name util mem_used mem_total temp; do
            printf "  %-3s %-27s %3s%%   %5s/%5s MB  %3s°C\n" \
                "$id" "$name" "$util" "$mem_used" "$mem_total" "$temp"
        done
        echo ""
    fi

    # Recent log entries (if available)
    RECENT_LOGS=$(find "$LOG_DIR" -type f -name "stderr" -newermt "-30 seconds" 2>/dev/null | head -3)
    if [[ -n "$RECENT_LOGS" ]]; then
        full_line '-'
        echo "Recent errors (last 30 seconds):"
        echo "$RECENT_LOGS" | while read -r logfile; do
            expname=$(basename "$(dirname "$logfile")")
            if [[ -s "$logfile" ]]; then
                echo "  From: ${expname}"
                tail -3 "$logfile" | sed 's/^/    /'
            fi
        done | head -15
        echo ""
    fi

    full_line '='
    echo "Refreshing every ${REFRESH}s... (Ctrl+C to exit)"

    sleep "$REFRESH"
done
