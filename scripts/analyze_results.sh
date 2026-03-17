#!/usr/bin/env bash
# ============================================================
#  DAKGEA Statistics Aggregator
#  Summarises reduction vs augmentation metrics across runs
# ============================================================
set -euo pipefail

# ============================================================
#  SUITE SELECTION
#  Modify this line to select a default suite to analyze
#  Can be overridden by --suite command line argument
# ============================================================
DEFAULT_SUITE="flant5_supp_visual/flant5_supp"

term_width() { tput cols 2>/dev/null || echo 80; }
full_line() { printf '%*s\n' "$(term_width)" '' | tr ' ' "$1"; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

mapfile -t _STAT_PATHS < <(python <<PY
from pathlib import Path
import sys
ROOT = Path("$PROJECT_ROOT")
sys.path.insert(0, str(ROOT))
from src.config.loader import PROJECT_ROOT, load_yaml
cfg = load_yaml(PROJECT_ROOT / "config/global.yaml") or {}
paths = cfg.get("paths", {})
print((PROJECT_ROOT / paths.get("results", "results")).resolve())
print((PROJECT_ROOT / paths.get("statistics", "results_analysis")).resolve())
PY
)
RESULTS_ROOT="${_STAT_PATHS[0]}"
DEFAULT_STATS_DIR="${_STAT_PATHS[1]}"

# Default options (can be overridden by command-line arguments)
# All advanced features are enabled by default for comprehensive analysis
ENABLE_ADVANCED_PLOTS=true
ENABLE_ADVANCED_STATS=true
EXPORT_FORMATS="tsv latex"  # TSV and LaTeX exports by default
SUITE="$DEFAULT_SUITE"

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-advanced-plots)
            ENABLE_ADVANCED_PLOTS=false
            shift
            ;;
        --no-advanced-stats)
            ENABLE_ADVANCED_STATS=false
            shift
            ;;
        --export-formats)
            EXPORT_FORMATS="$2"
            shift 2
            ;;
        --suite)
            SUITE="$2"
            shift 2
            ;;
        --basic)
            # Basic mode: no advanced features, TSV only
            ENABLE_ADVANCED_PLOTS=false
            ENABLE_ADVANCED_STATS=false
            EXPORT_FORMATS="tsv"
            shift
            ;;
        --full)
            # Full mode: all advanced features including LaTeX documents
            ENABLE_ADVANCED_PLOTS=true
            ENABLE_ADVANCED_STATS=true
            EXPORT_FORMATS="tsv latex latex-doc"
            shift
            ;;
        *)
            # Store other arguments to pass through
            break
            ;;
    esac
done

clear
full_line '='
printf "%*s\n" $((($(term_width) + 28) / 2)) "DAKGEA Statistics Aggregator"
full_line '='
echo "📂 Project root : ${PROJECT_ROOT}"
echo "📁 Results root : ${RESULTS_ROOT}"
echo "📊 Stats dir    : ${DEFAULT_STATS_DIR}"
echo "📈 Advanced viz : $([ "$ENABLE_ADVANCED_PLOTS" = true ] && echo "✓ enabled" || echo "✗ disabled")"
echo "📉 Advanced stats: $([ "$ENABLE_ADVANCED_STATS" = true ] && echo "✓ enabled" || echo "✗ disabled")"
echo "💾 Export formats: ${EXPORT_FORMATS}"
echo "🕓 Started at   : $(date '+%Y-%m-%d %H:%M:%S %Z')"
full_line '='

TARGET_PATH=""
if [[ -n "$SUITE" ]]; then
    # Suite specified: analyze results/suite_name/
    TARGET_PATH="${RESULTS_ROOT}/${SUITE}"
    if [[ ! -d "$TARGET_PATH" ]]; then
        echo "❌ Error: Suite directory not found: ${TARGET_PATH}"
        exit 1
    fi
    echo "🎯 Suite        : ${SUITE}"
    echo "🎯 Target path  : ${TARGET_PATH}"
    full_line '='
elif [[ $# -gt 0 && -e "$1" ]]; then
    TARGET_PATH="$1"
    shift
    echo "🎯 Target path  : ${TARGET_PATH}"
    full_line '='
fi

# Build command with options
CMD_ARGS=(
    ${TARGET_PATH:+ "$TARGET_PATH"}
    --results-root "$RESULTS_ROOT"
    --plots-dir "$DEFAULT_STATS_DIR"
    --tsv-dir "$DEFAULT_STATS_DIR"
    --export-formats $EXPORT_FORMATS
)

if [ "$ENABLE_ADVANCED_PLOTS" = true ]; then
    CMD_ARGS+=(--enable-advanced-plots)
fi

if [ "$ENABLE_ADVANCED_STATS" = true ]; then
    CMD_ARGS+=(--advanced-stats)
fi

# Add any remaining arguments
CMD_ARGS+=("$@")

python "$PROJECT_ROOT/experiments/statistics/analyze_results.py" "${CMD_ARGS[@]}"

# ============================================================
#  TIMING ANALYSIS
#  Compute execution times for reduction & augmentation stages
# ============================================================
echo ""
full_line '-'
printf "%*s\n" $((($(term_width) + 24) / 2)) "⏱️  Timing Analysis"
full_line '-'

TIMING_ARGS=(
    ${TARGET_PATH:+ "$TARGET_PATH"}
    --all-exports "${DEFAULT_STATS_DIR}/timing"
)

echo "📊 Computing execution times..."
python "$PROJECT_ROOT/experiments/statistics/timing_analysis.py" "${TIMING_ARGS[@]}"

echo ""
full_line '='
echo "✅ Statistics analysis completed!"
echo "📊 Output directory: ${DEFAULT_STATS_DIR}"
echo "⏱️  Timing reports  : ${DEFAULT_STATS_DIR}/timing/"
full_line '='
