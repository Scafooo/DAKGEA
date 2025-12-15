#!/usr/bin/env bash
# ============================================================
#  Run Synthetic Data Quality Evaluation
#  Compares Baseline (real only) vs Synthetic-only (synthetic only)
# ============================================================

set -euo pipefail

# ============================================================
#  CONFIGURATION
# ============================================================
DEFAULT_MODEL="bert_int"  # bert_int or rrea
DEFAULT_JOBS=2
DEFAULT_TIMEOUT=7200
DEFAULT_GPU_ID=0
RUN_BASELINE=true
RUN_SYNTHETIC=true
DRY_RUN=false
PATTERN="*.yaml"
RESUME=false
FAIR_COMPARISON=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================
#  USAGE
# ============================================================
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Run synthetic data quality evaluation experiments.

This script runs two modes of experiments to evaluate synthetic data quality:
  Mode 1 (Baseline):       Only original/real aligned pairs (no augmentation)
  Mode 2 (Synthetic-only): ONLY synthetic/generated pairs (no real data)

The comparison evaluates:
  - Quality Gap = Performance(Baseline) - Performance(Synthetic-only)
  - Transferability = Performance(Synthetic-only) / Performance(Baseline)

Research Questions:
  Q1: Can synthetic data replace real data?
  Q2: What is the quality gap between synthetic and real data?
  Q3: How does quality scale with augmentation ratio?

OPTIONS:
    --model MODEL           Model type: bert_int or rrea (default: ${DEFAULT_MODEL})
    --jobs N                Number of parallel jobs (default: ${DEFAULT_JOBS})
    --timeout SECONDS       Timeout per job (default: ${DEFAULT_TIMEOUT})
    --gpu-id ID             GPU device ID (default: ${DEFAULT_GPU_ID})
    --baseline-only         Run only baseline experiments
    --synthetic-only        Run only synthetic_only experiments
    --pattern PATTERN       YAML file pattern (default: "*.yaml")
    --fair-comparison       Run only configs with aug_ratio=1.0 (same N for fair quality comparison)
    --resume                Resume interrupted run
    --dry-run               Show what would be executed
    --help, -h              Show this help

EXAMPLES:
    # Run all experiments (baseline + synthetic_only) for bert_int
    $0 --model bert_int --jobs 4

    # Run FAIR COMPARISON (aug_ratio=1.0 only, same N for quality evaluation)
    $0 --model bert_int --fair-comparison --jobs 4

    # Run only baseline experiments
    $0 --model bert_int --baseline-only --jobs 4

    # Run only synthetic_only experiments
    $0 --model bert_int --synthetic-only --jobs 4

    # Dry run to see what will be executed
    $0 --dry-run

    # Resume interrupted run
    $0 --resume

    # Run specific pattern (e.g., one dataset)
    $0 --pattern "D_W_15K_V1_*.yaml" --jobs 4

EXPERIMENT DIRECTORIES:
    Baseline:       config/experiments/massive/{model}_baseline/
    Synthetic-only: config/experiments/massive/{model}_synthetic_only/

RESULTS:
    Results will be saved in results/ directory with experiment-specific names.

    To analyze results after completion, use:
        python experiments/statistics/compare_quality.py --model {model}

INTERPRETATION:
    Quality Gap < 5%:   EXCELLENT - Synthetic data can replace real data
    Quality Gap 5-15%:  GOOD - Synthetic data maintains general patterns
    Quality Gap > 15%:  POOR - Significant quality issues in synthetic data

EOF
    exit 0
}

# ============================================================
#  PARSE ARGUMENTS
# ============================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            DEFAULT_MODEL="$2"
            shift 2
            ;;
        --jobs)
            DEFAULT_JOBS="$2"
            shift 2
            ;;
        --timeout)
            DEFAULT_TIMEOUT="$2"
            shift 2
            ;;
        --gpu-id)
            DEFAULT_GPU_ID="$2"
            shift 2
            ;;
        --baseline-only)
            RUN_BASELINE=true
            RUN_SYNTHETIC=false
            shift
            ;;
        --synthetic-only)
            RUN_BASELINE=false
            RUN_SYNTHETIC=true
            shift
            ;;
        --pattern)
            PATTERN="$2"
            shift 2
            ;;
        --fair-comparison)
            FAIR_COMPARISON=true
            PATTERN="*_*_10.yaml"  # Only aug_ratio=1.0 configs
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate model
if [[ "$DEFAULT_MODEL" != "bert_int" && "$DEFAULT_MODEL" != "rrea" ]]; then
    echo -e "${RED}Error: Invalid model '${DEFAULT_MODEL}'. Must be 'bert_int' or 'rrea'.${NC}"
    exit 1
fi

# ============================================================
#  SETUP
# ============================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

BASELINE_DIR="config/experiments/massive/${DEFAULT_MODEL}_baseline"
SYNTHETIC_DIR="config/experiments/massive/${DEFAULT_MODEL}_synthetic_only"

# ============================================================
#  BANNER
# ============================================================
term_width() { tput cols 2>/dev/null || echo 80; }
full_line() { printf '%*s\n' "$(term_width)" '' | tr ' ' "$1"; }

clear
full_line '='
printf "%*s\n" $((($(term_width) + 55) / 2)) "DAKGEA: Synthetic Data Quality Evaluation"
full_line '='
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Model             : ${DEFAULT_MODEL}"
echo "  Parallel jobs     : ${DEFAULT_JOBS}"
echo "  Timeout per job   : ${DEFAULT_TIMEOUT}s"
echo "  GPU device        : ${DEFAULT_GPU_ID}"
echo "  Pattern           : ${PATTERN}"
echo "  Fair comparison   : ${FAIR_COMPARISON}"
echo ""
echo -e "${BLUE}Experiments to run:${NC}"
echo "  Baseline (Mode 1)      : ${RUN_BASELINE} (${BASELINE_DIR})"
echo "  Synthetic-only (Mode 2): ${RUN_SYNTHETIC} (${SYNTHETIC_DIR})"
echo ""
if [[ "$FAIR_COMPARISON" == "true" ]]; then
    echo -e "${YELLOW}Fair Comparison Mode:${NC}"
    echo "  Using only aug_ratio=1.0 configs (*_*_10.yaml)"
    echo "  This ensures same N for baseline vs synthetic-only"
    echo "  → Fair quality comparison (not affected by dataset size)"
    echo ""
fi

# Count files
if [[ "$RUN_BASELINE" == "true" ]]; then
    BASELINE_COUNT=$(find "${PROJECT_ROOT}/${BASELINE_DIR}" -name "${PATTERN}" 2>/dev/null | wc -l || echo 0)
    echo "  Baseline configs       : ${BASELINE_COUNT}"
fi

if [[ "$RUN_SYNTHETIC" == "true" ]]; then
    SYNTHETIC_COUNT=$(find "${PROJECT_ROOT}/${SYNTHETIC_DIR}" -name "${PATTERN}" 2>/dev/null | wc -l || echo 0)
    echo "  Synthetic-only configs : ${SYNTHETIC_COUNT}"
fi

echo ""
echo -e "${BLUE}Research Questions:${NC}"
echo "  Q1: Can synthetic data replace real data?"
echo "  Q2: What is the quality gap?"
echo "  Q3: How does quality scale with aug ratio?"
echo ""
full_line '-'

# ============================================================
#  DRY RUN
# ============================================================
if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}DRY RUN MODE${NC}"
    echo ""

    if [[ "$RUN_BASELINE" == "true" ]]; then
        echo -e "${BLUE}Would run BASELINE experiments:${NC}"
        echo "  bash scripts/run_experiments_parallel.sh \\"
        echo "    --dir ${BASELINE_DIR} \\"
        echo "    --jobs ${DEFAULT_JOBS} \\"
        echo "    --timeout ${DEFAULT_TIMEOUT} \\"
        echo "    --gpu-id ${DEFAULT_GPU_ID} \\"
        echo "    --pattern '${PATTERN}'"
        echo ""
    fi

    if [[ "$RUN_SYNTHETIC" == "true" ]]; then
        echo -e "${BLUE}Would run SYNTHETIC-ONLY experiments:${NC}"
        echo "  bash scripts/run_experiments_parallel.sh \\"
        echo "    --dir ${SYNTHETIC_DIR} \\"
        echo "    --jobs ${DEFAULT_JOBS} \\"
        echo "    --timeout ${DEFAULT_TIMEOUT} \\"
        echo "    --gpu-id ${DEFAULT_GPU_ID} \\"
        echo "    --pattern '${PATTERN}'"
        echo ""
    fi

    if [[ "$FAIR_COMPARISON" == "true" ]]; then
        echo -e "${YELLOW}Fair Comparison Mode Active:${NC}"
        echo "  Only running configs with aug_ratio=1.0 (*_*_10.yaml)"
        echo "  This ensures:"
        echo "    - Baseline: N real pairs"
        echo "    - Synthetic-only: N synthetic pairs (same N!)"
        echo "    - Fair quality comparison (not biased by dataset size)"
        echo ""
    fi

    echo "After completion, analyze with:"
    echo "  python experiments/statistics/compare_quality.py --model ${DEFAULT_MODEL}"
    echo ""
    echo "Expected output:"
    echo "  - Quality Gap = Performance(Baseline) - Performance(Synthetic)"
    echo "  - Transferability Score = Synthetic / Baseline"
    echo "  - Per-dataset quality analysis"
    echo ""
    exit 0
fi

# ============================================================
#  CONFIRMATION
# ============================================================
TOTAL_EXPERIMENTS=0
[[ "$RUN_BASELINE" == "true" ]] && TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + BASELINE_COUNT))
[[ "$RUN_SYNTHETIC" == "true" ]] && TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + SYNTHETIC_COUNT))

echo ""
read -r -p "Run ${TOTAL_EXPERIMENTS} experiments in total? [y/N] " CONFIRM
case "${CONFIRM,,}" in
    y|yes)
        ;;
    *)
        echo -e "${YELLOW}Aborted by user.${NC}"
        exit 0
        ;;
esac

# ============================================================
#  RUN EXPERIMENTS
# ============================================================
echo ""
full_line '='
echo "Starting experiments..."
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
full_line '='
echo ""

FAILED=0
START_TIME=$(date +%s)

# Build common args
PARALLEL_ARGS="--jobs ${DEFAULT_JOBS} --timeout ${DEFAULT_TIMEOUT} --gpu-id ${DEFAULT_GPU_ID} --pattern ${PATTERN}"
[[ "$RESUME" == "true" ]] && PARALLEL_ARGS="${PARALLEL_ARGS} --resume"

# Run baseline
if [[ "$RUN_BASELINE" == "true" ]]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running BASELINE experiments (Real data only)${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    if bash scripts/run_experiments_parallel.sh \
        --dir "${BASELINE_DIR}" \
        ${PARALLEL_ARGS}; then
        echo -e "${GREEN}✓ Baseline experiments completed${NC}"
    else
        echo -e "${RED}✗ Baseline experiments failed${NC}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
fi

# Run synthetic-only
if [[ "$RUN_SYNTHETIC" == "true" ]]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running SYNTHETIC-ONLY experiments (Generated data only)${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    if bash scripts/run_experiments_parallel.sh \
        --dir "${SYNTHETIC_DIR}" \
        ${PARALLEL_ARGS}; then
        echo -e "${GREEN}✓ Synthetic-only experiments completed${NC}"
    else
        echo -e "${RED}✗ Synthetic-only experiments failed${NC}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ============================================================
#  SUMMARY
# ============================================================
echo ""
full_line '='
echo "All experiments completed!"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total time: ${ELAPSED}s ($(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))))"
full_line '='
echo ""

if [[ ${FAILED} -gt 0 ]]; then
    echo -e "${RED}${FAILED} experiment set(s) had failures${NC}"
    echo "Check logs in results/logs/ for details"
else
    echo -e "${GREEN}All experiments completed successfully!${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Next Steps: Quality Analysis${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "To evaluate synthetic data quality:"
echo "  python experiments/statistics/compare_quality.py --model ${DEFAULT_MODEL}"
echo ""
echo "This will compute:"
echo "  1. Quality Gap = Baseline - Synthetic-only"
echo "  2. Transferability Score = Synthetic-only / Baseline"
echo "  3. Per-dataset quality breakdown"
echo ""
echo "Interpretation:"
echo "  Quality Gap < 5%:  EXCELLENT - Synthetic can replace real"
echo "  Quality Gap 5-15%: GOOD - Synthetic maintains patterns"
echo "  Quality Gap > 15%: POOR - Quality issues"
echo ""

full_line '='

exit ${FAILED}
