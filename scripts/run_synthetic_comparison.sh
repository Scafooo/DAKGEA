#!/bin/bash
# Run synthetic vs real data comparison experiments
#
# Usage:
#   bash scripts/run_synthetic_comparison.sh [OPTIONS]
#
# Options:
#   --dataset NAME       Dataset to use (default: D_W_15K_V1)
#   --ratio RATIO        Reduction ratio (default: 0.5)
#   --seed SEED          Random seed (default: 42)
#   --dry-run            Print commands without executing
#   --parallel           Run experiments in parallel (faster)
#   --experiments LIST   Comma-separated list of experiments (default: baseline,synthetic_only,augmented)

set -e  # Exit on error

# Default parameters
DATASET="D_W_15K_V1"
RATIO="0.5"
SEED="42"
DRY_RUN=false
PARALLEL=false
EXPERIMENTS="baseline,synthetic_only,augmented"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --ratio)
            RATIO="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --experiments)
            EXPERIMENTS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset NAME       Dataset to use (default: D_W_15K_V1)"
            echo "  --ratio RATIO        Reduction ratio (default: 0.5)"
            echo "  --seed SEED          Random seed (default: 42)"
            echo "  --dry-run            Print commands without executing"
            echo "  --parallel           Run experiments in parallel"
            echo "  --experiments LIST   Experiments to run (default: baseline,synthetic_only,augmented)"
            echo "  -h, --help           Show this help"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --dataset D_W_15K_V2 --ratio 0.3"
            echo "  $0 --experiments baseline,synthetic_only --parallel"
            echo "  $0 --dry-run"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Synthetic vs Real Data Comparison${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Dataset:     ${GREEN}${DATASET}${NC}"
echo -e "Ratio:       ${GREEN}${RATIO}${NC}"
echo -e "Seed:        ${GREEN}${SEED}${NC}"
echo -e "Experiments: ${GREEN}${EXPERIMENTS}${NC}"
echo -e "Parallel:    ${GREEN}${PARALLEL}${NC}"
echo -e "Dry run:     ${GREEN}${DRY_RUN}${NC}"
echo ""

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Check if Python environment is activated
if [[ -z "${VIRTUAL_ENV}" ]] && [[ ! -f ".venv/bin/activate" ]]; then
    echo -e "${YELLOW}Warning: No virtual environment detected${NC}"
    echo -e "${YELLOW}Consider activating .venv: source .venv/bin/activate${NC}"
    echo ""
fi

# Ensure results directory exists
RESULTS_DIR="results/synthetic_comparison"
mkdir -p "$RESULTS_DIR"/{baseline,synthetic_only,augmented}

# Function to run single experiment
run_experiment() {
    local exp_name=$1
    local config_file="config/experiments/synthetic_comparison/${exp_name}.yaml"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running: ${exp_name}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "Config: ${config_file}"
    echo -e "Output: ${RESULTS_DIR}/${exp_name}/"
    echo ""

    # Build command
    # TODO: Replace with your actual training command
    # This is a placeholder - adjust to your pipeline
    cmd="python -m src.main --config ${config_file}"

    # Override parameters from command line
    cmd="${cmd} --dataset ${DATASET}"
    cmd="${cmd} --reduction-ratio ${RATIO}"
    cmd="${cmd} --seed ${SEED}"
    cmd="${cmd} --output ${RESULTS_DIR}/${exp_name}/"

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}[DRY RUN] Would execute:${NC}"
        echo -e "  ${cmd}"
        echo ""
        return 0
    fi

    # Run command
    echo -e "${GREEN}Executing:${NC} ${cmd}"
    echo ""

    if eval "$cmd"; then
        echo -e "${GREEN}✓ ${exp_name} completed successfully${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ ${exp_name} failed!${NC}"
        echo ""
        return 1
    fi
}

# Export function for parallel execution
export -f run_experiment
export DATASET RATIO SEED RESULTS_DIR DRY_RUN
export RED GREEN YELLOW BLUE NC

# Parse experiments list
IFS=',' read -ra EXP_ARRAY <<< "$EXPERIMENTS"

# Run experiments
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Experiments${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

FAILED=0

if [[ "$PARALLEL" == true ]]; then
    echo -e "${YELLOW}Running experiments in PARALLEL${NC}"
    echo ""

    # Run in parallel using background jobs
    pids=()
    for exp in "${EXP_ARRAY[@]}"; do
        run_experiment "$exp" &
        pids+=($!)
    done

    # Wait for all to complete
    for pid in "${pids[@]}"; do
        if ! wait $pid; then
            FAILED=$((FAILED + 1))
        fi
    done
else
    echo -e "${YELLOW}Running experiments SEQUENTIALLY${NC}"
    echo ""

    # Run sequentially
    for exp in "${EXP_ARRAY[@]}"; do
        if ! run_experiment "$exp"; then
            FAILED=$((FAILED + 1))
        fi
    done
fi

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Experiment Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

for exp in "${EXP_ARRAY[@]}"; do
    results_file="${RESULTS_DIR}/${exp}/results.json"
    if [[ -f "$results_file" ]]; then
        echo -e "${GREEN}✓ ${exp}${NC}"
        # Extract key metric (adjust based on your output format)
        if command -v jq &> /dev/null; then
            hits1=$(jq -r '.hits_at_1 // "N/A"' "$results_file" 2>/dev/null || echo "N/A")
            echo -e "    Hits@1: ${hits1}"
        fi
    else
        echo -e "${RED}✗ ${exp} - No results found${NC}"
    fi
done

echo ""

# Compare results
if [[ "$DRY_RUN" == false ]] && [[ "$FAILED" -eq 0 ]]; then
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Running Comparison Analysis${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # TODO: Add comparison script
    # python experiments/compare_results.py \
    #     --baseline "${RESULTS_DIR}/baseline/results.json" \
    #     --synthetic "${RESULTS_DIR}/synthetic_only/results.json" \
    #     --augmented "${RESULTS_DIR}/augmented/results.json"

    echo -e "${YELLOW}Note: Implement comparison script for detailed analysis${NC}"
    echo ""
fi

# Exit status
if [[ "$FAILED" -gt 0 ]]; then
    echo -e "${RED}${FAILED} experiment(s) failed${NC}"
    exit 1
else
    echo -e "${GREEN}All experiments completed successfully!${NC}"
    exit 0
fi
