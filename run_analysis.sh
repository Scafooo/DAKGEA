#!/bin/bash
# Dataset Analysis Runner
# Analyzes HybEA attribute_data format datasets to verify structural invariants

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
VERBOSE=""
OUTPUT=""
QUIET=""

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS] DATASET_PATH

Analyze HybEA attribute_data format datasets.

Arguments:
  DATASET_PATH          Path to attribute_data directory

Options:
  -o, --output FILE     Save results to JSON file
  -v, --verbose         Enable verbose logging (DEBUG level)
  -q, --quiet           Suppress all output except errors
  -h, --help            Show this help message

Examples:
  # Analyze BBC_DB dataset
  $0 data/raw/hybea/BBC_DB/attribute_data

  # Analyze with output to JSON
  $0 data/raw/hybea/BBC_DB/attribute_data -o analysis_results.json

  # Verbose mode
  $0 data/raw/hybea/BBC_DB/attribute_data --verbose

  # Analyze all datasets in a directory
  for dataset in data/raw/hybea/*/attribute_data; do
      $0 "\$dataset" -o "results/\$(basename \$(dirname \$dataset))_analysis.json"
  done

EOF
    exit 0
}

# Parse arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -q|--quiet)
            QUIET="--quiet"
            shift
            ;;
        -o|--output)
            OUTPUT="--output $2"
            shift 2
            ;;
        -*)
            echo -e "${RED}Error: Unknown option $1${NC}" >&2
            usage
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Restore positional parameters
set -- "${POSITIONAL_ARGS[@]}"

# Check if dataset path is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: DATASET_PATH is required${NC}" >&2
    usage
fi

DATASET_PATH="$1"

# Check if dataset path exists
if [ ! -d "$DATASET_PATH" ]; then
    echo -e "${RED}Error: Dataset path not found: $DATASET_PATH${NC}" >&2
    exit 1
fi

# Header
if [ -z "$QUIET" ]; then
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}   Dataset Analysis Tool${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}📂 Dataset:${NC} $DATASET_PATH"
    echo -e "${BLUE}───────────────────────────────────────────────────────────────────${NC}"
fi

# Check if virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        if [ -z "$QUIET" ]; then
            echo -e "${YELLOW}⚠️  Activating virtual environment...${NC}"
        fi
        source .venv/bin/activate
    else
        echo -e "${YELLOW}⚠️  Warning: No virtual environment detected${NC}"
    fi
fi

# Run analysis
python3 experiments/dataset_analysis/run.py "$DATASET_PATH" $VERBOSE $QUIET $OUTPUT

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    if [ -z "$QUIET" ]; then
        echo -e "\n${GREEN}✅ Analysis completed successfully!${NC}"
    fi
else
    echo -e "\n${RED}❌ Analysis failed with exit code $EXIT_CODE${NC}" >&2
    exit $EXIT_CODE
fi
