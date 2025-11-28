#!/usr/bin/env bash
# Quality analysis script for DAKGEA augmented datasets
#
# Usage:
#   bash scripts/analyze_quality.sh <original_dataset> <augmented_dataset> [output_dir]
#
# Example:
#   bash scripts/analyze_quality.sh \
#       data/raw/openea/BBC_DB \
#       results/BBC_DB_01_05/augmentation \
#       results/quality_analysis

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <original_dataset> <augmented_dataset> [output_dir]"
    echo ""
    echo "Example:"
    echo "  $0 data/raw/openea/BBC_DB results/BBC_DB_01_05/augmentation"
    exit 1
fi

ORIGINAL_PATH="$1"
AUGMENTED_PATH="$2"
OUTPUT_DIR="${3:-results/quality_analysis}"

# Check paths exist
if [ ! -d "$ORIGINAL_PATH" ]; then
    echo "ERROR: Original dataset not found: $ORIGINAL_PATH"
    exit 1
fi

if [ ! -d "$AUGMENTED_PATH" ]; then
    echo "ERROR: Augmented dataset not found: $AUGMENTED_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "================================================================================"
echo "                    DAKGEA Quality Analysis"
echo "================================================================================"
echo "Original Dataset : $ORIGINAL_PATH"
echo "Augmented Dataset: $AUGMENTED_PATH"
echo "Output Directory : $OUTPUT_DIR"
echo "================================================================================"
echo ""

# Run quality report generation
echo "🔍 Generating comprehensive quality report..."
python -m experiments.qualitative_analysis.quality_report \
    --original "$ORIGINAL_PATH" \
    --augmented "$AUGMENTED_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --stage augmentation \
    --n-samples 50

echo ""
echo "================================================================================"
echo "✅ Analysis complete!"
echo "================================================================================"
echo ""
echo "📂 Output files:"
echo "  - $OUTPUT_DIR/quality_report.md     (Human-readable report)"
echo "  - $OUTPUT_DIR/quality_report.json   (Machine-readable metrics)"
echo "  - $OUTPUT_DIR/samples/random_samples.tsv   (For human evaluation)"
echo "  - $OUTPUT_DIR/samples/diverse_samples.tsv  (For human evaluation)"
echo ""
echo "📊 Next steps:"
echo "  1. Read the report: less $OUTPUT_DIR/quality_report.md"
echo "  2. Review samples: open $OUTPUT_DIR/samples/random_samples.tsv"
echo "  3. Annotate realism_score (1-5) and consistency_score (1-5)"
echo "  4. Use metrics to tune augmentation parameters"
echo ""
echo "================================================================================"
