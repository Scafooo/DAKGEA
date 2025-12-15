#!/bin/bash
# Quick test: Run synthetic comparison on small subset
#
# This is a fast test script to verify the setup works.
# Uses small reduction ratio for quick execution.

set -e

echo "========================================="
echo "Quick Synthetic Comparison Test"
echo "========================================="
echo ""
echo "This will run a QUICK test with:"
echo "  - Dataset: D_W_15K_V1"
echo "  - Ratio: 0.1 (10% of data - FAST)"
echo "  - Experiments: all three modes"
echo ""
echo "For full experiments, use: scripts/run_synthetic_comparison.sh"
echo ""

# Run with small ratio for quick test
bash scripts/run_synthetic_comparison.sh \
    --dataset D_W_15K_V1 \
    --ratio 0.1 \
    --seed 42

echo ""
echo "========================================="
echo "Quick test completed!"
echo "========================================="
echo ""
echo "Results saved in: results/synthetic_comparison/"
echo ""
echo "To run full experiment (slower):"
echo "  bash scripts/run_synthetic_comparison.sh --ratio 0.5"
