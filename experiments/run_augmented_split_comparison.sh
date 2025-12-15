#!/bin/bash
# Compare augmented_only_train vs default split behavior

set -e

echo "=============================================================="
echo "Testing augmented_only_train option comparison"
echo "=============================================================="
echo ""

# Test 1: Default split (augmented entities distributed uniformly)
echo "Test 1: Running with DEFAULT split (augmented distributed across train/test/valid)..."
echo "--------------------------------------------------------------"
claude-code run config/experiments/test_augmented_default_split.yaml
echo ""

# Check split distribution
WORKSPACE1="results/experiments/test_augmented_default_split/workspace"
if [ -d "$WORKSPACE1/augmentation/dataset" ]; then
    echo "Default split results:"
    echo "  sup_pairs (train):  $(wc -l < $WORKSPACE1/augmentation/dataset/sup_pairs) pairs"
    echo "  ref_pairs (test):   $(wc -l < $WORKSPACE1/augmentation/dataset/ref_pairs) pairs"
    echo "  valid_pairs (valid): $(wc -l < $WORKSPACE1/augmentation/dataset/valid_pairs) pairs"
    echo ""
fi

echo ""
echo "Test 2: Running with AUGMENTED_ONLY_TRAIN (all augmented in training set)..."
echo "--------------------------------------------------------------"
claude-code run config/experiments/test_augmented_only_train.yaml
echo ""

# Check split distribution
WORKSPACE2="results/experiments/test_augmented_only_train/workspace"
if [ -d "$WORKSPACE2/augmentation/dataset" ]; then
    echo "Augmented-only-train results:"
    echo "  sup_pairs (train):  $(wc -l < $WORKSPACE2/augmentation/dataset/sup_pairs) pairs"
    echo "  ref_pairs (test):   $(wc -l < $WORKSPACE2/augmentation/dataset/ref_pairs) pairs"
    echo "  valid_pairs (valid): $(wc -l < $WORKSPACE2/augmentation/dataset/valid_pairs) pairs"
    echo ""
fi

echo ""
echo "=============================================================="
echo "Comparison Summary"
echo "=============================================================="
echo ""
echo "Key difference:"
echo "  - Default: Augmented entities distributed across all splits (train/test/valid)"
echo "  - Augmented-only-train: ALL augmented entities in training set only"
echo ""
echo "Expected improvement:"
echo "  - Training set size increases with augmented_only_train=true"
echo "  - Test/valid sets remain clean (only original entities)"
echo "  - Model can learn from augmented data without leaking into evaluation"
echo ""
