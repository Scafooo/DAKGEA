#!/usr/bin/env bash
# Test script to verify parallel is working correctly

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Check if GNU Parallel is installed (local first, then system)
PARALLEL_BIN=""
if [ -f "${PROJECT_ROOT}/.local/bin/parallel" ]; then
    PARALLEL_BIN="${PROJECT_ROOT}/.local/bin/parallel"
    echo "✓ Found local GNU Parallel: ${PARALLEL_BIN}"
elif command -v parallel &> /dev/null; then
    PARALLEL_BIN="parallel"
    echo "✓ Found system GNU Parallel"
else
    echo "✗ GNU Parallel not found"
    exit 1
fi

# Test 1: Version check
echo ""
echo "Test 1: Version check"
"${PARALLEL_BIN}" --version | head -1

# Test 2: Simple echo test
echo ""
echo "Test 2: Simple command test"
echo -e "test1\ntest2\ntest3" | "${PARALLEL_BIN}" --will-cite --jobs 2 echo "Processing: {}"

# Test 3: Test with actual Python command (dry-run style)
echo ""
echo "Test 3: Python command test (dry)"
echo -e "file1.yaml\nfile2.yaml" | "${PARALLEL_BIN}" --will-cite --jobs 2 --dry-run \
    "python -c 'print(\"Processing: {}\")'"

# Test 4: Test with variables
echo ""
echo "Test 4: Variable expansion test"
TEST_VAR="test_value"
export TEST_VAR
echo -e "item1\nitem2" | "${PARALLEL_BIN}" --will-cite --jobs 2 \
    "echo 'Item: {} - Var: ${TEST_VAR}'"

echo ""
echo "✓ All tests completed successfully!"
