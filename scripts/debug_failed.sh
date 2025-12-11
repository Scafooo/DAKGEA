#!/bin/bash
# Script per debuggare un esperimento fallito

echo "=== Finding latest parallel run ==="
LATEST_LOG=$(ls -td results/logs/parallel_run_* 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No parallel runs found!"
    exit 1
fi

echo "Log directory: $LATEST_LOG"
echo ""

echo "=== Failed experiments ==="
tail -n +2 "$LATEST_LOG/joblog.txt" | awk '$7 != "" && $7 != 0 {print $1, $10, "exit:", $7}'
echo ""

echo "Which experiment number do you want to see? (e.g., 1, 2, 3...)"
read -r EXP_NUM

if [ -f "$LATEST_LOG/$EXP_NUM/stderr" ]; then
    echo ""
    echo "=== STDERR for experiment $EXP_NUM ==="
    cat "$LATEST_LOG/$EXP_NUM/stderr"
    echo ""
fi

if [ -f "$LATEST_LOG/$EXP_NUM/stdout" ]; then
    echo ""
    echo "=== STDOUT for experiment $EXP_NUM ==="
    cat "$LATEST_LOG/$EXP_NUM/stdout"
fi
