#!/bin/bash
echo "⏳ Monitoring hyperparameter tuning..."
echo "   Log file: experiments/tuning_output.log"
echo ""

while true; do
    if [ ! -f experiments/tuning_output.log ]; then
        echo "[$(date +%H:%M:%S)] ❌ Log file not found"
        sleep 30
        continue
    fi
    
    # Check if completed
    if grep -q "RECOMMENDED CONFIGURATION" experiments/tuning_output.log; then
        echo ""
        echo "✅ TUNING COMPLETED!"
        echo ""
        grep -A 15 "RECOMMENDED CONFIGURATION" experiments/tuning_output.log
        break
    fi
    
    # Check for errors
    if grep -q "Traceback\|Error:" experiments/tuning_output.log | tail -1; then
        echo "[$(date +%H:%M:%S)] ⚠️  Errors detected, check log"
    fi
    
    # Check progress
    collected=$(grep "Collected" experiments/tuning_output.log | tail -1)
    testing=$(grep -o "\[[0-9]*/72\]" experiments/tuning_output.log | tail -1)
    
    if [ -n "$collected" ]; then
        echo "[$(date +%H:%M:%S)] $collected"
    fi
    
    if [ -n "$testing" ]; then
        echo -ne "\r[$(date +%H:%M:%S)] Testing configuration $testing    "
    fi
    
    sleep 15
done
