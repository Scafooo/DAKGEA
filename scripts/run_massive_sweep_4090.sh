#!/bin/bash

# Script per avviare lo Sweep Massivo su RTX 4090
# Gestisce automaticamente il PYTHONPATH e il log su file

# Ottieni la directory assoluta dove si trova questo script (presunto essere nella root del progetto)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "--------------------------------------------------------------------------------"
echo "  DAKGEA: STARTING MASSIVE MIX-UP SWEEP (OPTIMIZED FOR RTX 4090)"
echo "  Project Root: $SCRIPT_DIR"
echo "  Date: $(date)"
echo "--------------------------------------------------------------------------------"

# Verifica se il virtual environment esiste
if [ -d ".venv" ]; then
    PYTHON_EXEC="./.venv/bin/python"
    echo "Using virtual environment: .venv"
else
    PYTHON_EXEC="python"
    echo "Virtual environment not found, using system python"
fi

# Configurazione Ambiente
export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES=0  # Assicurati di usare la prima GPU (solitamente la 4090)

# Nome del file di log
LOG_FILE="massive_sweep_4090_$(date +%Y%m%d_%H%M%S).log"

echo "Running massive sweep..."
echo "Log will be saved to: $LOG_FILE"
echo ""

# Esecuzione script con output in tempo reale e salvataggio su log
$PYTHON_EXEC -u tests/massive_mixup_sweep.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "--------------------------------------------------------------------------------"
echo "  SWEEP COMPLETED."
echo "  Check $LOG_FILE for the detailed report and suggested parameters."
echo "--------------------------------------------------------------------------------"
