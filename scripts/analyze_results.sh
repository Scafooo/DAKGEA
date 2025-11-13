#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

python "$PROJECT_ROOT/experiments/statistics/analyze_results.py" "$@"
