#!/usr/bin/env bash
# ============================================================
#  DAKGEA Statistics Aggregator
#  Summarises reduction vs augmentation metrics across runs
# ============================================================
set -euo pipefail

term_width() { tput cols 2>/dev/null || echo 80; }
full_line() { printf '%*s\n' "$(term_width)" '' | tr ' ' "$1"; }

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

mapfile -t _STAT_PATHS < <(python <<'PY'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config.loader import PROJECT_ROOT, load_yaml
cfg = load_yaml(PROJECT_ROOT / "config/global.yaml") or {}
paths = cfg.get("paths", {})
print((PROJECT_ROOT / paths.get("results", "results")).resolve())
print((PROJECT_ROOT / paths.get("statistics", "results_analysis")).resolve())
PY
)
RESULTS_ROOT="${_STAT_PATHS[0]}"
DEFAULT_STATS_DIR="${_STAT_PATHS[1]}"

clear
full_line '='
printf "%*s\n" $((($(term_width) + 28) / 2)) "DAKGEA Statistics Aggregator"
full_line '='
echo "📂 Project root : ${PROJECT_ROOT}"
echo "📁 Results root : ${RESULTS_ROOT}"
echo "📊 Stats dir    : ${DEFAULT_STATS_DIR}"
echo "🕓 Started at   : $(date '+%Y-%m-%d %H:%M:%S %Z')"
full_line '='

python "$PROJECT_ROOT/experiments/statistics/analyze_results.py" "$@" --results-root "$RESULTS_ROOT"
