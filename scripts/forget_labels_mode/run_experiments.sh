#!/usr/bin/env bash
# ============================================================
#  Run Forget Labels Experiments
#  Generates configs and runs them in parallel
# ============================================================

set -euo pipefail

# ============================================================
#  CONFIGURATION
# ============================================================
default_jobs=2
default_gpu_id=0
timeout=72000000 # 2 hours

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
project_root="$( cd "${script_dir}/../.." && pwd )"
export project_root
export pythonpath="${project_root}"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs) 
            default_jobs="$2"
            shift 2
            ;;
        --gpu-id) 
            default_gpu_id="$2"
            shift 2
            ;; 
        *) 
            echo "Unknown option: $1"
            exit 1
            ;; 
    esac
done

export gpu_id="${default_gpu_id}"

# 1. Generate Configurations
echo "============================================================"
echo " Generating Configurations"
echo "============================================================"
python "${SCRIPT_DIR}/generate_configs.py"

CONFIG_DIR="${PROJECT_ROOT}/config/experiments/massive/forget_labels"

# Find all generated configs
mapfile -t config_files < <(find "${config_dir}" -name "*.yaml" | sort)

if [[ ${#config_files[@]} -eq 0 ]]; then
    echo "No config files generated!"
    exit 1
fi

echo "Found ${#config_files[@]} configurations."

# 2. Run Experiments in Parallel
echo ""
echo "============================================================"
echo " Running Experiments (Jobs: ${default_jobs}, GPU: ${default_gpu_id})
"
echo "============================================================"

# Check for parallel
if command -v parallel &> /dev/null; then
    parallel_bin="parallel"
elif [ -f "${project_root}/.local/bin/parallel" ]; then
    parallel_bin="${project_root}/.local/bin/parallel"
else
    echo "GNU Parallel not found. Running sequentially."
    parallel_bin=""
fi

log_dir="${project_root}/results/logs/forget_labels_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${log_dir}"
joblog="${log_dir}/joblog.txt"

if [[ -n "${parallel_bin}" ]]; then
    "${parallel_bin}" --will-cite --jobs "${default_jobs}" \
        --joblog "${joblog}" --timeout "${timeout}" --progress \
        --results "${log_dir}" \
        bash "${script_dir}/_run_single_experiment.sh" ::: "${config_files[@]}"
else
    for config in "${config_files[@]}"; do
        echo "Running ${config}..."
        bash "${script_dir}/_run_single_experiment.sh" "${config}"
    done
fi

echo ""
echo "============================================================"
echo " Done!"
echo " Logs: ${log_dir}"
echo "============================================================"
