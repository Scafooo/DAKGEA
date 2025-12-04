
import argparse
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

def run_experiment(config_file, gpu_id, project_root, retry, timeout):
    """
    Runs a single experiment and retries on failure.
    """
    for i in range(retry):
        try:
            env = os.environ.copy()
            env["PROJECT_ROOT"] = str(project_root)
            env["PYTHONPATH"] = str(project_root)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["GPU_ID"] = str(gpu_id)

            script_path = project_root / "scripts" / "_run_single_experiment.sh"
            
            process = subprocess.run(
                ["bash", str(script_path), str(config_file)],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if process.returncode == 0:
                print(f"Successfully finished {config_file}")
                return 0
            else:
                print(f"Failed to run {config_file}. Stderr: {process.stderr}")

        except subprocess.TimeoutExpired:
            print(f"Timeout for {config_file} expired.")
        except Exception as e:
            print(f"An exception occurred while running {config_file}: {e}")

    return 1


def main():
    """
    Main function to run experiments in parallel.
    """
    parser = argparse.ArgumentParser(
        description="Run multiple DAKGEA experiments in parallel."
    )
    parser.add_argument("--dir", required=True, help="Directory containing YAML experiment configs")
    parser.add_argument("--jobs", type=int, default=4, help="Number of parallel jobs")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--retry", type=int, default=1, help="Number of retries for failed jobs")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout per job in seconds")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.resolve()
    
    # Resolve experiment directory
    search_paths = [
        Path(args.dir),
        project_root / "config" / "experiments" / args.dir,
        project_root / args.dir,
    ]
    
    target_dir = None
    for p in search_paths:
        if p.is_dir():
            target_dir = p
            break
            
    if not target_dir:
        print(f"Error: Directory not found: {args.dir}")
        sys.exit(1)

    config_files = sorted(list(target_dir.glob("*.yaml")))

    if not config_files:
        print(f"Error: No YAML files found in {target_dir}")
        sys.exit(1)

    print(f"Found {len(config_files)} experiments to run.")

    with Pool(processes=args.jobs) as pool:
        results = pool.starmap(
            run_experiment,
            [(config, args.gpu_id, project_root, args.retry, args.timeout) for config in config_files],
        )

    failed_jobs = sum(results)
    if failed_jobs > 0:
        print(f"{failed_jobs} experiments failed.")
        sys.exit(1)
    else:
        print("All experiments finished successfully.")


if __name__ == "__main__":
    main()
