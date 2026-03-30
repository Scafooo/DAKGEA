"""Run a no-aug + aug experiment pair and analyse per-entity score changes.

The script:
  1. Generates two experiment configs (reduction-only and augmented)
  2. Runs both via the experiment runner
  3. Calls analyze_score_distributions.py on the resulting score files

Usage
-----
    python scripts/run_score_analysis.py \\
        --dataset     openea/BBC_DB \\
        --red-ratio   0.1 \\
        --aug-ratio   0.5 \\
        [--seed       11037] \\
        [--topk       10] \\
        [--entity     42] \\
        [--csv        analysis/scores.csv]
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


# ── config generation ─────────────────────────────────────────────────────────

def make_no_aug_config(dataset: str, red_ratio: float, seed: int) -> dict:
    return {
        "experiment": {
            "name": "score_analysis_no_aug",
            "dataset": {"name": dataset, "writer": "bert_int"},
            "reduction": {
                "method": "random_entities",
                "ratio": red_ratio,
                "writer": "bert_int",
                "save": False,
                "eval": True,
            },
            "model": "bert_int",
            "seed": seed,
            "clear": True,
            "overwrite_existing": True,
        }
    }


def make_aug_config(dataset: str, red_ratio: float, aug_ratio: float, seed: int) -> dict:
    return {
        "experiment": {
            "name": "score_analysis_aug",
            "dataset": {"name": dataset, "writer": "bert_int"},
            "reduction": {
                "method": "random_entities",
                "ratio": red_ratio,
                "writer": "bert_int",
                "save": False,
                "eval": False,
            },
            "augmentation": {
                "method": "plm",
                "ratio": aug_ratio,
                "writer": "bert_int",
                "save": False,
                "eval": True,
            },
            "model": "bert_int",
            "seed": seed,
            "clear": True,
            "overwrite_existing": True,
        }
    }


# ── path discovery ────────────────────────────────────────────────────────────

def find_score_file(exp_name: str, variant: str) -> Path:
    pattern = f"results/{exp_name}/**/artifact/bert_int/{variant}/score_distributions.json"
    matches = list(Path(".").glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"score_distributions.json not found for variant '{variant}' in {exp_name}.\n"
            f"Searched: {pattern}"
        )
    if len(matches) > 1:
        # Take the most recently modified
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


# ── runner ────────────────────────────────────────────────────────────────────

def run_experiment(config: dict, label: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix=f"score_analysis_{label}_", delete=False
    ) as f:
        yaml.dump(config, f, default_flow_style=False)
        config_path = f.name

    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"Config:  {config_path}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, "-m", "experiments.runner", config_path],
        check=False,
    )
    if result.returncode != 0:
        print(f"ERROR: experiment '{label}' failed (exit code {result.returncode})")
        sys.exit(result.returncode)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",   required=True, help="e.g. openea/BBC_DB")
    parser.add_argument("--red-ratio", type=float, required=True,
                        help="Reduction ratio, e.g. 0.1")
    parser.add_argument("--aug-ratio", type=float, required=True,
                        help="Augmentation ratio, e.g. 0.5")
    parser.add_argument("--seed",      type=int, default=11037)
    # analysis args
    parser.add_argument("--topk",   type=int, default=10,
                        help="Candidates per entity to show (default 10)")
    parser.add_argument("--entity", type=int, default=None,
                        help="Show only this e1 ID")
    parser.add_argument("--csv",    default=None, metavar="PATH",
                        help="Save table to CSV")
    args = parser.parse_args()

    # 1. Run no-aug
    no_aug_cfg = make_no_aug_config(args.dataset, args.red_ratio, args.seed)
    run_experiment(no_aug_cfg, "no_aug")

    # 2. Run aug
    aug_cfg = make_aug_config(args.dataset, args.red_ratio, args.aug_ratio, args.seed)
    run_experiment(aug_cfg, "aug")

    # 3. Locate score files
    no_aug_file = find_score_file("score_analysis_no_aug", "reduced")
    aug_file    = find_score_file("score_analysis_aug",    "plm")

    print(f"\nno-aug scores : {no_aug_file}")
    print(f"aug scores    : {aug_file}")

    # 4. Run analysis
    print(f"\n{'='*60}")
    print("Score analysis")
    print(f"{'='*60}\n")

    analysis_args = [
        sys.executable,
        "scripts/tools/analyze_score_distributions.py",
        "--no-aug", str(no_aug_file),
        "--aug",    str(aug_file),
        "--topk",   str(args.topk),
    ]
    if args.entity is not None:
        analysis_args += ["--entity", str(args.entity)]
    if args.csv:
        analysis_args += ["--csv", args.csv]

    subprocess.run(analysis_args, check=True)


if __name__ == "__main__":
    main()
