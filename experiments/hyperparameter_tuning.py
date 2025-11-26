#!/usr/bin/env python3
"""Hyperparameter tuning for PLM augmentation.

This script performs grid/random search over PLM hyperparameters to find
the optimal configuration that maximizes hits@1 improvement over baseline.

Usage:
    python experiments/hyperparameter_tuning.py --config config/tuning.yaml --dataset BBC_DB --ratio 0.1
"""

import argparse
import json
import itertools
import random
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from experiments.runner.runner import ExperimentRunner

logger = get_logger("hyperparameter_tuning")


class HyperparameterTuner:
    """Hyperparameter tuner for PLM augmentation."""

    def __init__(
        self,
        base_config_path: Path,
        dataset: str,
        ratio: float,
        output_dir: Path,
        search_type: str = "grid",
        max_trials: int = None,
        metric: str = "hits@1",
    ):
        """Initialize tuner.

        Args:
            base_config_path: Path to base configuration YAML
            dataset: Dataset name (e.g., "BBC_DB")
            ratio: Reduction ratio
            output_dir: Directory to save tuning results
            search_type: "grid" or "random"
            max_trials: Maximum trials for random search (None = all combinations)
            metric: Metric to optimize (default: hits@1)
        """
        self.base_config_path = base_config_path
        self.dataset = dataset
        self.ratio = ratio
        self.output_dir = Path(output_dir)
        self.search_type = search_type
        self.max_trials = max_trials
        self.metric = metric

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load base configuration
        with open(base_config_path) as f:
            self.base_config = yaml.safe_load(f)

        # Tuning results
        self.results = []
        self.best_params = None
        self.best_score = -float('inf')

    def define_search_space(self) -> Dict[str, List[Any]]:
        """Define hyperparameter search space.

        Returns:
            Dictionary mapping parameter names to list of values to try
        """
        search_space = {
            # Alpha parameters (interpolation mixing)
            "base_alpha": [0.3, 0.4, 0.5, 0.6, 0.7],
            "alpha_spread": [0.2, 0.35, 0.45, 0.55, 0.7],

            # Generation parameters
            "temperature": [0.7, 0.85, 1.0, 1.2, 1.4],
            "top_p": [0.85, 0.9, 0.95],
            "top_k": [0, 30, 50],
            "num_beams": [1, 3, 5, 7],
            "repetition_penalty": [1.0, 1.3, 1.5, 1.7, 2.0],
            "no_repeat_ngram_size": [2, 3, 4],

            # Noise injection
            "noise_std": [0.0, 0.001, 0.01, 0.05, 0.1],

            # Retry mechanism
            "temperature_increment": [0.0, 0.01, 0.02, 0.05],
            "identical_tokens_threshold": [0.2, 0.3, 0.4, 0.5],

            # Sentence-level interpolation
            "sentence_chunk_max_tokens": [60, 80, 100],
            "sentence_min_length_for_chunking": [40, 60, 80],
        }

        return search_space

    def get_param_combinations(self, search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate parameter combinations based on search type.

        Args:
            search_space: Search space definition

        Returns:
            List of parameter dictionaries to try
        """
        param_names = list(search_space.keys())
        param_values = list(search_space.values())

        if self.search_type == "grid":
            # Grid search: try all combinations
            combinations = list(itertools.product(*param_values))
            param_dicts = [
                dict(zip(param_names, combo))
                for combo in combinations
            ]
            logger.info(f"Grid search: {len(param_dicts)} combinations")

        elif self.search_type == "random":
            # Random search: sample random combinations
            n_trials = self.max_trials or 100
            param_dicts = []
            for _ in range(n_trials):
                combo = {
                    name: random.choice(values)
                    for name, values in search_space.items()
                }
                param_dicts.append(combo)
            logger.info(f"Random search: {len(param_dicts)} trials")

        else:
            raise ValueError(f"Unknown search type: {self.search_type}")

        return param_dicts

    def create_config_with_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create configuration with specific hyperparameters.

        Args:
            params: Hyperparameters to set

        Returns:
            Modified configuration dictionary
        """
        import copy
        config = copy.deepcopy(self.base_config)

        # Set BART parameters
        bart_cfg = config["augmentation"]["bart"]
        gen_cfg = bart_cfg.setdefault("generation", {})

        # Alpha parameters
        if "base_alpha" in params:
            bart_cfg["base_alpha"] = params["base_alpha"]
        if "alpha_spread" in params:
            bart_cfg["alpha_spread"] = params["alpha_spread"]

        # Generation parameters
        if "temperature" in params:
            gen_cfg["temperature"] = params["temperature"]
        if "top_p" in params:
            gen_cfg["top_p"] = params["top_p"]
        if "top_k" in params:
            gen_cfg["top_k"] = params["top_k"]
        if "num_beams" in params:
            gen_cfg["num_beams"] = params["num_beams"]
        if "repetition_penalty" in params:
            gen_cfg["repetition_penalty"] = params["repetition_penalty"]
        if "no_repeat_ngram_size" in params:
            gen_cfg["no_repeat_ngram_size"] = params["no_repeat_ngram_size"]

        # Noise injection
        if "noise_std" in params:
            gen_cfg["noise_std"] = params["noise_std"]

        # Retry mechanism
        if "temperature_increment" in params:
            gen_cfg["temperature_increment"] = params["temperature_increment"]
        if "identical_tokens_threshold" in params:
            gen_cfg["identical_tokens_threshold"] = params["identical_tokens_threshold"]

        # Sentence-level interpolation
        if "sentence_chunk_max_tokens" in params:
            gen_cfg["sentence_chunk_max_tokens"] = params["sentence_chunk_max_tokens"]
        if "sentence_min_length_for_chunking" in params:
            gen_cfg["sentence_min_length_for_chunking"] = params["sentence_min_length_for_chunking"]

        return config

    def run_trial(self, trial_id: int, params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Run a single tuning trial.

        Args:
            trial_id: Trial number
            params: Hyperparameters to test

        Returns:
            Tuple of (score, full_results)
        """
        logger.info("="*80)
        logger.info(f"TRIAL {trial_id + 1}")
        logger.info("="*80)
        logger.info(f"Parameters: {params}")

        # Create config with these parameters
        config = self.create_config_with_params(params)

        # Save trial config
        trial_config_path = self.output_dir / f"trial_{trial_id:04d}_config.yaml"
        with open(trial_config_path, 'w') as f:
            yaml.dump(config, f)

        # Create trial-specific workspace
        trial_workspace = self.output_dir / f"trial_{trial_id:04d}"
        trial_workspace.mkdir(exist_ok=True)

        # Run experiment
        try:
            runner = ExperimentRunner(
                dataset_spec=self.dataset,
                ratio=self.ratio,
                augmentation_method="plm",
                workspace=str(trial_workspace),
                config_override=config,
            )
            runner.run()

            # Extract results
            results_path = trial_workspace / "augmentation" / "results.json"
            if not results_path.exists():
                logger.error(f"Trial {trial_id}: Results not found")
                return -float('inf'), {}

            with open(results_path) as f:
                results = json.load(f)

            # Get metric value (assume first model in results)
            score = -float('inf')
            for model_name, model_results in results.items():
                if isinstance(model_results, dict) and self.metric in model_results:
                    score = model_results[self.metric]
                    break

            logger.info(f"Trial {trial_id}: {self.metric} = {score:.4f}")

            return score, results

        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}", exc_info=True)
            return -float('inf'), {}

    def run_tuning(self):
        """Run hyperparameter tuning."""
        logger.info("="*80)
        logger.info("HYPERPARAMETER TUNING")
        logger.info("="*80)
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"Ratio: {self.ratio}")
        logger.info(f"Metric: {self.metric}")
        logger.info(f"Search type: {self.search_type}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("="*80)

        # Define search space
        search_space = self.define_search_space()
        logger.info(f"Search space dimensions: {len(search_space)}")
        for param, values in search_space.items():
            logger.info(f"  {param}: {len(values)} values")

        # Generate combinations
        param_combinations = self.get_param_combinations(search_space)

        # Run trials
        for trial_id, params in enumerate(param_combinations):
            score, results = self.run_trial(trial_id, params)

            # Record result
            trial_result = {
                "trial_id": trial_id,
                "params": params,
                "score": score,
                "metric": self.metric,
                "timestamp": datetime.now().isoformat(),
            }
            self.results.append(trial_result)

            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                logger.info(f"🎯 NEW BEST: {self.metric} = {score:.4f}")
                logger.info(f"   Params: {params}")

            # Save intermediate results
            self._save_results()

        # Final summary
        self._print_summary()

    def _save_results(self):
        """Save tuning results to disk."""
        results_file = self.output_dir / "tuning_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "search_type": self.search_type,
                "dataset": self.dataset,
                "ratio": self.ratio,
                "metric": self.metric,
                "best_score": self.best_score,
                "best_params": self.best_params,
                "all_results": self.results,
            }, f, indent=2)
        logger.info(f"Results saved to {results_file}")

    def _print_summary(self):
        """Print tuning summary."""
        logger.info("="*80)
        logger.info("TUNING COMPLETE")
        logger.info("="*80)
        logger.info(f"Trials run: {len(self.results)}")
        logger.info(f"Best {self.metric}: {self.best_score:.4f}")
        logger.info(f"Best parameters:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        logger.info("="*80)

        # Sort by score and show top 5
        sorted_results = sorted(self.results, key=lambda x: x["score"], reverse=True)
        logger.info("\nTop 5 configurations:")
        for i, result in enumerate(sorted_results[:5], 1):
            logger.info(f"{i}. {self.metric}={result['score']:.4f}: {result['params']}")


def main():
    parser = argparse.ArgumentParser(description="PLM hyperparameter tuning")
    parser.add_argument("--config", type=Path, required=True, help="Base config file")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--ratio", type=float, required=True, help="Reduction ratio")
    parser.add_argument("--output", type=Path, default="results/tuning", help="Output directory")
    parser.add_argument("--search-type", choices=["grid", "random"], default="random", help="Search type")
    parser.add_argument("--max-trials", type=int, help="Max trials for random search")
    parser.add_argument("--metric", default="hits@1", help="Metric to optimize")

    args = parser.parse_args()

    # Create tuner
    tuner = HyperparameterTuner(
        base_config_path=args.config,
        dataset=args.dataset,
        ratio=args.ratio,
        output_dir=args.output,
        search_type=args.search_type,
        max_trials=args.max_trials,
        metric=args.metric,
    )

    # Run tuning
    tuner.run_tuning()


if __name__ == "__main__":
    main()
