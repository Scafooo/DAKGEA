#!/usr/bin/env python3
"""Phased hyperparameter tuning for PLM augmentation.

Instead of exhaustive grid search, this performs sequential optimization:
1. Phase 1: Coarse search on key parameters (alpha, temperature)
2. Phase 2: Fine-tune generation parameters (top_p, beams, penalties)
3. Phase 3: Optimize retry and noise parameters
4. Phase 4: Optimize sentence-level parameters

Each phase uses the best parameters from the previous phase as baseline.

Usage:
    python experiments/phased_tuning.py --dataset BBC_DB --ratio 0.1
"""

import argparse
import json
import sys
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from src.config.loader import load_yaml

logger = get_logger("phased_tuning")


class PhasedTuner:
    """Phased hyperparameter tuner."""

    def __init__(
        self,
        dataset: str,
        ratio: float,
        output_dir: Path,
        base_config_path: Path = None,
        metric: str = "hits@1",
        fast_mode: bool = False,
    ):
        """Initialize phased tuner.

        Args:
            dataset: Dataset name
            ratio: Reduction ratio
            output_dir: Output directory
            base_config_path: Base config (default: config/augmentation/plm.yaml)
            metric: Metric to optimize
            fast_mode: Use smaller search spaces for faster tuning
        """
        self.dataset = dataset
        self.ratio = ratio
        self.output_dir = Path(output_dir)
        self.metric = metric
        self.fast_mode = fast_mode

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load base config
        if base_config_path is None:
            base_config_path = PROJECT_ROOT / "config" / "augmentation" / "plm.yaml"

        self.base_config = load_yaml(base_config_path)

        # Current best params (start with base config values)
        bart_cfg = self.base_config["augmentation"]["bart"]
        gen_cfg = bart_cfg.get("generation", {})

        self.best_params = {
            "base_alpha": bart_cfg.get("base_alpha", 0.5),
            "alpha_spread": bart_cfg.get("alpha_spread", 0.45),
            "temperature": gen_cfg.get("temperature", 0.85),
            "top_p": gen_cfg.get("top_p", 0.9),
            "top_k": gen_cfg.get("top_k", 0),
            "num_beams": gen_cfg.get("num_beams", 5),
            "repetition_penalty": gen_cfg.get("repetition_penalty", 1.7),
            "no_repeat_ngram_size": gen_cfg.get("no_repeat_ngram_size", 3),
            "noise_std": gen_cfg.get("noise_std", 0.001),
            "temperature_increment": gen_cfg.get("temperature_increment", 0.02),
            "identical_tokens_threshold": gen_cfg.get("identical_tokens_threshold", 0.3),
            "sentence_chunk_max_tokens": gen_cfg.get("sentence_chunk_max_tokens", 80),
            "sentence_min_length_for_chunking": gen_cfg.get("sentence_min_length_for_chunking", 60),
        }

        self.best_score = -float('inf')
        self.phase_results = []

    def define_phase_search_spaces(self) -> List[Dict[str, Any]]:
        """Define search space for each phase.

        Returns:
            List of phase definitions, each containing:
            - name: Phase name
            - description: What this phase optimizes
            - search_space: Dict of parameters to tune
        """
        if self.fast_mode:
            # Smaller search spaces for fast tuning
            phases = [
                {
                    "name": "Phase 1: Alpha & Temperature",
                    "description": "Optimize core interpolation and sampling",
                    "search_space": {
                        "base_alpha": [0.4, 0.5, 0.6],
                        "alpha_spread": [0.35, 0.45, 0.55],
                        "temperature": [0.7, 0.85, 1.0],
                    }
                },
                {
                    "name": "Phase 2: Generation Parameters",
                    "description": "Optimize sampling and beam search",
                    "search_space": {
                        "top_p": [0.85, 0.9, 0.95],
                        "num_beams": [3, 5, 7],
                        "repetition_penalty": [1.5, 1.7, 2.0],
                    }
                },
                {
                    "name": "Phase 3: Retry & Noise",
                    "description": "Optimize retry mechanism and noise injection",
                    "search_space": {
                        "noise_std": [0.001, 0.01, 0.05],
                        "temperature_increment": [0.01, 0.02, 0.03],
                        "identical_tokens_threshold": [0.2, 0.3, 0.4],
                    }
                },
                {
                    "name": "Phase 4: Sentence-Level",
                    "description": "Optimize long text handling",
                    "search_space": {
                        "sentence_chunk_max_tokens": [70, 80, 90],
                        "sentence_min_length_for_chunking": [50, 60, 70],
                    }
                },
            ]
        else:
            # Full search spaces
            phases = [
                {
                    "name": "Phase 1: Alpha & Temperature",
                    "description": "Optimize core interpolation and sampling",
                    "search_space": {
                        "base_alpha": [0.3, 0.4, 0.5, 0.6, 0.7],
                        "alpha_spread": [0.2, 0.35, 0.45, 0.55, 0.7],
                        "temperature": [0.7, 0.85, 1.0, 1.2, 1.4],
                    }
                },
                {
                    "name": "Phase 2: Sampling Parameters",
                    "description": "Optimize nucleus and top-k sampling",
                    "search_space": {
                        "top_p": [0.85, 0.9, 0.95, 0.98],
                        "top_k": [0, 30, 50, 70],
                    }
                },
                {
                    "name": "Phase 3: Beam Search",
                    "description": "Optimize beam search and length penalty",
                    "search_space": {
                        "num_beams": [1, 3, 5, 7, 10],
                        "no_repeat_ngram_size": [2, 3, 4],
                    }
                },
                {
                    "name": "Phase 4: Penalties",
                    "description": "Optimize repetition and length penalties",
                    "search_space": {
                        "repetition_penalty": [1.0, 1.3, 1.5, 1.7, 2.0],
                    }
                },
                {
                    "name": "Phase 5: Noise Injection",
                    "description": "Optimize noise for creativity",
                    "search_space": {
                        "noise_std": [0.0, 0.001, 0.01, 0.05, 0.1],
                    }
                },
                {
                    "name": "Phase 6: Retry Mechanism",
                    "description": "Optimize retry parameters",
                    "search_space": {
                        "temperature_increment": [0.0, 0.01, 0.02, 0.05],
                        "identical_tokens_threshold": [0.2, 0.3, 0.4, 0.5],
                    }
                },
                {
                    "name": "Phase 7: Sentence-Level",
                    "description": "Optimize long text handling",
                    "search_space": {
                        "sentence_chunk_max_tokens": [60, 70, 80, 90],
                        "sentence_min_length_for_chunking": [40, 50, 60, 70],
                    }
                },
            ]

        return phases

    def generate_param_combinations(self, search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations for a search space.

        Args:
            search_space: Dict of parameter -> list of values

        Returns:
            List of parameter dictionaries
        """
        import itertools

        param_names = list(search_space.keys())
        param_values = list(search_space.values())

        combinations = list(itertools.product(*param_values))

        param_dicts = [
            dict(zip(param_names, combo))
            for combo in combinations
        ]

        return param_dicts

    def create_trial_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create config with specific parameters.

        Args:
            params: Parameters to set

        Returns:
            Full configuration dict
        """
        import copy
        config = copy.deepcopy(self.base_config)

        bart_cfg = config["augmentation"]["bart"]
        gen_cfg = bart_cfg.setdefault("generation", {})

        # Set all parameters
        if "base_alpha" in params:
            bart_cfg["base_alpha"] = params["base_alpha"]
        if "alpha_spread" in params:
            bart_cfg["alpha_spread"] = params["alpha_spread"]

        gen_params = [
            "temperature", "top_p", "top_k", "num_beams",
            "repetition_penalty", "no_repeat_ngram_size",
            "noise_std", "temperature_increment", "identical_tokens_threshold",
            "sentence_chunk_max_tokens", "sentence_min_length_for_chunking"
        ]

        for param in gen_params:
            if param in params:
                gen_cfg[param] = params[param]

        return config

    def run_trial(self, trial_id: str, params: Dict[str, Any]) -> float:
        """Run single trial with given parameters.

        Args:
            trial_id: Trial identifier
            params: Parameters to test

        Returns:
            Score (metric value)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TRIAL: {trial_id}")
        logger.info(f"{'='*80}")
        logger.info(f"Testing: {params}")

        # Create full param dict (base + new)
        full_params = self.best_params.copy()
        full_params.update(params)

        # Create config
        config = self.create_trial_config(full_params)

        # Save config
        trial_dir = self.output_dir / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        config_path = trial_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Run experiment using runner
        try:
            # Import here to avoid circular imports
            from experiments.runner.runner import ExperimentRunner

            runner = ExperimentRunner(
                dataset_spec=self.dataset,
                ratio=self.ratio,
                augmentation_method="plm",
                workspace=str(trial_dir),
                config_override=config,
                overwrite_existing=True,
            )
            runner.run()

            # Read results
            results_path = trial_dir / "augmentation" / "results.json"

            if not results_path.exists():
                logger.error(f"Results not found: {results_path}")
                return -float('inf')

            with open(results_path) as f:
                results = json.load(f)

            # Extract metric
            score = -float('inf')
            for model_name, model_results in results.items():
                if isinstance(model_results, dict) and self.metric in model_results:
                    score = float(model_results[self.metric])
                    break

            logger.info(f"Result: {self.metric} = {score:.4f}")
            return score

        except Exception as e:
            logger.error(f"Trial failed: {e}", exc_info=True)
            return -float('inf')

    def run_phase(self, phase_def: Dict[str, Any], phase_num: int) -> Dict[str, Any]:
        """Run a single tuning phase.

        Args:
            phase_def: Phase definition (name, description, search_space)
            phase_num: Phase number

        Returns:
            Phase results dict
        """
        logger.info(f"\n{'#'*80}")
        logger.info(f"# {phase_def['name']}")
        logger.info(f"# {phase_def['description']}")
        logger.info(f"{'#'*80}\n")

        search_space = phase_def["search_space"]
        param_combinations = self.generate_param_combinations(search_space)

        logger.info(f"Combinations to try: {len(param_combinations)}")
        logger.info(f"Current best {self.metric}: {self.best_score:.4f}")
        logger.info(f"Current best params: {self.best_params}\n")

        phase_results = []
        phase_best_score = self.best_score
        phase_best_params = None

        for i, params in enumerate(param_combinations):
            trial_id = f"phase{phase_num}_trial{i:03d}"
            score = self.run_trial(trial_id, params)

            result = {
                "trial_id": trial_id,
                "params": params,
                "score": score,
                "timestamp": datetime.now().isoformat(),
            }
            phase_results.append(result)

            # Update phase best
            if score > phase_best_score:
                phase_best_score = score
                phase_best_params = params
                logger.info(f"🎯 NEW PHASE BEST: {self.metric} = {score:.4f}")
                logger.info(f"   Params: {params}")

        # Update global best if phase improved
        if phase_best_score > self.best_score:
            improvement = phase_best_score - self.best_score
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ PHASE IMPROVED: +{improvement:.4f}")
            logger.info(f"{'='*80}")

            self.best_score = phase_best_score
            self.best_params.update(phase_best_params)

            logger.info(f"New best {self.metric}: {self.best_score:.4f}")
            logger.info(f"Updated params: {phase_best_params}")
        else:
            logger.info(f"\n{'='*80}")
            logger.info(f"❌ NO IMPROVEMENT in this phase")
            logger.info(f"{'='*80}")

        return {
            "phase_name": phase_def["name"],
            "phase_num": phase_num,
            "trials": len(phase_results),
            "best_score": phase_best_score,
            "best_params": phase_best_params,
            "improvement": phase_best_score - self.best_score if phase_best_params else 0,
            "all_results": phase_results,
        }

    def run_tuning(self):
        """Run complete phased tuning."""
        logger.info(f"\n{'#'*80}")
        logger.info(f"# PHASED HYPERPARAMETER TUNING")
        logger.info(f"{'#'*80}")
        logger.info(f"Dataset: {self.dataset}")
        logger.info(f"Ratio: {self.ratio}")
        logger.info(f"Metric: {self.metric}")
        logger.info(f"Fast mode: {self.fast_mode}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"{'#'*80}\n")

        # Get phases
        phases = self.define_phase_search_spaces()
        logger.info(f"Total phases: {len(phases)}\n")

        # Run each phase
        for phase_num, phase_def in enumerate(phases, 1):
            phase_result = self.run_phase(phase_def, phase_num)
            self.phase_results.append(phase_result)

            # Save intermediate results
            self.save_results()

        # Final summary
        self.print_summary()

    def save_results(self):
        """Save tuning results."""
        results_file = self.output_dir / "tuning_results.json"

        results = {
            "dataset": self.dataset,
            "ratio": self.ratio,
            "metric": self.metric,
            "fast_mode": self.fast_mode,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "phases": self.phase_results,
            "timestamp": datetime.now().isoformat(),
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

        # Also save best config
        best_config = self.create_trial_config(self.best_params)
        best_config_file = self.output_dir / "best_config.yaml"

        with open(best_config_file, 'w') as f:
            yaml.dump(best_config, f)

        logger.info(f"Best config saved to {best_config_file}")

    def print_summary(self):
        """Print final summary."""
        logger.info(f"\n{'#'*80}")
        logger.info(f"# TUNING COMPLETE")
        logger.info(f"{'#'*80}\n")

        logger.info(f"Best {self.metric}: {self.best_score:.4f}\n")

        logger.info("Best parameters:")
        for param, value in sorted(self.best_params.items()):
            logger.info(f"  {param}: {value}")

        logger.info(f"\nPhase summary:")
        for phase in self.phase_results:
            improvement_str = f"+{phase['improvement']:.4f}" if phase['improvement'] > 0 else "no improvement"
            logger.info(f"  {phase['phase_name']}: {improvement_str}")

        logger.info(f"\n{'#'*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Phased hyperparameter tuning")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--ratio", type=float, required=True, help="Reduction ratio")
    parser.add_argument("--output", type=Path, default="results/phased_tuning", help="Output directory")
    parser.add_argument("--config", type=Path, help="Base config file")
    parser.add_argument("--metric", default="hits@1", help="Metric to optimize")
    parser.add_argument("--fast", action="store_true", help="Use smaller search spaces")

    args = parser.parse_args()

    tuner = PhasedTuner(
        dataset=args.dataset,
        ratio=args.ratio,
        output_dir=args.output,
        base_config_path=args.config,
        metric=args.metric,
        fast_mode=args.fast,
    )

    tuner.run_tuning()


if __name__ == "__main__":
    main()
