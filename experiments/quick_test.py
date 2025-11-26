#!/usr/bin/env python3
"""Quick test script for testing a single parameter configuration.

Usage:
    # Test with custom parameters
    python experiments/quick_test.py --dataset BBC_DB --ratio 0.1 \\
        --temperature 1.0 --alpha 0.5 --top_p 0.9

    # Test with config file
    python experiments/quick_test.py --dataset BBC_DB --ratio 0.1 \\
        --config my_config.yaml
"""

import argparse
import json
import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import get_logger
from src.config.loader import load_yaml
from experiments.runner.runner import ExperimentRunner

logger = get_logger("quick_test")


def main():
    parser = argparse.ArgumentParser(description="Quick parameter test")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--ratio", type=float, required=True, help="Reduction ratio")
    parser.add_argument("--output", type=Path, default="results/quick_test", help="Output directory")
    parser.add_argument("--config", type=Path, help="Custom config file")

    # Individual parameters (override config if specified)
    parser.add_argument("--alpha", type=float, help="base_alpha")
    parser.add_argument("--alpha-spread", type=float, help="alpha_spread")
    parser.add_argument("--temperature", type=float, help="temperature")
    parser.add_argument("--top-p", type=float, help="top_p")
    parser.add_argument("--top-k", type=int, help="top_k")
    parser.add_argument("--beams", type=int, help="num_beams")
    parser.add_argument("--rep-penalty", type=float, help="repetition_penalty")
    parser.add_argument("--noise", type=float, help="noise_std")
    parser.add_argument("--temp-increment", type=float, help="temperature_increment")
    parser.add_argument("--threshold", type=float, help="identical_tokens_threshold")

    args = parser.parse_args()

    # Load base config
    if args.config:
        config = load_yaml(args.config)
    else:
        config = load_yaml(PROJECT_ROOT / "config" / "augmentation" / "plm.yaml")

    # Override with command-line parameters
    bart_cfg = config["augmentation"]["bart"]
    gen_cfg = bart_cfg.setdefault("generation", {})

    if args.alpha is not None:
        bart_cfg["base_alpha"] = args.alpha
    if args.alpha_spread is not None:
        bart_cfg["alpha_spread"] = args.alpha_spread
    if args.temperature is not None:
        gen_cfg["temperature"] = args.temperature
    if args.top_p is not None:
        gen_cfg["top_p"] = args.top_p
    if args.top_k is not None:
        gen_cfg["top_k"] = args.top_k
    if args.beams is not None:
        gen_cfg["num_beams"] = args.beams
    if args.rep_penalty is not None:
        gen_cfg["repetition_penalty"] = args.rep_penalty
    if args.noise is not None:
        gen_cfg["noise_std"] = args.noise
    if args.temp_increment is not None:
        gen_cfg["temperature_increment"] = args.temp_increment
    if args.threshold is not None:
        gen_cfg["identical_tokens_threshold"] = args.threshold

    # Print configuration
    logger.info("="*80)
    logger.info("QUICK TEST")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Ratio: {args.ratio}")
    logger.info(f"Output: {args.output}")
    logger.info("\nParameters:")
    logger.info(f"  base_alpha: {bart_cfg.get('base_alpha')}")
    logger.info(f"  alpha_spread: {bart_cfg.get('alpha_spread')}")
    logger.info(f"  temperature: {gen_cfg.get('temperature')}")
    logger.info(f"  top_p: {gen_cfg.get('top_p')}")
    logger.info(f"  top_k: {gen_cfg.get('top_k')}")
    logger.info(f"  num_beams: {gen_cfg.get('num_beams')}")
    logger.info(f"  repetition_penalty: {gen_cfg.get('repetition_penalty')}")
    logger.info(f"  noise_std: {gen_cfg.get('noise_std')}")
    logger.info(f"  temperature_increment: {gen_cfg.get('temperature_increment')}")
    logger.info(f"  identical_tokens_threshold: {gen_cfg.get('identical_tokens_threshold')}")
    logger.info("="*80)

    # Run experiment
    runner = ExperimentRunner(
        dataset_spec=args.dataset,
        ratio=args.ratio,
        augmentation_method="plm",
        workspace=str(args.output),
        config_override=config,
        overwrite_existing=True,
    )

    runner.run()

    # Print results
    results_path = args.output / "augmentation" / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        logger.info("\n" + "="*80)
        logger.info("RESULTS")
        logger.info("="*80)

        for model_name, model_results in results.items():
            if isinstance(model_results, dict):
                logger.info(f"\n{model_name}:")
                for metric in ["hits@1", "hits@5", "hits@10", "mrr", "precision", "recall", "f-measure"]:
                    if metric in model_results:
                        logger.info(f"  {metric}: {model_results[metric]:.4f}")

        logger.info("="*80)


if __name__ == "__main__":
    main()
