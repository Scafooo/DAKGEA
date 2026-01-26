#!/usr/bin/env python3
"""Custom runner for Forget Labels mode experiments.

This script registers the custom 'forget_labels' reducer before executing the standard
experiment pipeline.
"""

from __future__ import annotations

import argparse
import logging
import time
import sys
from pathlib import Path
from typing import Optional, Sequence

# Add project root to python path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the custom reducer to register it
from scripts.forget_labels_mode.reducer import ForgetLabelsReducer

from experiments.runner import (
    ExperimentRunner,
    autoload_registries,
    load_experiment_cfg,
)

from src.logger import get_logger

logger = get_logger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run Forget Labels mode experiments.",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the experiment YAML configuration file.",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_const",
        const=True,
        default=None,
        help="Reuse cached reductions/augmentations/results when present.",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_const",
        const=False,
        help="Force recomputation even if cached artefacts exist.",
    )
    parser.add_argument(
        "--overwrite-existing",
        dest="overwrite_existing",
        action="store_const",
        const=True,
        default=None,
        help="Recompute and overwrite previously generated artefacts.",
    )
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable tqdm progress display.",
    )
    parser.set_defaults(show_progress=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Script entry point."""
    args = parse_args(argv)
    
    # Load standard registries first
    autoload_registries()
    
    # Custom reducer is already imported and registered via decorator
    logger.info("Registered custom reducer: forget_labels")

    exp_cfg = load_experiment_cfg(args.config)
    overwrite_existing = exp_cfg.get("overwrite_existing")

    if args.overwrite_existing is True:
        overwrite_existing = True
    if args.resume is not None:
        overwrite_existing = not args.resume

    # Auto-disable progress bar if logging is not ERROR level
    root_logger = logging.getLogger("KG_EA")
    if root_logger.level < logging.ERROR and args.show_progress:
        logger.debug("Disabling progress bar due to verbose logging (level < ERROR)")
        args.show_progress = False

    logger.info("Started Forget Labels Experiment at %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    runner = ExperimentRunner(
        exp_cfg,
        overwrite_existing=overwrite_existing,
        show_progress=args.show_progress,
    )
    runner.run()
    logger.info("Finished at %s", time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
