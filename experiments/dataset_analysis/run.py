#!/usr/bin/env python3
"""Run dataset analysis."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.dataset_analysis.analyzer import DatasetAnalyzer
from src.logger import get_logger, set_global_level

logger = get_logger(__name__)


def main():
    """Main entry point for dataset analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze HybEA attribute_data format datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze BBC_DB dataset
  python experiments/dataset_analysis/run.py data/raw/hybea/BBC_DB/attribute_data

  # Save results to JSON
  python experiments/dataset_analysis/run.py data/raw/hybea/BBC_DB/attribute_data --output results.json

  # Verbose logging
  python experiments/dataset_analysis/run.py data/raw/hybea/BBC_DB/attribute_data --verbose
        """
    )

    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to attribute_data directory"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output JSON file for results (optional)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except errors"
    )

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        set_global_level("ERROR")
    elif args.verbose:
        set_global_level("DEBUG")
    else:
        set_global_level("INFO")

    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset path not found: {dataset_path}")
        sys.exit(1)

    # Run analysis
    try:
        analyzer = DatasetAnalyzer(dataset_path)
        results = analyzer.run_full_analysis()

        # Save to JSON if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"\n📄 Results saved to: {output_path}")

        logger.info("\n✅ Analysis completed successfully!")

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
