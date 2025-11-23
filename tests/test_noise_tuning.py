"""
Systematic test script for BART noise injection parameter tuning.

This script:
1. Reuses the existing trained BART model (no retraining)
2. Runs augmentation on a fixed dataset
3. Displays matched attribute results in a clear format
4. Allows iterative tuning of noise_std parameter

Usage:
    python tests/test_noise_tuning.py [--noise-std 0.15]
"""

import logging
import argparse
import re
from tabulate import tabulate
from typing import List, Dict
from colorama import Fore, Style, init

from src.augmentation.methods.plm import PLMAugmenter
from src.core.dataset.reader.dataset_reader_factory import DatasetReaderFactory
from src.augmentation.methods.plm.set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph
from src.reduction.registry import load_builtin_reducers, REDUCTION_REGISTRY

# Initialize colorama for colored output
init(autoreset=True)

# Suppress RDF warnings
logging.getLogger("rdflib.term").setLevel(logging.ERROR)


class MatchedAttributeCollector(logging.Handler):
    """Custom logging handler that captures matched attribute augmentation results."""

    def __init__(self):
        super().__init__()
        self.results: List[Dict] = []
        # Pattern to match log messages like:
        #     [1/5] name ↔ name (conf:0.91) | 'priest judas' + 'priest judas' → 'Priest Judas' / 'Priest Judas'
        self.pattern = re.compile(
            r"\s*\[(\d+)/(\d+)\]\s+(\S+)\s+↔\s+(\S+)\s+\(conf:([\d.]+)\)\s+\|\s+'([^']+)'\s+\+\s+'([^']+)'\s+→\s+'([^']+)'\s+/\s+'([^']+)'"
        )

    def emit(self, record):
        """Capture matched attribute log messages."""
        msg = self.format(record)
        # Strip ANSI color codes
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        msg = ansi_escape.sub('', msg)
        match = self.pattern.search(msg)

        if match:
            idx, total, src_pred, tgt_pred, conf, src_in, tgt_in, src_out, tgt_out = match.groups()
            self.results.append({
                'idx': int(idx),
                'total': int(total),
                'src_predicate': src_pred,
                'tgt_predicate': tgt_pred,
                'confidence': float(conf),
                'source_in': src_in,
                'target_in': tgt_in,
                'source_out': src_out,
                'target_out': tgt_out,
                'identical_inputs': src_in.lower().strip() == tgt_in.lower().strip()
            })

    def clear(self):
        """Clear collected results."""
        self.results.clear()

    def categorize_result(self, result: Dict) -> str:
        """Categorize result quality: GOOD, CONSERVATIVE, or TOO_CREATIVE."""
        src_in = result['source_in'].lower().strip()
        tgt_in = result['target_in'].lower().strip()
        src_out = result['source_out'].lower().strip()
        tgt_out = result['target_out'].lower().strip()

        # Check if inputs are identical
        identical = src_in == tgt_in

        # Check if outputs changed meaningfully
        def normalize(text):
            """Remove punctuation and extra spaces for comparison."""
            return ' '.join(''.join(c.lower() for c in text if c.isalnum() or c.isspace()).split())

        src_in_norm = normalize(src_in)
        src_out_norm = normalize(src_out)
        tgt_in_norm = normalize(tgt_in)
        tgt_out_norm = normalize(tgt_out)

        src_changed = src_in_norm != src_out_norm
        tgt_changed = tgt_in_norm != tgt_out_norm

        # Conservative: identical inputs but outputs barely changed (just capitalization/punctuation)
        if identical and not src_changed and not tgt_changed:
            return "CONSERVATIVE"

        # Too creative: outputs have gibberish or completely unrelated tokens
        def has_gibberish(text: str) -> bool:
            # Check for sequences of consonants > 5 or very long words
            tokens = text.split()
            for token in tokens:
                # Remove punctuation
                clean_token = ''.join(c for c in token if c.isalpha())
                if len(clean_token) > 15:  # Very long word
                    return True
                # Check consonant sequences
                consonants = 0
                for c in clean_token.lower():
                    if c not in 'aeiou':
                        consonants += 1
                        if consonants > 6:  # More than 6 consecutive consonants
                            return True
                    else:
                        consonants = 0
            return False

        if has_gibberish(src_out) or has_gibberish(tgt_out):
            return "TOO_CREATIVE"

        # Good: reasonable variation
        return "GOOD"

    def print_summary(self, noise_std: float):
        """Print a summary table of all results."""
        if not self.results:
            print("No matched attribute results collected.")
            return {'GOOD': 0, 'CONSERVATIVE': 0, 'TOO_CREATIVE': 0}, 0

        # Prepare table data
        table_data = []
        stats = {'GOOD': 0, 'CONSERVATIVE': 0, 'TOO_CREATIVE': 0}

        for r in self.results:
            category = self.categorize_result(r)
            stats[category] += 1

            # Color coding
            if category == 'GOOD':
                color = Fore.GREEN
                symbol = '✓'
            elif category == 'CONSERVATIVE':
                color = Fore.YELLOW
                symbol = '⚠'
            else:  # TOO_CREATIVE
                color = Fore.RED
                symbol = '✗'

            # Format row
            identical_mark = ' (ID)' if r['identical_inputs'] else ''
            predicate = f"{r['src_predicate'][:15]}"

            table_data.append([
                f"{color}{symbol}{Style.RESET_ALL}",
                category,
                predicate,
                f"{r['source_in']}{identical_mark}",
                r['target_in'],
                f"{color}{r['source_out']}{Style.RESET_ALL}",
                f"{color}{r['target_out']}{Style.RESET_ALL}",
                f"{r['confidence']:.2f}"
            ])

        # Print table
        headers = ['', 'Quality', 'Pred', 'Source In', 'Target In', 'Source Out', 'Target Out', 'Conf']
        print("\n" + "="*140)
        print(f"AUGMENTATION RESULTS (noise_std = {noise_std})")
        print("="*140)
        print(tabulate(table_data, headers=headers, tablefmt='simple'))

        # Print statistics
        total = len(self.results)
        print("\n" + "="*140)
        print("STATISTICS")
        print("="*140)
        print(f"{Fore.GREEN}✓ GOOD:          {stats['GOOD']:3d} ({stats['GOOD']/total*100:5.1f}%){Style.RESET_ALL}")
        print(f"{Fore.YELLOW}⚠ CONSERVATIVE:  {stats['CONSERVATIVE']:3d} ({stats['CONSERVATIVE']/total*100:5.1f}%){Style.RESET_ALL}")
        print(f"{Fore.RED}✗ TOO_CREATIVE:  {stats['TOO_CREATIVE']:3d} ({stats['TOO_CREATIVE']/total*100:5.1f}%){Style.RESET_ALL}")
        print(f"  TOTAL:         {total:3d}")
        print("="*140 + "\n")

        return stats, total


# Global collector instance
collector = MatchedAttributeCollector()


def run_augmentation_test(noise_std: float = 0.15, verbose: bool = True):
    """Run augmentation with specified noise_std and collect results."""

    print(f"\n{'='*140}")
    print(f"RUNNING AUGMENTATION TEST WITH noise_std = {noise_std}")
    print(f"{'='*140}\n")

    # Clear previous results
    collector.clear()

    # Configure logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add collector handler with formatter
    collector.setLevel(logging.DEBUG)
    collector_format = logging.Formatter('%(message)s')
    collector.setFormatter(collector_format)
    root_logger.addHandler(collector)

    # Also add to specific node_expander logger
    node_expander_logger = logging.getLogger('src.augmentation.methods.plm.node_expander')
    node_expander_logger.setLevel(logging.DEBUG)
    node_expander_logger.addHandler(collector)

    # Add console handler for progress messages (less verbose)
    if verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)  # Show INFO and above
        console_format = logging.Formatter('%(asctime)s │ [%(levelname)-7s] │ %(message)s', datefmt='%H:%M:%S')
        console.setFormatter(console_format)
        root_logger.addHandler(console)

    # Suppress noisy loggers
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("rdflib.term").setLevel(logging.ERROR)

    # ----------------------------------------------------------------------------
    # 1. Dataset loading
    # ----------------------------------------------------------------------------
    if verbose:
        print("Loading dataset...")
    reader = DatasetReaderFactory.create_reader("bert_int")
    dataset = reader.read("/home/federico/Programming/Python/DAKGEA/data/raw/openea/BBC_DB/attribute_data")

    # ----------------------------------------------------------------------------
    # 2. Reduction
    # ----------------------------------------------------------------------------
    if verbose:
        print("Reducing dataset to 400 entities...")
    load_builtin_reducers()
    reducer = REDUCTION_REGISTRY.get("random_entities")(
        {"reduction": {"target_entities": 400}, "experiment": {"seed": 11037}}
    )
    reducer.reduce(dataset)

    # ----------------------------------------------------------------------------
    # 3. SetKnowledgeGraph creation
    # ----------------------------------------------------------------------------
    if verbose:
        print("Creating SetKnowledgeGraph...")
    skg = SetKnowledgeGraph.from_dataset(dataset)

    # ----------------------------------------------------------------------------
    # 4. PLM Augmentation
    # ----------------------------------------------------------------------------
    if verbose:
        print(f"Running PLM augmentation (noise_std={noise_std}, reusing model)...\n")

    augmenter = PLMAugmenter({
        "augmentation": {
            "ratio": 0.5,
            "max_depth": 2,
            "bart": {
                "model_name": "./bart_plm_model_base",  # LOAD FINE-TUNED MODEL DIRECTLY
                "enable_finetuning": False,  # DISABLE FINETUNING - JUST LOAD MODEL
                "force_retrain": False,  # REUSE EXISTING MODEL
                "out_dir": "./bart_plm_model_base",
                "epochs": 2,
                "batch_size": 4,
                "max_train_samples": 1000,

                # Regularization
                "weight_decay": 0.01,
                "warmup_steps": 50,
                "max_grad_norm": 1.0,

                # BART interpolation
                "base_alpha": 0.5,
                "alpha_spread": 0.35,

                # Generation parameters (OPTIMAL TUNED)
                "generation": {
                    "max_new_tokens": 32,
                    "do_sample": True,
                    "top_k": 0,
                    "top_p": 0.9,
                    "temperature": 1.0,
                    "num_beams": 2,
                    "repetition_penalty": 1.7,
                    "length_penalty": 1.0,
                    "no_repeat_ngram_size": 3,

                    # NOISE INJECTION - PARAMETER TO TUNE
                    "enable_noise_injection": True,
                    "noise_std": noise_std,  # <<< TUNABLE PARAMETER
                    "noise_apply_when": "identical_inputs",
                },

                # Predicate matching
                "predicate_matching": {
                    "similarity_threshold": 0.6,
                    "use_value_similarity": True,
                    "name_weight": 0.85,
                    "value_weight": 0.15,
                    "alignment_sample_size": 100,
                },

                # Unmatched generation
                "generate_unmatched": True,
                "unmatched_sample_rate": 1.0,
            }
        },
        "experiment": {"seed": 11037}
    })

    # Run augmentation
    dataset_augmented = augmenter.augment(dataset)

    # Print results
    stats, total = collector.print_summary(noise_std)

    return stats, total


def main():
    parser = argparse.ArgumentParser(description='Test BART noise injection parameter tuning')
    parser.add_argument('--noise-std', type=float, default=0.15,
                        help='Noise standard deviation (default: 0.15)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run parameter sweep over multiple noise_std values')
    parser.add_argument('--range', type=str, default='0.10,0.125,0.15,0.175,0.20,0.25',
                        help='Comma-separated noise_std values for sweep (default: 0.10,0.125,0.15,0.175,0.20,0.25)')

    args = parser.parse_args()

    if args.sweep:
        # Run parameter sweep
        noise_values = [float(x.strip()) for x in args.range.split(',')]
        sweep_results = {}

        for noise_std in noise_values:
            stats, total = run_augmentation_test(noise_std, verbose=False)
            sweep_results[noise_std] = {
                'good': stats['GOOD'],
                'conservative': stats['CONSERVATIVE'],
                'too_creative': stats['TOO_CREATIVE'],
                'total': total,
                'good_pct': stats['GOOD'] / total * 100 if total > 0 else 0,
                'conservative_pct': stats['CONSERVATIVE'] / total * 100 if total > 0 else 0,
                'too_creative_pct': stats['TOO_CREATIVE'] / total * 100 if total > 0 else 0,
            }

        # Print sweep summary
        print("\n" + "="*140)
        print("PARAMETER SWEEP SUMMARY")
        print("="*140)

        sweep_table = []
        best_noise_std = None
        best_score = -1

        for noise_std, stats in sorted(sweep_results.items()):
            # Calculate a score: maximize GOOD, minimize CONSERVATIVE and TOO_CREATIVE
            # Score = GOOD% - 0.5*CONSERVATIVE% - TOO_CREATIVE%
            score = stats['good_pct'] - 0.5 * stats['conservative_pct'] - stats['too_creative_pct']

            if score > best_score:
                best_score = score
                best_noise_std = noise_std

            sweep_table.append([
                f"{noise_std:.3f}",
                f"{Fore.GREEN}{stats['good_pct']:5.1f}%{Style.RESET_ALL}",
                f"{Fore.YELLOW}{stats['conservative_pct']:5.1f}%{Style.RESET_ALL}",
                f"{Fore.RED}{stats['too_creative_pct']:5.1f}%{Style.RESET_ALL}",
                f"{score:6.1f}",
                f"{stats['total']} samples"
            ])

        print(tabulate(sweep_table,
                      headers=['noise_std', 'GOOD', 'CONSERVATIVE', 'TOO_CREATIVE', 'Score', 'Total'],
                      tablefmt='simple'))
        print("="*140)
        print(f"\n{Fore.GREEN}✓ BEST noise_std: {best_noise_std} (score: {best_score:.1f}){Style.RESET_ALL}\n")

    else:
        # Run single test
        run_augmentation_test(args.noise_std)


if __name__ == "__main__":
    main()
