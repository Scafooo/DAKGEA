"""
Parse augmentation results from log output and display statistics.

Usage:
    # Run augmentation and pipe to this script
    python tests/test_reduction_augmentation.py 2>&1 | python tests/analyze_noise_results.py

    # Or analyze saved log file
    python tests/analyze_noise_results.py < augmentation.log
"""

import sys
import re
from tabulate import tabulate
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Pattern to match log messages like:
#     [1/5] name ↔ name (conf:0.91) | 'priest judas' + 'priest judas' → 'Priest Judas' / 'Priest Judas'
pattern = re.compile(
    r"\[(\d+)/(\d+)\]\s+(\S+)\s+↔\s+(\S+)\s+\(conf:([\d.]+)\)\s+\|\s+'([^']+)'\s+\+\s+'([^']+)'\s+→\s+'([^']+)'\s+/\s+'([^']+)'"
)

# ANSI escape code pattern
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def categorize_result(src_in: str, tgt_in: str, src_out: str, tgt_out: str) -> str:
    """Categorize result quality: GOOD, CONSERVATIVE, or TOO_CREATIVE."""
    src_in_lower = src_in.lower().strip()
    tgt_in_lower = tgt_in.lower().strip()
    src_out_lower = src_out.lower().strip()
    tgt_out_lower = tgt_out.lower().strip()

    # Check if inputs are identical
    identical = src_in_lower == tgt_in_lower

    # Normalize text (remove punctuation and extra spaces)
    def normalize(text):
        return ' '.join(''.join(c.lower() for c in text if c.isalnum() or c.isspace()).split())

    src_in_norm = normalize(src_in)
    src_out_norm = normalize(src_out)
    tgt_in_norm = normalize(tgt_in)
    tgt_out_norm = normalize(tgt_out)

    src_changed = src_in_norm != src_out_norm
    tgt_changed = tgt_in_norm != tgt_out_norm

    # Conservative: identical inputs but outputs barely changed
    if identical and not src_changed and not tgt_changed:
        return "CONSERVATIVE"

    # Too creative: outputs have gibberish or completely unrelated tokens
    def has_gibberish(text: str) -> bool:
        tokens = text.split()
        for token in tokens:
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


def main():
    results = []

    # Read from stdin
    for line in sys.stdin:
        # Strip ANSI codes
        clean_line = ansi_escape.sub('', line)

        # Try to match the pattern
        match = pattern.search(clean_line)
        if match:
            idx, total, src_pred, tgt_pred, conf, src_in, tgt_in, src_out, tgt_out = match.groups()
            results.append({
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

    if not results:
        print("No matched attribute results found in input.")
        return

    # Prepare table data
    table_data = []
    stats = {'GOOD': 0, 'CONSERVATIVE': 0, 'TOO_CREATIVE': 0}

    for r in results:
        category = categorize_result(r['source_in'], r['target_in'], r['source_out'], r['target_out'])
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
            f"{r['source_in'][:30]}{identical_mark}",
            r['target_in'][:30],
            f"{color}{r['source_out'][:30]}{Style.RESET_ALL}",
            f"{color}{r['target_out'][:30]}{Style.RESET_ALL}",
            f"{r['confidence']:.2f}"
        ])

    # Print table
    headers = ['', 'Quality', 'Pred', 'Source In', 'Target In', 'Source Out', 'Target Out', 'Conf']
    print("\n" + "="*140)
    print(f"AUGMENTATION RESULTS ANALYSIS")
    print("="*140)
    print(tabulate(table_data, headers=headers, tablefmt='simple'))

    # Print statistics
    total = len(results)
    print("\n" + "="*140)
    print("STATISTICS")
    print("="*140)
    print(f"{Fore.GREEN}✓ GOOD:          {stats['GOOD']:3d} ({stats['GOOD']/total*100:5.1f}%){Style.RESET_ALL}")
    print(f"{Fore.YELLOW}⚠ CONSERVATIVE:  {stats['CONSERVATIVE']:3d} ({stats['CONSERVATIVE']/total*100:5.1f}%){Style.RESET_ALL}")
    print(f"{Fore.RED}✗ TOO_CREATIVE:  {stats['TOO_CREATIVE']:3d} ({stats['TOO_CREATIVE']/total*100:5.1f}%){Style.RESET_ALL}")
    print(f"  TOTAL:         {total:3d}")
    print("="*140 + "\n")

    # Calculate metrics
    # True Positives (TP): GOOD results
    # False Positives (FP): TOO_CREATIVE results
    # False Negatives (FN): CONSERVATIVE results
    tp = stats['GOOD']
    fp = stats['TOO_CREATIVE']
    fn = stats['CONSERVATIVE']

    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1-Score: 2 × (Precision × Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Show score
    score = stats['GOOD'] - 0.5 * stats['CONSERVATIVE'] - stats['TOO_CREATIVE']
    print(f"SCORE: {score:.1f} (max = {total})")
    print(f"  Formula: GOOD - 0.5×CONSERVATIVE - TOO_CREATIVE")
    print()

    # Show metrics
    print("="*140)
    print("METRICS")
    print("="*140)
    print(f"Precision: {precision:.3f}  (TP / (TP + FP) = GOOD / (GOOD + TOO_CREATIVE))")
    print(f"Recall:    {recall:.3f}  (TP / (TP + FN) = GOOD / (GOOD + CONSERVATIVE))")
    print(f"F1-Score:  {f1_score:.3f}  (2 × Precision × Recall / (Precision + Recall))")
    print("="*140 + "\n")
    print(f"Where:")
    print(f"  TP (True Positives)  = {tp:3d} (GOOD results)")
    print(f"  FP (False Positives) = {fp:3d} (TOO_CREATIVE results)")
    print(f"  FN (False Negatives) = {fn:3d} (CONSERVATIVE results)")
    print()


if __name__ == "__main__":
    main()
