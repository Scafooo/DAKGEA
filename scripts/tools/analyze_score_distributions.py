"""Analyze per-entity candidate score changes caused by data augmentation.

One row per (e1, candidate) pair:
  e1 | e2 | match | score_na | rank_na | score_aug | rank_aug | delta_score | delta_rank

Usage
-----
    python scripts/tools/analyze_score_distributions.py \\
        --no-aug  results/<exp>/<run>/artifact/bert_int/reduced/score_distributions.json \\
        --aug     results/<exp>/<run>/artifact/bert_int/augmented/score_distributions.json \\
        [--topk   10]          # candidates per entity to show (default 10)
        [--entity 42]          # show only a specific e1
        [--csv    out.csv]
"""

import argparse
import csv
import json
from pathlib import Path


def load(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {e["e1"]: e for e in data["entities"]}


def build_rows(na: dict, aug: dict, topk: int, entity_filter: int | None) -> list:
    rows = []
    e1_ids = sorted(set(na) & set(aug))
    if entity_filter is not None:
        e1_ids = [e for e in e1_ids if e == entity_filter]

    for e1 in e1_ids:
        na_e  = na[e1]
        aug_e = aug[e1]
        true_match = na_e["true_match"]

        # Build lookup: e2 -> (score, rank) for each condition
        na_cands  = {e2: (score, rank + 1)
                     for rank, (e2, score) in enumerate(na_e["candidates"][:topk])}
        aug_cands = {e2: (score, rank + 1)
                     for rank, (e2, score) in enumerate(aug_e["candidates"][:topk])}

        # Union of candidates from both conditions
        all_e2 = sorted(set(na_cands) | set(aug_cands))

        for e2 in all_e2:
            s_na,  r_na  = na_cands.get(e2,  (None, None))
            s_aug, r_aug = aug_cands.get(e2, (None, None))

            ds = round(s_aug - s_na, 4) if (s_na is not None and s_aug is not None) else None
            dr = (r_aug - r_na)         if (r_na is not None and r_aug is not None) else None

            rows.append({
                "e1":          e1,
                "e2":          e2,
                "match":       "YES" if e2 == true_match else "",
                "score_na":    round(s_na,  4) if s_na  is not None else None,
                "rank_na":     r_na,
                "score_aug":   round(s_aug, 4) if s_aug is not None else None,
                "rank_aug":    r_aug,
                "delta_score": ds,
                "delta_rank":  dr,
            })

    return rows


def fmt(v, width, fmt_str):
    if v is None:
        return " " * (width - 3) + "N/A"
    return format(v, fmt_str).rjust(width)


def print_table(rows: list) -> None:
    header = (f"{'e1':>8}  {'e2':>8}  {'match':>5}  "
              f"{'score_na':>9}  {'rank_na':>7}  "
              f"{'score_aug':>9}  {'rank_aug':>8}  "
              f"{'Δscore':>8}  {'Δrank':>6}")
    sep = "-" * len(header)

    prev_e1 = None
    print(header)
    print(sep)
    for r in rows:
        if prev_e1 is not None and r["e1"] != prev_e1:
            print()   # blank line between entities
        prev_e1 = r["e1"]

        ds = f"{r['delta_score']:+.4f}" if r["delta_score"] is not None else "    N/A"
        dr = f"{r['delta_rank']:+d}"    if r["delta_rank"]  is not None else "  N/A"
        sna  = f"{r['score_na']:.4f}"  if r["score_na"]  is not None else "    N/A"
        saug = f"{r['score_aug']:.4f}" if r["score_aug"] is not None else "    N/A"
        rna  = str(r["rank_na"])  if r["rank_na"]  is not None else "N/A"
        raug = str(r["rank_aug"]) if r["rank_aug"] is not None else "N/A"

        print(f"{r['e1']:>8}  {r['e2']:>8}  {r['match']:>5}  "
              f"{sna:>9}  {rna:>7}  "
              f"{saug:>9}  {raug:>8}  "
              f"{ds:>8}  {dr:>6}")
    print(sep)


def save_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["e1", "e2", "match", "score_na", "rank_na",
              "score_aug", "rank_aug", "delta_score", "delta_rank"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-aug",  required=True, metavar="PATH")
    parser.add_argument("--aug",     required=True, metavar="PATH")
    parser.add_argument("--topk",    type=int, default=10,
                        help="Candidates per entity (default 10)")
    parser.add_argument("--entity",  type=int, default=None,
                        help="Show only this e1 ID")
    parser.add_argument("--csv",     default=None, metavar="PATH")
    args = parser.parse_args()

    na  = load(args.no_aug)
    aug = load(args.aug)

    rows = build_rows(na, aug, args.topk, args.entity)
    print_table(rows)

    if args.csv:
        save_csv(rows, Path(args.csv))


if __name__ == "__main__":
    main()
