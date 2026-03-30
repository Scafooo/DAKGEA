"""Analyze per-entity true-match score and rank changes caused by data augmentation.

One row per test entity:
  e1 | true_match | score_na | rank_na | score_aug | rank_aug | delta_score | delta_rank

Usage
-----
    python scripts/tools/analyze_score_distributions.py \\
        --no-aug  results/<exp>/artifact/bert_int/reduced/score_distributions.json \\
        --aug     results/<exp>/artifact/bert_int/plm/score_distributions.json \\
        [--entity 42]   # show only a specific e1
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


def build_rows(na: dict, aug: dict, entity_filter: int | None) -> list:
    rows = []
    e1_ids = sorted(set(na) & set(aug))
    if entity_filter is not None:
        e1_ids = [e for e in e1_ids if e == entity_filter]

    for e1 in e1_ids:
        na_e  = na[e1]
        aug_e = aug[e1]

        s_na  = na_e.get("true_match_score")
        r_na  = na_e.get("rank")
        s_aug = aug_e.get("true_match_score")
        r_aug = aug_e.get("rank")

        ds = round(s_aug - s_na, 4) if (s_na is not None and s_aug is not None) else None
        dr = (r_aug - r_na)         if (r_na is not None and r_aug is not None) else None

        rows.append({
            "e1":          e1,
            "true_match":  na_e.get("true_match"),
            "score_na":    round(s_na,  4) if s_na  is not None else None,
            "rank_na":     r_na  if r_na  != -1 else None,
            "score_aug":   round(s_aug, 4) if s_aug is not None else None,
            "rank_aug":    r_aug if r_aug != -1 else None,
            "delta_score": ds,
            "delta_rank":  dr,
        })

    return rows


def print_table(rows: list) -> None:
    header = (f"{'e1':>8}  {'true_match':>10}  "
              f"{'score_na':>9}  {'rank_na':>7}  "
              f"{'score_aug':>9}  {'rank_aug':>8}  "
              f"{'Δscore':>8}  {'Δrank':>6}")
    sep = "-" * len(header)

    print(header)
    print(sep)
    for r in rows:
        sna  = f"{r['score_na']:.4f}"  if r["score_na"]  is not None else "    N/A"
        saug = f"{r['score_aug']:.4f}" if r["score_aug"] is not None else "    N/A"
        rna  = str(r["rank_na"])  if r["rank_na"]  is not None else "N/A"
        raug = str(r["rank_aug"]) if r["rank_aug"] is not None else "N/A"
        ds   = f"{r['delta_score']:+.4f}" if r["delta_score"] is not None else "    N/A"
        dr   = f"{r['delta_rank']:+d}"    if r["delta_rank"]  is not None else "  N/A"

        print(f"{r['e1']:>8}  {r['true_match']:>10}  "
              f"{sna:>9}  {rna:>7}  "
              f"{saug:>9}  {raug:>8}  "
              f"{ds:>8}  {dr:>6}")
    print(sep)

    # Summary statistics
    scored = [r for r in rows if r["delta_score"] is not None]
    ranked = [r for r in rows if r["delta_rank"]  is not None]
    if scored:
        avg_ds = sum(r["delta_score"] for r in scored) / len(scored)
        improved_s = sum(1 for r in scored if r["delta_score"] > 0)
        print(f"\nΔscore: mean={avg_ds:+.4f}  improved={improved_s}/{len(scored)}")
    if ranked:
        avg_dr = sum(r["delta_rank"] for r in ranked) / len(ranked)
        improved_r = sum(1 for r in ranked if r["delta_rank"] < 0)  # lower rank = better
        print(f"Δrank:  mean={avg_dr:+.2f}  improved={improved_r}/{len(ranked)}")


def save_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["e1", "true_match", "score_na", "rank_na",
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
    parser.add_argument("--entity",  type=int, default=None,
                        help="Show only this e1 ID")
    parser.add_argument("--csv",     default=None, metavar="PATH")
    args = parser.parse_args()

    na  = load(args.no_aug)
    aug = load(args.aug)

    rows = build_rows(na, aug, args.entity)
    print_table(rows)

    if args.csv:
        save_csv(rows, Path(args.csv))


if __name__ == "__main__":
    main()
