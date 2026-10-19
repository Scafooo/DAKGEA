#!/usr/bin/env python3
"""Collect every Luffy-L run under results/, verify it and write manifest.csv.

Usage (repo root):
  python scripts/luffy_l/build_manifest.py [--results results] [--out results/luffy_l_manifest.csv]

A run is found through its augmentation/luffy_l_stats.json; its BERT-INT dataset
is the folder under augmentation/dataset/ that contains sup_pairs.
The experiment name must be <DATASET>_<tf:02d>_<af:02d>_s<seed> (as written by
generate_configs.py). Exit code 1 if any run fails a hard check or is missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_outputs import verify  # noqa: E402

NAME = re.compile(r"(?P<ds>[A-Z0-9_]+?)_(?P<tf>\d{2})_(?P<af>\d{2})_s(?P<seed>\d+)$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", type=Path)
    ap.add_argument("--raw-root", default="data/raw/openea", type=Path)
    ap.add_argument("--out", default="results/luffy_l_manifest.csv", type=Path)
    ap.add_argument("--configs", default="config/experiments/luffy_l", type=Path,
                    help="used to list expected runs that produced nothing")
    args = ap.parse_args()

    rows, seen, bad = [], set(), 0
    for stats_path in sorted(args.results.rglob("augmentation/luffy_l_stats.json")):
        stats = json.loads(stats_path.read_text())
        name = next((p.name for p in stats_path.parents if NAME.match(p.name)), None)
        if name is None:
            print(f"skip (unrecognised experiment folder): {stats_path}")
            continue
        m = NAME.match(name)
        variant = stats["variant"]
        key = (variant, name)
        if key in seen:
            print(f"duplicate run for {key}: {stats_path}")
            bad += 1
            continue
        seen.add(key)
        ds_dirs = [p.parent for p in (stats_path.parent / "dataset").rglob("sup_pairs")]
        row = {"variant": variant, "dataset": m["ds"], "tf": int(m["tf"]) / 10, "af": int(m["af"]) / 10,
               "seed": int(m["seed"]), "path": str(ds_dirs[0]) if len(ds_dirs) == 1 else "",
               **{k: stats.get(k) for k in (
                   "copy_original_literals", "initial_pairs", "pair_budget", "pairs_augmented",
                   "llm_requests", "api_calls", "disk_cache_hits", "llm_cache_hits",
                   "parse_failures", "values_generated", "values_fallback_original",
                   "values_identical_to_original", "out_of_alphabet_dropped", "truncated_outputs",
                   "input_tokens", "output_tokens")}}
        if len(ds_dirs) != 1:
            row["status"], row["problems"] = "FAIL", f"expected 1 dataset folder, found {len(ds_dirs)}"
        else:
            res = verify(ds_dirs[0], stats_path, args.raw_root / m["ds"], int(m["tf"]) / 10, variant)
            row["status"] = "FAIL" if res["failed"] else "OK"
            row["problems"] = "; ".join(f"{k}:{v['status']} {v['detail']}"
                                        for k, v in res["checks"].items() if v["status"] != "OK")
        bad += row["status"] == "FAIL"
        rows.append(row)

    missing = []
    if args.configs.exists():
        for cfg in sorted(args.configs.rglob("*.yaml")):
            if (cfg.parent.name, cfg.stem) not in seen:
                missing.append(str(cfg))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (args.out.parent / "luffy_l_missing_runs.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    print(f"runs: {len(rows)} | FAIL: {bad} | missing (config without output): {len(missing)} -> {args.out}")
    return 1 if bad or missing else 0


if __name__ == "__main__":
    sys.exit(main())
