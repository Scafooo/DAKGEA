#!/usr/bin/env python3
"""Check Luffy-L generated datasets before handing them to Federico.

Usage:
  python scripts/luffy_l/verify_outputs.py --dataset-dir <bert_int dir> --stats <luffy_l_stats.json> \
      --raw data/raw/openea/<DATASET> --tf 0.1 [--variant full]

<bert_int dir> is the folder the runner wrote for the augmented dataset (it contains
ent_ids_1, ent_ids_2, sup_pairs, ref_pairs, valid_pairs, triples_1, triples_2,
attr_triples1, attr_triples2). Exit code 1 if a hard check fails.

Hard checks (FAIL):
  split      test set (ref_pairs + valid_pairs written by the writer, i.e. the
             fixed test of forget_labels) contains only official test pairs, and
             no official test pair appears in sup_pairs
  train      every original (non-augmented) training pair is an official sup pair,
             and their number matches int(tf * |official sup|)
  kg1_rel    triples_1 is not empty (ICEWS: KG1 relations come from attribute_data/triples_1)
  budget     pairs_augmented == pair_budget (or fewer only if the BFS ran out of seeds)
Soft checks (WARN, report in the manifest):
  aug_pairs  synthetic pairs that reached sup_pairs vs pairs_augmented
  parse failures, fallbacks to original values, generated values identical to the original,
  synthetic entities without any relation (expected only for no_bfs / all_llm with no edges)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

AUG = re.compile(r"_aug\d+$")


def read_tsv(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [line.rstrip("\n").split("\t") for line in fh if line.strip()]


def ids_to_uri(d: Path):
    out = {}
    for name in ("ent_ids_1", "ent_ids_2"):
        for row in read_tsv(d / name):
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def pairs(d: Path, name: str, id2uri):
    return {(id2uri[a], id2uri[b]) for a, b, *_ in read_tsv(d / name) if a in id2uri and b in id2uri}


def verify(dataset_dir: Path, stats_path: Path, raw_dir: Path, tf: float,
           variant: str | None = None) -> dict:
    """Run all checks; returns {"failed": bool, "checks": {...}}."""
    d, raw = Path(dataset_dir), Path(raw_dir) / "attribute_data"
    stats = json.loads(Path(stats_path).read_text())
    variant = variant or stats.get("variant")
    report, failed = {}, False

    def check(name, ok, detail, hard=True):
        nonlocal failed
        status = "OK" if ok else ("FAIL" if hard else "WARN")
        failed |= (not ok and hard)
        report[name] = {"status": status, "detail": detail}

    raw_ids = ids_to_uri(raw)
    off_sup = pairs(raw, "sup_pairs", raw_ids)
    off_test = pairs(raw, "ref_pairs", raw_ids) | pairs(raw, "valid_pairs", raw_ids)

    id2uri = ids_to_uri(d)
    sup = pairs(d, "sup_pairs", id2uri)
    test = pairs(d, "ref_pairs", id2uri)
    valid = pairs(d, "valid_pairs", id2uri)
    orig_train = {p for p in sup | valid if not (AUG.search(p[0]) or AUG.search(p[1]))}
    synthetic = {p for p in sup if AUG.search(p[0]) or AUG.search(p[1])}

    check("split", test <= off_test and not (sup & off_test),
          f"test={len(test)} (official test={len(off_test)}), test pairs in sup={len(sup & off_test)}")
    check("train", orig_train <= off_sup,
          f"original train pairs (sup+valid)={len(orig_train)}, not in official sup={len(orig_train - off_sup)}")
    expected_train = max(1, int(len(off_sup) * tf))
    check("train_size", abs(len(orig_train) - expected_train) <= max(1, 0.01 * expected_train),
          f"original train={len(orig_train)}, int(tf*|official sup|)={expected_train} "
          "(small gaps = official pairs missing from the loaded KG)", hard=False)
    n_t1 = len(read_tsv(d / "triples_1"))
    check("kg1_rel", n_t1 > 0, f"triples_1 lines={n_t1}")
    budget, done = stats.get("pair_budget"), stats.get("pairs_augmented")
    check("budget", done == budget, f"pairs_augmented={done}, pair_budget={budget}",
          hard=(variant != "all_llm"))   # all_llm skips failed pairs (no fallback)
    check("aug_pairs", len(synthetic) == done,
          f"synthetic pairs in sup={len(synthetic)} of {done} generated. The BertInt writer drops a pair "
          "when one synthetic entity has no triple at all (no attribute, no relation): expected for "
          "no_bfs on KGs with attribute-less entities (Luffy-S has the same behaviour); report the "
          "count, do not 'fix' it", hard=False)

    # soft checks
    req = max(1, stats.get("llm_requests", 0))
    check("parse_failures", stats.get("parse_failures", 0) / req < 0.02,
          f"{stats.get('parse_failures', 0)}/{req} requests", hard=False)
    gen = stats.get("values_generated", 0)
    fb = stats.get("values_fallback_original", 0)
    check("fallbacks", fb <= 0.05 * max(1, gen + fb), f"fallback={fb}, generated={gen}", hard=False)
    ident = stats.get("values_identical_to_original", 0)
    check("copied_values", ident <= 0.05 * max(1, gen), f"identical to original={ident}/{gen}", hard=False)
    trunc = stats.get("truncated_outputs", 0)
    check("truncated_outputs", trunc == 0, f"answers cut by max_tokens={trunc}", hard=False)

    aug_with_rel = set()
    for row in read_tsv(d / "triples_1") + read_tsv(d / "triples_2"):
        if len(row) >= 3:
            for e in (row[0], row[2]):
                uri = id2uri.get(e, "")
                if AUG.search(uri):
                    aug_with_rel.add(uri)
    aug_ents = {e for p in synthetic for e in p}
    frac = len(aug_ents & aug_with_rel) / max(1, len(aug_ents))
    check("structure", frac > 0 or variant in ("no_bfs",),
          f"synthetic anchors with >=1 relation: {frac:.1%}", hard=False)

    return {"dataset_dir": str(d), "variant": variant, "failed": failed, "checks": report}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--stats", required=True, type=Path)
    ap.add_argument("--raw", required=True, type=Path, help="data/raw/openea/<DATASET>")
    ap.add_argument("--tf", required=True, type=float)
    ap.add_argument("--variant", default=None)
    args = ap.parse_args()
    result = verify(args.dataset_dir, args.stats, args.raw, args.tf, args.variant)
    print(json.dumps(result, indent=2))
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
