#!/usr/bin/env python3
"""Generate the Luffy-L experiment configs (generation only, no BERT-INT).

The configs mirror the Luffy-S ones in config/experiments/massive/flan_t5_bert_int/
(same dataset names, forget_labels reduction, bert_int writer with
augmented_only_train) and differ only in the augmentation block and in the
evaluation flags, which are all off: the runner stops after writing the
augmented BERT-INT dataset.

Usage (from the repo root):
  python scripts/luffy_l/generate_configs.py --copy-original-literals {true,false} \
      [--variants full no_cache no_bfs all_llm] [--datasets BBC_DB ...] \
      [--seeds 11037 11038 11039 11040 11041] [--tf 1..10] [--af 1..10] \
      [--provider bedrock|fake|replay] [--out config/experiments/luffy_l]

One YAML per (variant, dataset, tf, af, seed):
  <out>/<variant>/<DATASET>_<tf:02d>_<af:02d>_s<seed>.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

DATASETS = ["BBC_DB", "D_W_15K_V1", "D_W_15K_V2", "ICEW_WIKI", "ICEW_YAGO"]
VARIANTS = ["full", "no_cache", "no_bfs", "all_llm"]
SEEDS = [11037, 11038, 11039, 11040, 11041]      # Federico: start at 11037, +1 per run


def parse_bool(text: str) -> bool:
    if text.lower() in ("true", "1", "yes"):
        return True
    if text.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def make_config(variant, dataset, tf, af, seed, copy, provider, cache_path):
    llm = {"provider": provider}
    if cache_path:
        llm["cache_path"] = cache_path
    return {
        "experiment": {
            "suite": f"luffy_l_{variant}",
            "name": f"{dataset}_{tf:02d}_{af:02d}_s{seed}",
            "dataset": {"name": f"openea/{dataset}", "writer": "bert_int"},
            "reduction": {
                "method": "forget_labels",
                "ratio": round(tf / 10, 1),
                "writer": "bert_int",
                "eval": False,
                "save_dataset": False,
                "save_model": False,
            },
            "augmentation": {
                "method": "llm",
                "ratio": round(af / 10, 1),
                "variant": variant,
                "copy_original_literals": copy,
                "llm": llm,
                "writer": {"type": "bert_int", "augmented_only_train": True},
                "eval": False,
                "save_dataset": True,
                "save_model": False,
            },
            "model": "bert_int",
            "seed": seed,
            "clear": False,
            "overwrite_existing": False,
        }
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy-original-literals", type=parse_bool, required=True,
                    help="true = same as the Luffy-S runs (original literals copied onto augmented "
                         "entities); false = paper behaviour. Must be decided with Federico.")
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--tf", nargs="+", type=int, default=list(range(1, 11)), help="tenths: 1..10")
    ap.add_argument("--af", nargs="+", type=int, default=list(range(1, 11)), help="tenths: 1..10")
    ap.add_argument("--provider", default="bedrock", choices=["bedrock", "fake", "replay"])
    ap.add_argument("--cache-path", default=None,
                    help="override llm.cache_path (use a separate file for --provider fake)")
    ap.add_argument("--out", default="config/experiments/luffy_l")
    args = ap.parse_args()

    out = Path(args.out)
    n = 0
    for variant in args.variants:
        for dataset in args.datasets:
            for tf in args.tf:
                for af in args.af:
                    for seed in args.seeds:
                        cfg = make_config(variant, dataset, tf, af, seed,
                                          args.copy_original_literals, args.provider, args.cache_path)
                        path = out / variant / f"{dataset}_{tf:02d}_{af:02d}_s{seed}.yaml"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                        n += 1
    print(f"wrote {n} configs under {out}/")


if __name__ == "__main__":
    main()
