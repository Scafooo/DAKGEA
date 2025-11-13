#!/usr/bin/env python3
"""Compare reduction vs augmentation metrics aggregated per dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import logging
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.loader import PROJECT_ROOT, load_yaml

RESULTS_FILENAME = "results.json"
SUMMARY_FILENAME = "summary.json"
METADATA_FILENAME = "metadata.json"
STAGES = ("reduction", "augmentation")
DEFAULT_METRICS = ["hits@1", "hits@5", "hits@10", "hits@25", "hits@50", "mrr", "mr"]
PLOT_METRICS = ["hits@1", "hits@5", "hits@10", "hits@25", "hits@50"]
DEFAULT_DPI = 200
PLOT_DPI = DEFAULT_DPI


def parse_args(default_plots_dir: Path, default_results_dir: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate reduction vs augmentation metrics per dataset.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Experiment directories to include. If omitted, scans all ./results/* directories.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to aggregate (default: %(default)s).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Restrict aggregation to specific model names (default: use all).",
    )
    parser.add_argument(
        "--plot-metric",
        default="hits@1",
        help="Metric used for the comparison plots (default: hits@1).",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to dump the aggregated statistics as JSON.",
    )
    parser.add_argument(
        "--plots-dir",
        default=str(default_plots_dir),
        help=f"Directory where comparison plots will be saved (default: {default_plots_dir}).",
    )
    parser.add_argument(
        "--tsv-dir",
        default=str(default_plots_dir),
        help=f"Directory where TSV summaries will be written (default: {default_plots_dir}).",
    )
    parser.add_argument(
        "--results-root",
        default=str(default_results_dir),
        help=f"Root directory containing experiment runs (default: {default_results_dir}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Resolution (DPI) for saved plots (default: %(default)s).",
    )
    return parser.parse_args()


def discover_experiments(candidate_paths: Iterable[str], results_root: Path) -> List[Path]:
    if candidate_paths:
        paths = [Path(p) for p in candidate_paths]
        return [p.resolve() for p in paths if p.exists()]
    root = results_root
    if not root.exists():
        raise FileNotFoundError("No experiment directories found and no explicit paths provided.")
    return sorted(p.resolve() for p in root.iterdir() if p.is_dir())


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage_metrics(
    stage_dir: Path,
    metrics: List[str],
    model_filter: List[str] | None,
) -> List[Dict]:
    result_file = stage_dir / RESULTS_FILENAME
    if not result_file.exists():
        return []
    payload = load_json(result_file)
    records: List[Dict] = []
    for model_name, stats in payload.items():
        if model_filter and model_name not in model_filter:
            continue
        record_metrics = {metric: float(stats[metric]) for metric in metrics if metric in stats}
        if record_metrics:
            records.append({"model": model_name, "metrics": record_metrics})
    return records


def load_stage_ratio(stage_dir: Path) -> float | None:
    summary = load_json(stage_dir / SUMMARY_FILENAME)
    ratio = summary.get("ratio")
    if isinstance(ratio, (int, float)):
        return float(ratio)
    return None


def average_record(records: List[Dict], metrics: List[str]) -> Dict[str, float]:
    """Average metric values across models for a single experiment."""
    averages: Dict[str, float] = {}
    for metric in metrics:
        values = [record["metrics"][metric] for record in records if metric in record["metrics"]]
        if values:
            averages[metric] = mean(values)
    return averages


def collect_data(
    experiments: List[Path],
    metrics: List[str],
    model_filter: List[str] | None,
) -> Tuple[Dict[str, Dict[str, Dict]], Dict[str, List[Dict]]]:
    datasets: Dict[str, Dict[str, Dict]] = defaultdict(
        lambda: {stage: {"records": [], "ratios": []} for stage in STAGES}
    )
    ratio_entries: Dict[str, List[Dict]] = defaultdict(list)
    for exp_dir in experiments:
        metadata = load_json(exp_dir / METADATA_FILENAME)
        dataset_name = metadata.get("dataset") or exp_dir.name
        entry_data: Dict[str, Dict] = {}
        for stage in STAGES:
            stage_dir = exp_dir / stage
            if not stage_dir.exists():
                continue
            records = load_stage_metrics(stage_dir, metrics, model_filter)
            if not records:
                continue
            ratio = load_stage_ratio(stage_dir)
            datasets[dataset_name][stage]["records"].extend(records)
            if ratio is not None:
                datasets[dataset_name][stage]["ratios"].append(ratio)
            entry_data[stage] = {
                "ratio": ratio,
                "metrics": average_record(records, metrics),
            }
        if entry_data:
            ratio_entries[dataset_name].append(entry_data)
    return datasets, ratio_entries


def summarize_records(records: List[Dict], metrics: List[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        values = [record["metrics"][metric] for record in records if metric in record["metrics"]]
        if not values:
            continue
        summary[metric] = {
            "count": len(values),
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def summarize_ratios(ratios: List[float]) -> Dict[str, float] | None:
    if not ratios:
        return None
    return {
        "count": len(ratios),
        "mean": mean(ratios),
        "min": min(ratios),
        "max": max(ratios),
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_dataset(
    dataset: str,
    metric: str,
    stage_stats: Dict[str, Dict[str, Dict]],
    plots_dir: Path,
) -> None:
    values = []
    labels = []
    for stage in STAGES:
        stats = stage_stats.get(stage, {})
        metric_stats = stats.get(metric)
        if metric_stats:
            values.append(metric_stats["mean"])
            labels.append(stage.capitalize())
    if len(values) < 2:
        return
    ensure_dir(plots_dir)
    plt.figure(figsize=(5, 4))
    bars = plt.bar(labels, values, color=["#457b9d", "#e76f51"])
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
    plt.title(f"{dataset} – {metric}")
    plt.ylabel(metric)
    plt.ylim(0, max(values) * 1.1)
    outfile = plots_dir / f"{dataset}_{metric.replace('@', 'at').replace('/', '_')}.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=PLOT_DPI)
    plt.close()


def build_ratio_plot_data(
    ratio_entries: Dict[str, List[Dict]],
    metrics: List[str],
) -> Dict[str, Dict[float, Dict[float, Dict[str, Dict[str, float]]]]]:
    plot_data: Dict[str, Dict[float, Dict[float, Dict[str, Dict[str, float]]]]] = {}
    for dataset, entries in ratio_entries.items():
        dataset_group = plot_data.setdefault(dataset, {})
        for entry in entries:
            red_ratio = entry.get("reduction", {}).get("ratio")
            aug_ratio = entry.get("augmentation", {}).get("ratio")
            if red_ratio is None or aug_ratio is None:
                continue
            try:
                red_ratio = round(float(red_ratio), 6)
                aug_ratio = round(float(aug_ratio), 6)
            except (TypeError, ValueError):
                continue
            red_metrics = entry.get("reduction", {}).get("metrics", {})
            aug_metrics = entry.get("augmentation", {}).get("metrics", {})
            red_group = dataset_group.setdefault(red_ratio, {})
            agg_entry = red_group.setdefault(
                aug_ratio,
                {
                    "reduction": {metric: [] for metric in metrics},
                    "augmentation": {metric: [] for metric in metrics},
                },
            )
            for metric in metrics:
                if metric in red_metrics:
                    agg_entry["reduction"][metric].append(red_metrics[metric])
                if metric in aug_metrics:
                    agg_entry["augmentation"][metric].append(aug_metrics[metric])

    for _, red_group in plot_data.items():
        for _, aug_group in red_group.items():
            for _, values in aug_group.items():
                for stage in ("reduction", "augmentation"):
                    for metric in metrics:
                        vals = values[stage][metric]
                        values[stage][metric] = mean(vals) if vals else None
    return plot_data


def plot_ratio_groups(
    dataset: str,
    ratio_plot_data: Dict[float, Dict[float, Dict[str, Dict[str, float]]]],
    metrics: List[str],
    plots_dir: Path,
) -> None:
    ratio_dir = plots_dir / "ratio_charts"
    ensure_dir(ratio_dir)
    for red_ratio, aug_group in sorted(ratio_plot_data.items()):
        if not aug_group:
            continue
        aug_ratios = sorted(aug_group.keys())
        if not aug_ratios:
            continue

        metrics_count = len(metrics)
        bar_width = 0.35

        fig, axes = plt.subplots(
            1,
            len(aug_ratios),
            figsize=(4 * len(aug_ratios), 4),
            sharey=True,
        )
        if len(aug_ratios) == 1:
            axes = [axes]

        for ax, aug_ratio in zip(axes, aug_ratios):
            data = aug_group[aug_ratio]
            x = range(metrics_count)
            red_values = [data["reduction"].get(metric) for metric in metrics]
            aug_values = [data["augmentation"].get(metric) for metric in metrics]

            ax.bar(
                [i - bar_width / 2 for i in x],
                red_values,
                width=bar_width,
                color="#264653",
                label="Reduction",
            )
            ax.bar(
                [i + bar_width / 2 for i in x],
                aug_values,
                width=bar_width,
                color="#e76f51",
                label="Augmentation",
            )

            ax.set_xticks(list(x))
            ax.set_xticklabels(metrics, rotation=45)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Aug ratio {aug_ratio:.2f}")
            for i, (r_val, a_val) in enumerate(zip(red_values, aug_values)):
                if r_val is not None:
                    ax.text(
                        i - bar_width / 2,
                        r_val + 0.01,
                        f"{r_val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
                if a_val is not None:
                    ax.text(
                        i + bar_width / 2,
                        a_val + 0.01,
                        f"{a_val:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
            if ax is axes[0]:
                ax.set_ylabel("Score")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.suptitle(f"{dataset} – reduction ratio {red_ratio:.2f}", y=1.05)
        plt.tight_layout()
        outfile = ratio_dir / f"{dataset}_red{red_ratio:.2f}_comparison.png"
        plt.savefig(outfile, bbox_inches="tight", dpi=args_dpi())
        plt.close(fig)


def write_dataset_summary_tsv(
    path: Path,
    dataset_stage_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    metrics: List[str],
) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        header = [
            "dataset",
            "metric",
            "reduction_mean",
            "reduction_std",
            "reduction_count",
            "augmentation_mean",
            "augmentation_std",
            "augmentation_count",
            "delta_mean",
        ]
        handle.write("\t".join(header) + "\n")
        for dataset, stage_stats in sorted(dataset_stage_stats.items()):
            for metric in metrics:
                red_stats = stage_stats.get("reduction", {}).get(metric)
                aug_stats = stage_stats.get("augmentation", {}).get(metric)
                if not red_stats and not aug_stats:
                    continue
                row = [
                    dataset,
                    metric,
                    f"{red_stats['mean']:.6f}" if red_stats else "",
                    f"{red_stats['std']:.6f}" if red_stats else "",
                    str(red_stats["count"]) if red_stats else "",
                    f"{aug_stats['mean']:.6f}" if aug_stats else "",
                    f"{aug_stats['std']:.6f}" if aug_stats else "",
                    str(aug_stats["count"]) if aug_stats else "",
                    (
                        f"{(aug_stats['mean'] - red_stats['mean']):.6f}"
                        if red_stats and aug_stats
                        else ""
                    ),
                ]
                handle.write("\t".join(row) + "\n")


def write_ratio_summary_tsv(
    path: Path,
    ratio_entries: Dict[str, List[Dict]],
    metrics: List[str],
) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        header = [
            "dataset",
            "reduction_ratio",
            "augmentation_ratio",
            "metric",
            "reduction_mean",
            "augmentation_mean",
            "delta_mean",
            "experiments",
        ]
        handle.write("\t".join(header) + "\n")
        for dataset, entries in sorted(ratio_entries.items()):
            group: Dict[Tuple[float | None, float | None], List[Dict]] = defaultdict(list)
            for entry in entries:
                red_ratio = entry.get("reduction", {}).get("ratio")
                aug_ratio = entry.get("augmentation", {}).get("ratio")
                key = (round(red_ratio, 6) if red_ratio is not None else None,
                       round(aug_ratio, 6) if aug_ratio is not None else None)
                group[key].append(entry)
            for (red_ratio, aug_ratio), grouped_entries in sorted(group.items()):
                for metric in metrics:
                    red_values = [
                        e.get("reduction", {}).get("metrics", {}).get(metric)
                        for e in grouped_entries
                        if metric in e.get("reduction", {}).get("metrics", {})
                    ]
                    aug_values = [
                        e.get("augmentation", {}).get("metrics", {}).get(metric)
                        for e in grouped_entries
                        if metric in e.get("augmentation", {}).get("metrics", {})
                    ]
                    if not red_values and not aug_values:
                        continue
                    red_mean = mean(red_values) if red_values else None
                    aug_mean = mean(aug_values) if aug_values else None
                    delta = (
                        aug_mean - red_mean
                        if (red_mean is not None and aug_mean is not None)
                        else None
                    )
                    row = [
                        dataset,
                        "" if red_ratio is None else f"{red_ratio:.6f}",
                        "" if aug_ratio is None else f"{aug_ratio:.6f}",
                        metric,
                        "" if red_mean is None else f"{red_mean:.6f}",
                        "" if aug_mean is None else f"{aug_mean:.6f}",
                        "" if delta is None else f"{delta:.6f}",
                        str(len(grouped_entries)),
                    ]
                    handle.write("\t".join(row) + "\n")


def load_global_config() -> Dict:
    cfg_path = PROJECT_ROOT / "config/global.yaml"
    return load_yaml(cfg_path) if cfg_path.exists() else {}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger("experiments.statistics")

    global_cfg = load_global_config()
    default_plots_dir = (PROJECT_ROOT / global_cfg.get("paths", {}).get("statistics", "results_analysis")).resolve()
    default_results_dir = (PROJECT_ROOT / global_cfg.get("paths", {}).get("results", "results")).resolve()
    args = parse_args(default_plots_dir, default_results_dir)
    global PLOT_DPI
    PLOT_DPI = args.dpi
    experiments = discover_experiments(args.paths, Path(args.results_root).resolve())
    datasets, ratio_entries = collect_data(experiments, args.metrics, args.models)
    if not datasets:
        raise SystemExit("No metrics found in the specified experiments.")

    aggregated_output = {}
    dataset_stage_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    plots_base = Path(args.plots_dir)
    logger.info("Loaded data from %d experiment directories.", len(experiments))
    logger.info("")
    for dataset, stages in sorted(datasets.items()):
        logger.info("=== Dataset: %s ===", dataset)
        stage_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        for stage in STAGES:
            records = stages[stage]["records"]
            if not records:
                logger.info("  %s: no data", stage.capitalize())
                continue
            stats = summarize_records(records, args.metrics)
            ratios = summarize_ratios(stages[stage]["ratios"])
            stage_stats[stage] = stats
            ratio_str = ""
            if ratios:
                ratio_str = f" (ratio mean={ratios['mean']:.3f})"
            logger.info("  %s%s", stage.capitalize(), ratio_str)
            for metric in args.metrics:
                if metric not in stats:
                    continue
                m = stats[metric]
                logger.info(
                    "    - %s: mean=%.4f, std=%.4f, min=%.4f, max=%.4f, n=%d",
                    metric,
                    m["mean"],
                    m["std"],
                    m["min"],
                    m["max"],
                    m["count"],
                )
        plot_dataset(dataset, args.plot_metric, stage_stats, plots_base)
        dataset_stage_stats[dataset] = stage_stats
        aggregated_output[dataset] = {
            stage: {
                "metrics": stage_stats.get(stage, {}),
                "ratios": summarize_ratios(stages[stage]["ratios"]),
            }
            for stage in STAGES
        }
        logger.info("")

    tsv_dir = Path(args.tsv_dir)
    write_dataset_summary_tsv(tsv_dir / "dataset_summary.tsv", dataset_stage_stats, args.metrics)
    write_ratio_summary_tsv(tsv_dir / "ratio_summary.tsv", ratio_entries, args.metrics)

    ratio_plot_groups = build_ratio_plot_data(ratio_entries, PLOT_METRICS)
    for dataset, ratio_group in ratio_plot_groups.items():
        plot_ratio_groups(dataset, ratio_group, PLOT_METRICS, plots_base)

    if args.output_json:
        out_path = Path(args.output_json)
        ensure_dir(out_path.parent)
        out_path.write_text(json.dumps(aggregated_output, indent=2), encoding="utf-8")
        logger.info("Aggregated data saved to %s", out_path.resolve())

    logger.info("Plots (if produced) are stored under %s", Path(args.plots_dir).resolve())
    logger.info("TSV summaries saved under %s", tsv_dir.resolve())


if __name__ == "__main__":
    main()
