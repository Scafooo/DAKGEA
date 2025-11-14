#!/usr/bin/env python3
"""Export utilities for statistics - CSV, Markdown, LaTeX, Excel."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def write_csv(
    path: Path,
    headers: List[str],
    rows: List[List[str]],
) -> None:
    """Write CSV file with proper quoting."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        writer.writerows(rows)


def write_dataset_summary_csv(
    path: Path,
    dataset_stage_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    metrics: List[str],
) -> None:
    """Export dataset summary as CSV."""
    headers = [
        "dataset",
        "metric",
        "reduction_mean",
        "reduction_std",
        "reduction_min",
        "reduction_max",
        "reduction_count",
        "augmentation_mean",
        "augmentation_std",
        "augmentation_min",
        "augmentation_max",
        "augmentation_count",
        "delta_mean",
        "delta_percentage",
    ]

    rows = []
    for dataset, stage_stats in sorted(dataset_stage_stats.items()):
        for metric in metrics:
            red_stats = stage_stats.get("reduction", {}).get(metric)
            aug_stats = stage_stats.get("augmentation", {}).get(metric)
            if not red_stats and not aug_stats:
                continue

            delta = None
            delta_pct = None
            if red_stats and aug_stats:
                delta = aug_stats["mean"] - red_stats["mean"]
                if red_stats["mean"] != 0:
                    delta_pct = (delta / red_stats["mean"]) * 100

            row = [
                dataset,
                metric,
                f"{red_stats['mean']:.6f}" if red_stats else "",
                f"{red_stats['std']:.6f}" if red_stats else "",
                f"{red_stats['min']:.6f}" if red_stats else "",
                f"{red_stats['max']:.6f}" if red_stats else "",
                str(red_stats["count"]) if red_stats else "",
                f"{aug_stats['mean']:.6f}" if aug_stats else "",
                f"{aug_stats['std']:.6f}" if aug_stats else "",
                f"{aug_stats['min']:.6f}" if aug_stats else "",
                f"{aug_stats['max']:.6f}" if aug_stats else "",
                str(aug_stats["count"]) if aug_stats else "",
                f"{delta:.6f}" if delta is not None else "",
                f"{delta_pct:.2f}" if delta_pct is not None else "",
            ]
            rows.append(row)

    write_csv(path, headers, rows)


def write_markdown_table(
    path: Path,
    headers: List[str],
    rows: List[List[str]],
    alignments: List[str] | None = None,
) -> None:
    """Write Markdown table."""
    ensure_dir(path.parent)

    if alignments is None:
        alignments = ["left"] * len(headers)

    with path.open("w", encoding="utf-8") as f:
        # Header
        f.write("| " + " | ".join(headers) + " |\n")

        # Separator with alignment
        sep_parts = []
        for align in alignments:
            if align == "left":
                sep_parts.append(":---")
            elif align == "right":
                sep_parts.append("---:")
            elif align == "center":
                sep_parts.append(":---:")
            else:
                sep_parts.append("---")
        f.write("| " + " | ".join(sep_parts) + " |\n")

        # Data rows
        for row in rows:
            f.write("| " + " | ".join(str(cell) for cell in row) + " |\n")


def write_dataset_summary_markdown(
    path: Path,
    dataset_stage_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    metrics: List[str],
) -> None:
    """Export dataset summary as Markdown table."""
    headers = [
        "Dataset",
        "Metric",
        "Red Mean",
        "Red Std",
        "Aug Mean",
        "Aug Std",
        "Delta",
        "Δ%",
    ]

    alignments = ["left", "left", "right", "right", "right", "right", "right", "right"]

    rows = []
    for dataset, stage_stats in sorted(dataset_stage_stats.items()):
        for metric in metrics:
            red_stats = stage_stats.get("reduction", {}).get(metric)
            aug_stats = stage_stats.get("augmentation", {}).get(metric)
            if not red_stats and not aug_stats:
                continue

            delta = None
            delta_pct = None
            if red_stats and aug_stats:
                delta = aug_stats["mean"] - red_stats["mean"]
                if red_stats["mean"] != 0:
                    delta_pct = (delta / red_stats["mean"]) * 100

            row = [
                dataset,
                metric,
                f"{red_stats['mean']:.4f}" if red_stats else "—",
                f"{red_stats['std']:.4f}" if red_stats else "—",
                f"{aug_stats['mean']:.4f}" if aug_stats else "—",
                f"{aug_stats['std']:.4f}" if aug_stats else "—",
                f"+{delta:.4f}" if delta and delta > 0 else f"{delta:.4f}" if delta else "—",
                f"+{delta_pct:.1f}%" if delta_pct and delta_pct > 0 else f"{delta_pct:.1f}%" if delta_pct else "—",
            ]
            rows.append(row)

    write_markdown_table(path, headers, rows, alignments)


def write_latex_table(
    path: Path,
    headers: List[str],
    rows: List[List[str]],
    caption: str = "",
    label: str = "",
) -> None:
    """Write LaTeX table."""
    ensure_dir(path.parent)

    n_cols = len(headers)
    col_spec = "l" + "r" * (n_cols - 1)  # First column left, rest right-aligned

    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        if caption:
            f.write(f"\\caption{{{caption}}}\n")
        if label:
            f.write(f"\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")

        # Header
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")

        # Data rows
        for row in rows:
            escaped_row = [str(cell).replace("_", "\\_").replace("%", "\\%") for cell in row]
            f.write(" & ".join(escaped_row) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def write_dataset_summary_latex(
    path: Path,
    dataset_stage_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    metrics: List[str],
) -> None:
    """Export dataset summary as LaTeX table."""
    headers = [
        "Dataset",
        "Metric",
        "Red Mean",
        "Red Std",
        "Aug Mean",
        "Aug Std",
        "Delta",
    ]

    rows = []
    for dataset, stage_stats in sorted(dataset_stage_stats.items()):
        for metric in metrics:
            red_stats = stage_stats.get("reduction", {}).get(metric)
            aug_stats = stage_stats.get("augmentation", {}).get(metric)
            if not red_stats and not aug_stats:
                continue

            delta = None
            if red_stats and aug_stats:
                delta = aug_stats["mean"] - red_stats["mean"]

            row = [
                dataset,
                metric,
                f"{red_stats['mean']:.4f}" if red_stats else "—",
                f"$\\pm${red_stats['std']:.4f}" if red_stats else "—",
                f"{aug_stats['mean']:.4f}" if aug_stats else "—",
                f"$\\pm${aug_stats['std']:.4f}" if aug_stats else "—",
                f"+{delta:.4f}" if delta and delta > 0 else f"{delta:.4f}" if delta else "—",
            ]
            rows.append(row)

    write_latex_table(
        path,
        headers,
        rows,
        caption="Reduction vs Augmentation Performance Comparison",
        label="tab:reduction_augmentation_comparison",
    )


def try_write_excel(
    path: Path,
    dataset_stage_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    metrics: List[str],
) -> bool:
    """Try to export to Excel if openpyxl is available."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
    except ImportError:
        return False

    ensure_dir(path.parent)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Summary"

    # Headers
    headers = [
        "Dataset",
        "Metric",
        "Red Mean",
        "Red Std",
        "Red Min",
        "Red Max",
        "Red N",
        "Aug Mean",
        "Aug Std",
        "Aug Min",
        "Aug Max",
        "Aug N",
        "Delta",
        "Delta %",
    ]

    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")
        cell.alignment = Alignment(horizontal="center")

    # Data
    row_idx = 2
    for dataset, stage_stats in sorted(dataset_stage_stats.items()):
        for metric in metrics:
            red_stats = stage_stats.get("reduction", {}).get(metric)
            aug_stats = stage_stats.get("augmentation", {}).get(metric)
            if not red_stats and not aug_stats:
                continue

            delta = None
            delta_pct = None
            if red_stats and aug_stats:
                delta = aug_stats["mean"] - red_stats["mean"]
                if red_stats["mean"] != 0:
                    delta_pct = (delta / red_stats["mean"]) * 100

            ws.cell(row=row_idx, column=1, value=dataset)
            ws.cell(row=row_idx, column=2, value=metric)
            ws.cell(row=row_idx, column=3, value=red_stats["mean"] if red_stats else None)
            ws.cell(row=row_idx, column=4, value=red_stats["std"] if red_stats else None)
            ws.cell(row=row_idx, column=5, value=red_stats["min"] if red_stats else None)
            ws.cell(row=row_idx, column=6, value=red_stats["max"] if red_stats else None)
            ws.cell(row=row_idx, column=7, value=red_stats["count"] if red_stats else None)
            ws.cell(row=row_idx, column=8, value=aug_stats["mean"] if aug_stats else None)
            ws.cell(row=row_idx, column=9, value=aug_stats["std"] if aug_stats else None)
            ws.cell(row=row_idx, column=10, value=aug_stats["min"] if aug_stats else None)
            ws.cell(row=row_idx, column=11, value=aug_stats["max"] if aug_stats else None)
            ws.cell(row=row_idx, column=12, value=aug_stats["count"] if aug_stats else None)
            ws.cell(row=row_idx, column=13, value=delta)
            ws.cell(row=row_idx, column=14, value=delta_pct)

            # Highlight positive deltas in green
            if delta and delta > 0:
                ws.cell(row=row_idx, column=13).fill = PatternFill(
                    start_color="C6EFCE", end_color="C6EFCE", fill_type="solid"
                )

            row_idx += 1

    # Auto-adjust column widths
    for column in ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if cell.value and len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        ws.column_dimensions[column_letter].width = adjusted_width

    wb.save(path)
    return True


__all__ = [
    "write_dataset_summary_csv",
    "write_dataset_summary_markdown",
    "write_dataset_summary_latex",
    "try_write_excel",
    "ensure_dir",
]
