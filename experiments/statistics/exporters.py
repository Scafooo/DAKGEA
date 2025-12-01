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


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "_": "\\_",
        "%": "\\%",
        "$": "\\$",
        "&": "\\&",
        "#": "\\#",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    result = str(text)
    for char, escaped in replacements.items():
        result = result.replace(char, escaped)
    return result


def write_latex_table(
    path: Path,
    headers: List[str],
    rows: List[List[str]],
    caption: str = "",
    label: str = "",
    col_spec: str | None = None,
    position: str = "htbp",
    small: bool = False,
) -> None:
    """Write LaTeX table with booktabs style.

    Args:
        path: Output file path
        headers: Column headers
        rows: Data rows
        caption: Table caption
        label: Table label for referencing
        col_spec: Column specification (e.g., "lrrr"). If None, auto-generates (left + right-aligned)
        position: Table float position (default: htbp)
        small: Use small font size
    """
    ensure_dir(path.parent)

    n_cols = len(headers)
    if col_spec is None:
        col_spec = "l" + "r" * (n_cols - 1)  # First column left, rest right-aligned

    with path.open("w", encoding="utf-8") as f:
        f.write(f"\\begin{{table}}[{position}]\n")
        f.write("\\centering\n")
        if small:
            f.write("\\small\n")
        if caption:
            f.write(f"\\caption{{{escape_latex(caption)}}}\n")
        if label:
            f.write(f"\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")

        # Header
        escaped_headers = [escape_latex(h) for h in headers]
        f.write(" & ".join(escaped_headers) + " \\\\\n")
        f.write("\\midrule\n")

        # Data rows
        for row in rows:
            escaped_row = [escape_latex(cell) for cell in row]
            f.write(" & ".join(escaped_row) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def write_latex_table_colored(
    path: Path,
    headers: List[str],
    rows: List[List[str]],
    highlight_cols: List[int] | None = None,
    caption: str = "",
    label: str = "",
    col_spec: str | None = None,
    position: str = "htbp",
    small: bool = False,
) -> None:
    """Write LaTeX table with conditional coloring for delta columns.

    Args:
        path: Output file path
        headers: Column headers
        rows: Data rows - each row can be str or tuple (value, color_type) where color_type in ['positive', 'negative', 'neutral']
        highlight_cols: Column indices to apply conditional coloring (None = auto-detect delta columns)
        caption: Table caption
        label: Table label
        col_spec: Column specification
        position: Table float position
        small: Use small font size
    """
    ensure_dir(path.parent)

    n_cols = len(headers)
    if col_spec is None:
        col_spec = "l" + "r" * (n_cols - 1)

    # Auto-detect delta columns if not specified
    if highlight_cols is None:
        highlight_cols = [i for i, h in enumerate(headers) if "delta" in h.lower() or "Δ" in h]

    with path.open("w", encoding="utf-8") as f:
        # Write table header with xcolor package requirement
        f.write(f"% Requires: \\usepackage{{xcolor}}\n")
        f.write(f"\\begin{{table}}[{position}]\n")
        f.write("\\centering\n")
        if small:
            f.write("\\small\n")
        if caption:
            f.write(f"\\caption{{{escape_latex(caption)}}}\n")
        if label:
            f.write(f"\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")

        # Header
        escaped_headers = [escape_latex(h) for h in headers]
        f.write(" & ".join(escaped_headers) + " \\\\\n")
        f.write("\\midrule\n")

        # Data rows
        for row in rows:
            formatted_cells = []
            for col_idx, cell in enumerate(row):
                cell_str = escape_latex(str(cell))

                # Apply coloring to delta columns if cell contains +/- and is numeric
                if col_idx in highlight_cols and cell_str and cell_str not in ["—", ""]:
                    try:
                        # Extract numeric value
                        value_str = cell_str.replace("+", "").replace("\\%", "").strip()
                        value = float(value_str)
                        if value > 0:
                            cell_str = f"\\textcolor{{green!70!black}}{{{cell_str}}}"
                        elif value < 0:
                            cell_str = f"\\textcolor{{red!70!black}}{{{cell_str}}}"
                    except (ValueError, AttributeError):
                        pass

                formatted_cells.append(cell_str)

            f.write(" & ".join(formatted_cells) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def write_dataset_summary_latex(
    path: Path,
    dataset_stage_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    metrics: List[str],
    colored: bool = True,
) -> None:
    """Export dataset summary as LaTeX table with optional conditional coloring.

    Args:
        path: Output file path
        dataset_stage_stats: Nested dict with dataset -> stage -> metric -> statistics
        metrics: List of metrics to include
        colored: Apply conditional coloring to delta values
    """
    headers = [
        "Dataset",
        "Metric",
        "Red Mean",
        "Red Std",
        "Aug Mean",
        "Aug Std",
        "Delta",
        "Δ\\%",
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
                f"{red_stats['mean']:.4f}" if red_stats else "—",
                f"$\\pm${red_stats['std']:.4f}" if red_stats else "—",
                f"{aug_stats['mean']:.4f}" if aug_stats else "—",
                f"$\\pm${aug_stats['std']:.4f}" if aug_stats else "—",
                f"+{delta:.4f}" if delta and delta > 0 else f"{delta:.4f}" if delta else "—",
                f"+{delta_pct:.2f}\\%" if delta_pct and delta_pct > 0 else f"{delta_pct:.2f}\\%" if delta_pct else "—",
            ]
            rows.append(row)

    if colored:
        write_latex_table_colored(
            path,
            headers,
            rows,
            caption="Reduction vs Augmentation Performance Comparison",
            label="tab:reduction_augmentation_comparison",
            small=True,
        )
    else:
        write_latex_table(
            path,
            headers,
            rows,
            caption="Reduction vs Augmentation Performance Comparison",
            label="tab:reduction_augmentation_comparison",
            small=True,
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
    "write_latex_table",
    "write_latex_table_colored",
    "escape_latex",
    "try_write_excel",
    "ensure_dir",
]
