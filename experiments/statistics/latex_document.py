#!/usr/bin/env python3
"""LaTeX document generator for experiment results.

This module creates complete LaTeX documents with tables and figures
from experiment statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from .exporters import escape_latex, ensure_dir


class LaTeXDocument:
    """Builder for creating complete LaTeX documents with tables and figures."""

    def __init__(
        self,
        title: str = "Experiment Results",
        author: str = "DAKGEA Statistics",
        document_class: str = "article",
        packages: List[str] | None = None,
    ):
        """Initialize LaTeX document.

        Args:
            title: Document title
            author: Document author
            document_class: LaTeX document class
            packages: Additional packages to include
        """
        self.title = escape_latex(title)
        self.author = escape_latex(author)
        self.document_class = document_class
        self.sections: List[Dict] = []

        # Default packages
        self.packages = [
            "graphicx",
            "booktabs",
            "xcolor",
            "float",
            "caption",
            "subcaption",
            "geometry",
            "hyperref",
        ]
        if packages:
            self.packages.extend(packages)

    def add_section(self, title: str, level: int = 1) -> None:
        """Add a section to the document.

        Args:
            title: Section title
            level: Section level (1=section, 2=subsection, 3=subsubsection)
        """
        self.sections.append({
            "type": "section",
            "title": escape_latex(title),
            "level": level,
        })

    def add_text(self, text: str) -> None:
        """Add text to the document.

        Args:
            text: Text to add (will be escaped)
        """
        self.sections.append({
            "type": "text",
            "content": text,  # Allow raw LaTeX in text
        })

    def add_table_file(self, table_path: Path, caption: str | None = None) -> None:
        """Include a LaTeX table from file.

        Args:
            table_path: Path to .tex file containing table
            caption: Optional caption override
        """
        self.sections.append({
            "type": "table_file",
            "path": table_path,
            "caption": escape_latex(caption) if caption else None,
        })

    def add_figure(
        self,
        image_path: Path,
        caption: str = "",
        label: str = "",
        width: str = "0.8\\textwidth",
    ) -> None:
        """Add a figure to the document.

        Args:
            image_path: Path to image file (PDF, PNG, etc.)
            caption: Figure caption
            label: Figure label for referencing
            width: Figure width (LaTeX dimension)
        """
        self.sections.append({
            "type": "figure",
            "path": image_path,
            "caption": escape_latex(caption),
            "label": label,
            "width": width,
        })

    def add_subfigures(
        self,
        images: List[Dict[str, str | Path]],
        main_caption: str = "",
        main_label: str = "",
        cols: int = 2,
    ) -> None:
        """Add multiple subfigures in a grid.

        Args:
            images: List of dicts with keys: 'path', 'caption', 'label'
            main_caption: Caption for the entire figure
            main_label: Label for the entire figure
            cols: Number of columns in the grid
        """
        self.sections.append({
            "type": "subfigures",
            "images": images,
            "main_caption": escape_latex(main_caption),
            "main_label": main_label,
            "cols": cols,
        })

    def add_table_inline(
        self,
        headers: List[str],
        rows: List[List[str]],
        caption: str = "",
        label: str = "",
        col_spec: str | None = None,
    ) -> None:
        """Add a table directly to the document.

        Args:
            headers: Column headers
            rows: Data rows
            caption: Table caption
            label: Table label
            col_spec: Column specification (e.g., "lrrr")
        """
        self.sections.append({
            "type": "table_inline",
            "headers": [escape_latex(h) for h in headers],
            "rows": [[escape_latex(str(cell)) for cell in row] for row in rows],
            "caption": escape_latex(caption),
            "label": label,
            "col_spec": col_spec or ("l" + "r" * (len(headers) - 1)),
        })

    def _write_preamble(self, f) -> None:
        """Write document preamble."""
        f.write(f"\\documentclass[11pt,a4paper]{{{self.document_class}}}\n\n")

        # Packages
        for package in self.packages:
            f.write(f"\\usepackage{{{package}}}\n")

        # Geometry settings
        f.write("\\geometry{margin=2.5cm}\n\n")

        # Hyperref setup
        f.write("\\hypersetup{\n")
        f.write("    colorlinks=true,\n")
        f.write("    linkcolor=blue,\n")
        f.write("    urlcolor=blue,\n")
        f.write("    citecolor=blue\n")
        f.write("}\n\n")

        # Title and author
        f.write(f"\\title{{{self.title}}}\n")
        f.write(f"\\author{{{self.author}}}\n")
        f.write("\\date{\\today}\n\n")

    def _write_section(self, f, section: Dict) -> None:
        """Write a single section element."""
        if section["type"] == "section":
            level = section["level"]
            title = section["title"]
            if level == 1:
                f.write(f"\\section{{{title}}}\n\n")
            elif level == 2:
                f.write(f"\\subsection{{{title}}}\n\n")
            elif level == 3:
                f.write(f"\\subsubsection{{{title}}}\n\n")

        elif section["type"] == "text":
            f.write(section["content"] + "\n\n")

        elif section["type"] == "table_file":
            # Include external table file
            table_path = section["path"]
            if table_path.exists():
                f.write(f"% Table from {table_path.name}\n")
                f.write(table_path.read_text(encoding="utf-8"))
                f.write("\n\n")

        elif section["type"] == "table_inline":
            # Write table directly
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\small\n")
            if section["caption"]:
                f.write(f"\\caption{{{section['caption']}}}\n")
            if section["label"]:
                f.write(f"\\label{{{section['label']}}}\n")
            f.write(f"\\begin{{tabular}}{{{section['col_spec']}}}\n")
            f.write("\\toprule\n")
            f.write(" & ".join(section["headers"]) + " \\\\\n")
            f.write("\\midrule\n")
            for row in section["rows"]:
                f.write(" & ".join(row) + " \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")

        elif section["type"] == "figure":
            # Single figure
            f.write("\\begin{figure}[H]\n")
            f.write("\\centering\n")
            # Convert to relative path if needed
            img_path = section["path"]
            if img_path.exists():
                # Use relative path from LaTeX document location
                f.write(f"\\includegraphics[width={section['width']}]{{{img_path.name}}}\n")
                if section["caption"]:
                    f.write(f"\\caption{{{section['caption']}}}\n")
                if section["label"]:
                    f.write(f"\\label{{{section['label']}}}\n")
            f.write("\\end{figure}\n\n")

        elif section["type"] == "subfigures":
            # Multiple subfigures
            cols = section["cols"]
            width = 1.0 / cols - 0.05  # Leave some margin

            f.write("\\begin{figure}[H]\n")
            f.write("\\centering\n")

            for idx, img_info in enumerate(section["images"]):
                img_path = Path(img_info["path"])
                if img_path.exists():
                    f.write(f"\\begin{{subfigure}}{{{width:.2f}\\textwidth}}\n")
                    f.write("\\centering\n")
                    f.write(f"\\includegraphics[width=\\textwidth]{{{img_path.name}}}\n")
                    if img_info.get("caption"):
                        f.write(f"\\caption{{{escape_latex(img_info['caption'])}}}\n")
                    if img_info.get("label"):
                        f.write(f"\\label{{{img_info['label']}}}\n")
                    f.write("\\end{subfigure}\n")

                    # Add line break after every `cols` images
                    if (idx + 1) % cols == 0 and idx < len(section["images"]) - 1:
                        f.write("\n")

            if section["main_caption"]:
                f.write(f"\\caption{{{section['main_caption']}}}\n")
            if section["main_label"]:
                f.write(f"\\label{{{section['main_label']}}}\n")
            f.write("\\end{figure}\n\n")

    def write(self, output_path: Path) -> None:
        """Write the complete LaTeX document to file.

        Args:
            output_path: Path to output .tex file
        """
        ensure_dir(output_path.parent)

        with output_path.open("w", encoding="utf-8") as f:
            # Preamble
            self._write_preamble(f)

            # Begin document
            f.write("\\begin{document}\n\n")
            f.write("\\maketitle\n\n")

            # Table of contents
            if len(self.sections) > 5:  # Only add TOC if doc is large enough
                f.write("\\tableofcontents\n")
                f.write("\\clearpage\n\n")

            # Write all sections
            for section in self.sections:
                self._write_section(f, section)

            # End document
            f.write("\\end{document}\n")


def create_results_document(
    dataset_stage_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    metrics: List[str],
    plots_dir: Path,
    output_path: Path,
    include_figures: bool = True,
) -> None:
    """Create a complete LaTeX document with experiment results.

    Args:
        dataset_stage_stats: Nested dict with statistics
        metrics: List of metrics analyzed
        plots_dir: Directory containing generated plots
        output_path: Path to output .tex file
        include_figures: Whether to include figure plots
    """
    doc = LaTeXDocument(
        title="DAKGEA Experiment Results",
        author="Data Augmentation for Knowledge Graph Entity Alignment",
    )

    # Introduction
    doc.add_section("Overview", level=1)
    doc.add_text(
        "This document presents the statistical analysis of reduction vs augmentation "
        "experiments across multiple datasets and metrics."
    )

    # Summary table
    doc.add_section("Performance Summary", level=1)
    doc.add_text("Table~\\ref{tab:summary} shows the mean performance across all experiments.")

    # Build summary table
    headers = ["Dataset", "Metric", "Reduction", "Augmentation", "Delta", "Δ\\%"]
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
                f"{aug_stats['mean']:.4f}" if aug_stats else "—",
                f"{delta:+.4f}" if delta is not None else "—",
                f"{delta_pct:+.2f}\\%" if delta_pct is not None else "—",
            ]
            rows.append(row)

    doc.add_table_inline(headers, rows, caption="Performance Summary", label="tab:summary")

    # Add figures if requested
    if include_figures and plots_dir.exists():
        doc.add_section("Visualizations", level=1)

        # Group plots by dataset
        for dataset in sorted(dataset_stage_stats.keys()):
            doc.add_section(f"Dataset: {dataset}", level=2)

            # Find all plots for this dataset
            dataset_plots = list(plots_dir.glob(f"{dataset}_*.png"))
            if dataset_plots:
                doc.add_text(f"Figures for {dataset} dataset:")

                for plot_path in sorted(dataset_plots)[:4]:  # Limit to avoid huge docs
                    metric_name = plot_path.stem.replace(f"{dataset}_", "").replace("_", " ")
                    doc.add_figure(
                        plot_path,
                        caption=f"{dataset}: {metric_name}",
                        label=f"fig:{dataset}_{metric_name}",
                        width="0.7\\textwidth",
                    )

    # Write document
    doc.write(output_path)


__all__ = [
    "LaTeXDocument",
    "create_results_document",
]
