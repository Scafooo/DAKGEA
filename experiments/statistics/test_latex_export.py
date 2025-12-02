#!/usr/bin/env python3
"""Test script for LaTeX export functionality."""

from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.statistics.exporters import (
    write_latex_table,
    write_latex_table_colored,
    write_dataset_summary_latex,
)
from experiments.statistics.latex_document import LaTeXDocument


def test_basic_table():
    """Test basic LaTeX table export."""
    print("Testing basic LaTeX table...")

    headers = ["Dataset", "Metric", "Reduction", "Augmentation", "Delta"]
    rows = [
        ["BBC_DB", "hits@1", "0.8542", "0.8921", "+0.0379"],
        ["EN_DE_15K", "hits@1", "0.7234", "0.7812", "+0.0578"],
        ["BBC_DB", "mrr", "0.8901", "0.9123", "+0.0222"],
    ]

    output_path = Path("/tmp/test_basic_table.tex")
    write_latex_table(
        output_path,
        headers,
        rows,
        caption="Test Performance Comparison",
        label="tab:test_basic",
    )

    print(f"✓ Basic table written to {output_path}")
    print("\nTable content:")
    print(output_path.read_text())
    print()


def test_colored_table():
    """Test colored LaTeX table with delta highlighting."""
    print("Testing colored LaTeX table...")

    headers = ["Dataset", "Metric", "Reduction", "Augmentation", "Delta", "Δ%"]
    rows = [
        ["BBC_DB", "hits@1", "0.8542", "0.8921", "+0.0379", "+4.44%"],
        ["EN_DE_15K", "hits@1", "0.7234", "0.7812", "+0.0578", "+7.99%"],
        ["Test_DB", "hits@1", "0.9000", "0.8800", "-0.0200", "-2.22%"],
    ]

    output_path = Path("/tmp/test_colored_table.tex")
    write_latex_table_colored(
        output_path,
        headers,
        rows,
        caption="Performance with Conditional Coloring",
        label="tab:test_colored",
    )

    print(f"✓ Colored table written to {output_path}")
    print("\nTable content (with coloring):")
    print(output_path.read_text())
    print()


def test_latex_document():
    """Test complete LaTeX document generation."""
    print("Testing complete LaTeX document...")

    doc = LaTeXDocument(
        title="Test DAKGEA Results",
        author="LaTeX Export Test"
    )

    # Add introduction
    doc.add_section("Introduction", level=1)
    doc.add_text(
        "This is a test document demonstrating the LaTeX export functionality "
        "of the DAKGEA statistics module."
    )

    # Add a table
    doc.add_section("Results", level=1)
    doc.add_text("Table~\\ref{tab:results} shows the test results.")

    doc.add_table_inline(
        headers=["Dataset", "Metric", "Value"],
        rows=[
            ["BBC_DB", "hits@1", "0.8542"],
            ["BBC_DB", "hits@5", "0.9234"],
            ["BBC_DB", "mrr", "0.8901"],
        ],
        caption="Test Results Summary",
        label="tab:results",
    )

    # Add methodology section
    doc.add_section("Methodology", level=1)
    doc.add_text(
        "The experiments were conducted using reduction ratios ranging from "
        "0.1 to 0.9 and augmentation ratios from 0.5 to 2.0."
    )

    output_path = Path("/tmp/test_document.tex")
    doc.write(output_path)

    print(f"✓ Complete document written to {output_path}")
    print("\nTo compile:")
    print(f"  cd {output_path.parent}")
    print(f"  pdflatex {output_path.name}")
    print()


def test_dataset_summary():
    """Test dataset summary LaTeX export."""
    print("Testing dataset summary export...")

    # Mock data structure
    dataset_stage_stats = {
        "BBC_DB": {
            "reduction": {
                "hits@1": {"mean": 0.8542, "std": 0.0234, "min": 0.82, "max": 0.89, "count": 10},
                "mrr": {"mean": 0.8901, "std": 0.0189, "min": 0.86, "max": 0.92, "count": 10},
            },
            "augmentation": {
                "hits@1": {"mean": 0.8921, "std": 0.0198, "min": 0.85, "max": 0.93, "count": 10},
                "mrr": {"mean": 0.9123, "std": 0.0156, "min": 0.88, "max": 0.94, "count": 10},
            },
        },
        "EN_DE_15K": {
            "reduction": {
                "hits@1": {"mean": 0.7234, "std": 0.0312, "min": 0.68, "max": 0.76, "count": 8},
            },
            "augmentation": {
                "hits@1": {"mean": 0.7812, "std": 0.0278, "min": 0.73, "max": 0.82, "count": 8},
            },
        },
    }

    metrics = ["hits@1", "mrr"]
    output_path = Path("/tmp/test_dataset_summary.tex")

    write_dataset_summary_latex(
        output_path,
        dataset_stage_stats,
        metrics,
        colored=True,
    )

    print(f"✓ Dataset summary written to {output_path}")
    print("\nTable preview (first 30 lines):")
    lines = output_path.read_text().split("\n")
    print("\n".join(lines[:30]))
    if len(lines) > 30:
        print(f"... ({len(lines) - 30} more lines)")
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("LaTeX Export Test Suite")
    print("=" * 60)
    print()

    try:
        test_basic_table()
        test_colored_table()
        test_dataset_summary()
        test_latex_document()

        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print()
        print("Output files in /tmp:")
        print("  - test_basic_table.tex")
        print("  - test_colored_table.tex")
        print("  - test_dataset_summary.tex")
        print("  - test_document.tex")
        print()
        print("To compile any document:")
        print("  cd /tmp")
        print("  pdflatex <filename>.tex")
        print()

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
