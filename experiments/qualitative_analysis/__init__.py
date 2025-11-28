"""Qualitative analysis tools for DAKGEA augmented datasets.

This package provides comprehensive tools for analyzing the quality
of generated entities from PLM augmentation.

Modules:
    - diversity_metrics: Measures diversity of generated entities
    - realism_metrics: Evaluates realism and quality
    - entity_sampler: Extracts samples for human evaluation
    - quality_report: Generates comprehensive reports

Quick Start:
    >>> from experiments.qualitative_analysis import generate_quality_report
    >>> report = generate_quality_report(
    ...     original_path="data/original",
    ...     augmented_path="results/experiment/augmentation",
    ...     output_dir="results/quality_analysis"
    ... )
"""

from experiments.qualitative_analysis.diversity_metrics import (
    DiversityAnalyzer,
    analyze_diversity,
)
from experiments.qualitative_analysis.ea_specific_metrics import (
    EntityAlignmentMetrics,
    analyze_ea_metrics,
)
from experiments.qualitative_analysis.entity_sampler import (
    EntitySampler,
    sample_entities,
)
from experiments.qualitative_analysis.quality_report import (
    QualityReportGenerator,
    generate_quality_report,
)
from experiments.qualitative_analysis.realism_metrics import (
    RealismAnalyzer,
    analyze_realism,
)

__all__ = [
    # Analyzers
    "DiversityAnalyzer",
    "RealismAnalyzer",
    "EntityAlignmentMetrics",
    "EntitySampler",
    "QualityReportGenerator",
    # Convenience functions
    "analyze_diversity",
    "analyze_realism",
    "analyze_ea_metrics",
    "sample_entities",
    "generate_quality_report",
]

__version__ = "1.0.0"
__author__ = "DAKGEA Team"
