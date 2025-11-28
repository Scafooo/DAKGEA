#!/usr/bin/env python3
"""Quality report generator for augmented datasets.

Generates comprehensive quality reports combining diversity, realism,
and sample analysis. Creates both machine-readable (JSON) and
human-readable (Markdown) reports.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from experiments.qualitative_analysis.diversity_metrics import DiversityAnalyzer
from experiments.qualitative_analysis.ea_specific_metrics import EntityAlignmentMetrics
from experiments.qualitative_analysis.entity_sampler import EntitySampler
from experiments.qualitative_analysis.realism_metrics import RealismAnalyzer
from src.core.data_io import load_dataset


class QualityReportGenerator:
    """Generates comprehensive quality reports."""

    def __init__(self):
        """Initialize report generator."""
        self.diversity_analyzer = DiversityAnalyzer()
        self.realism_analyzer = RealismAnalyzer()
        self.ea_analyzer = EntityAlignmentMetrics()
        self.sampler = EntitySampler()

    def generate_report(
        self,
        original_path: Path,
        augmented_path: Path,
        output_dir: Path,
        stage: str = "augmentation",
        n_samples: int = 20,
    ) -> Dict:
        """Generate complete quality report.

        Args:
            original_path: Path to original dataset
            augmented_path: Path to augmented dataset
            output_dir: Directory to save reports
            stage: Stage name ('reduction' or 'augmentation')
            n_samples: Number of samples for human evaluation

        Returns:
            Dictionary with all metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load datasets
        print("Loading datasets...")
        orig_dataset = load_dataset(original_path)
        aug_dataset = load_dataset(augmented_path)

        # Compute metrics
        print("Computing diversity metrics...")
        diversity_metrics = self.diversity_analyzer.analyze(
            orig_dataset, aug_dataset, stage=stage
        )

        print("Computing realism metrics...")
        realism_metrics = self.realism_analyzer.analyze(
            orig_dataset, aug_dataset, stage=stage
        )

        print("Computing EA-specific metrics...")
        ea_metrics = self.ea_analyzer.analyze_all(
            orig_dataset, aug_dataset, stage=stage
        )

        # Extract samples
        print(f"Extracting {n_samples} samples...")
        samples_random = self.sampler.extract_samples(
            orig_dataset, aug_dataset,
            n_samples=n_samples,
            strategy="random",
            stage=stage
        )

        samples_diverse = self.sampler.extract_samples(
            orig_dataset, aug_dataset,
            n_samples=n_samples,
            strategy="diverse",
            stage=stage
        )

        # Combine all data
        report = {
            "metadata": {
                "original_dataset": str(original_path),
                "augmented_dataset": str(augmented_path),
                "stage": stage,
                "generated_at": datetime.now().isoformat(),
            },
            "diversity": diversity_metrics,
            "realism": realism_metrics,
            "entity_alignment": ea_metrics,
            "samples": {
                "random": len(samples_random),
                "diverse": len(samples_diverse),
            }
        }

        # Save JSON report
        json_path = output_dir / "quality_report.json"
        with json_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"✓ Saved JSON report: {json_path}")

        # Save Markdown report
        md_path = output_dir / "quality_report.md"
        self._write_markdown_report(report, md_path)
        print(f"✓ Saved Markdown report: {md_path}")

        # Save samples
        samples_dir = output_dir / "samples"
        samples_dir.mkdir(exist_ok=True)

        self.sampler.export_to_tsv(
            samples_random,
            samples_dir / "random_samples.tsv"
        )
        print(f"✓ Saved random samples: {samples_dir / 'random_samples.tsv'}")

        self.sampler.export_to_tsv(
            samples_diverse,
            samples_dir / "diverse_samples.tsv"
        )
        print(f"✓ Saved diverse samples: {samples_dir / 'diverse_samples.tsv'}")

        return report

    def _write_markdown_report(self, report: Dict, output_path: Path) -> None:
        """Write human-readable Markdown report."""
        with output_path.open("w") as f:
            # Header
            f.write("# DAKGEA Quality Analysis Report\n\n")
            f.write(f"**Generated**: {report['metadata']['generated_at']}\n\n")
            f.write(f"**Original Dataset**: `{report['metadata']['original_dataset']}`\n\n")
            f.write(f"**Augmented Dataset**: `{report['metadata']['augmented_dataset']}`\n\n")
            f.write(f"**Stage**: {report['metadata']['stage']}\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## 📊 Executive Summary\n\n")
            self._write_summary(f, report)
            f.write("\n---\n\n")

            # Diversity Metrics
            f.write("## 🎨 Diversity Metrics\n\n")
            f.write("Diversity measures how varied and non-redundant the generated entities are.\n\n")
            self._write_metrics_table(f, report["diversity"], "Diversity")
            f.write("\n")

            # Realism Metrics
            f.write("## ✨ Realism Metrics\n\n")
            f.write("Realism measures how plausible and well-formed the generated entities are.\n\n")
            self._write_metrics_table(f, report["realism"], "Realism")
            f.write("\n")

            # Entity Alignment Metrics
            f.write("## 🔗 Entity Alignment-Specific Metrics\n\n")
            f.write("EA-specific metrics assess properties critical for Entity Alignment tasks.\n\n")
            self._write_metrics_table(f, report["entity_alignment"], "Entity Alignment")
            f.write("\n")

            # Interpretation Guide
            f.write("## 📖 Interpretation Guide\n\n")
            self._write_interpretation_guide(f)
            f.write("\n")

            # Recommendations
            f.write("## 💡 Recommendations\n\n")
            self._write_recommendations(f, report)
            f.write("\n")

            # Next Steps
            f.write("## 🚀 Next Steps\n\n")
            f.write("1. **Review Samples**: Examine extracted samples in `samples/` directory\n\n")
            f.write("2. **Human Evaluation**: Annotate samples with realism and consistency scores\n\n")
            f.write("3. **Iterate Parameters**: Adjust augmentation parameters based on findings\n\n")
            f.write("4. **Run Experiments**: Test impact on downstream EA models\n\n")

    def _write_summary(self, f, report: Dict) -> None:
        """Write executive summary section."""
        div_metrics = report["diversity"]
        real_metrics = report["realism"]

        f.write("### Key Findings\n\n")

        # Augmentation ratio
        aug_ratio = div_metrics.get("augmentation_ratio", 0)
        f.write(f"- **Augmentation Ratio**: {aug_ratio:.2%} ")
        f.write(f"({div_metrics.get('num_synthetic_entities', 0)} synthetic ")
        f.write(f"/ {div_metrics.get('num_original_entities', 0)} original)\n\n")

        # Novelty
        novelty = div_metrics.get("novelty_ratio", 0)
        f.write(f"- **Novelty**: {novelty:.2%} of synthetic values are new\n\n")

        # Diversity
        emb_div = div_metrics.get("embedding_diversity_synthetic", 0)
        f.write(f"- **Semantic Diversity**: {emb_div:.3f} (avg cosine distance)\n\n")

        # Realism
        fluency = real_metrics.get("fluency_rate", 0)
        f.write(f"- **Fluency**: {fluency:.2%} of text attributes are fluent\n\n")

        # Validity
        empty_rate = real_metrics.get("empty_value_rate", 0)
        f.write(f"- **Empty Values**: {empty_rate:.2%} (should be low)\n\n")

        # Consistency
        consistency = real_metrics.get("alignment_consistency_mean", 0)
        f.write(f"- **Alignment Consistency**: {consistency:.3f} (semantic similarity of aligned pairs)\n\n")

    def _write_metrics_table(self, f, metrics: Dict, category: str) -> None:
        """Write metrics as Markdown table."""
        f.write(f"### {category} Metrics Table\n\n")
        f.write("| Metric | Value | Interpretation |\n")
        f.write("|--------|-------|----------------|\n")

        interpretations = self._get_interpretations()

        for key, value in sorted(metrics.items()):
            if key == "error":
                f.write(f"| **ERROR** | {value} | Check dataset |\n")
                continue

            interp = interpretations.get(key, "")

            if isinstance(value, float):
                f.write(f"| `{key}` | {value:.4f} | {interp} |\n")
            else:
                f.write(f"| `{key}` | {value} | {interp} |\n")

        f.write("\n")

    def _get_interpretations(self) -> Dict[str, str]:
        """Get metric interpretations."""
        return {
            # Diversity
            "novelty_ratio": "Higher = more new values (good for diversity)",
            "embedding_diversity_synthetic": "Higher = more semantically diverse",
            "self_bleu_synthetic": "Lower = more diverse (less repetitive)",
            # Realism
            "fluency_rate": "Higher = better text quality",
            "empty_value_rate": "Lower = better (fewer missing values)",
            "date_validity_rate": "Higher = better formatted dates",
            "number_validity_rate": "Higher = better formatted numbers",
            "semantic_coherence_mean": "Moderate (0.3-0.6) = good diversity within entity",
            "alignment_consistency_mean": "Higher = aligned pairs are more consistent",
            "repetition_rate": "Lower = better (less repetitive text)",
            # EA-specific
            "alignment_preservation_score": "Higher = synthetic pairs remain alignable (target: >0.7)",
            "avg_alignment_similarity": "Similarity of aligned pairs (target: >0.5)",
            "avg_random_similarity": "Similarity to random entities (should be lower)",
            "structural_consistency_score": "Higher = KG structure preserved (target: >0.7)",
            "predicate_overlap_jaccard": "Predicate set overlap (target: >0.6)",
            "kl_divergence": "Distribution difference (lower = better, target: <0.5)",
            "cooccurrence_preservation_score": "Attribute patterns preserved (target: >0.6)",
            "style_consistency_score": "Cross-KG style maintained (target: >0.2)",
            "within_kg_similarity": "Same-KG entities similarity (should be high)",
            "cross_kg_similarity": "Cross-KG entities similarity (should be lower)",
            "nndr_mean": "Diversity/realism balance (target: 0.8-1.2)",
        }

    def _write_interpretation_guide(self, f) -> None:
        """Write guide for interpreting metrics."""
        f.write("### Diversity Metrics\n\n")
        f.write("- **Novelty Ratio**: Percentage of synthetic attribute values that are new (not in originals)\n")
        f.write("  - Target: 40-70% (balance between diversity and realism)\n\n")
        f.write("- **Embedding Diversity**: Average semantic distance between synthetic entities\n")
        f.write("  - Target: 0.3-0.6 (too low = redundant, too high = incoherent)\n\n")
        f.write("- **Self-BLEU**: N-gram overlap within synthetic entities\n")
        f.write("  - Target: <0.3 (lower = more diverse)\n\n")

        f.write("### Realism Metrics\n\n")
        f.write("- **Fluency Rate**: Percentage of text attributes that are well-formed\n")
        f.write("  - Target: >80%\n\n")
        f.write("- **Validity Rates**: Percentage of dates/numbers that are properly formatted\n")
        f.write("  - Target: >90%\n\n")
        f.write("- **Alignment Consistency**: Semantic similarity of aligned pairs\n")
        f.write("  - Target: >0.5 (aligned pairs should be semantically related)\n\n")

        f.write("### Entity Alignment Metrics\n\n")
        f.write("- **Alignment Preservation Score**: Percentage of synthetic pairs that remain correctly alignable\n")
        f.write("  - Target: >70% (synthetic entities should preserve alignability)\n\n")
        f.write("- **Structural Consistency Score**: Degree to which KG structural patterns are preserved\n")
        f.write("  - Target: >70% (entity structure should match originals)\n\n")
        f.write("- **Co-occurrence Preservation**: Consistency of predicate co-occurrence patterns\n")
        f.write("  - Target: >60% (attribute patterns should be maintained)\n\n")
        f.write("- **Style Consistency Score**: Separation between source/target KG styles\n")
        f.write("  - Target: >0.2 (each KG should maintain distinct style)\n\n")
        f.write("- **NNDR (Nearest Neighbor Distance Ratio)**: Balance between diversity and realism\n")
        f.write("  - Target: 0.8-1.2 (not too similar to originals, not too different)\n\n")

    def _write_recommendations(self, f, report: Dict) -> None:
        """Write recommendations based on metrics."""
        div_metrics = report["diversity"]
        real_metrics = report["realism"]

        novelty = div_metrics.get("novelty_ratio", 0)
        fluency = real_metrics.get("fluency_rate", 0)
        empty_rate = real_metrics.get("empty_value_rate", 0)
        consistency = real_metrics.get("alignment_consistency_mean", 0)

        issues = []
        suggestions = []

        # Check novelty
        if novelty < 0.3:
            issues.append("⚠️ **Low Novelty**: Synthetic entities are too similar to originals")
            suggestions.append("Increase `base_alpha` or `alpha_spread` in BART config")
        elif novelty > 0.8:
            issues.append("⚠️ **Very High Novelty**: Synthetic entities might be unrealistic")
            suggestions.append("Decrease `base_alpha` or reduce `temperature`")

        # Check fluency
        if fluency < 0.7:
            issues.append("⚠️ **Low Fluency**: Generated text quality is poor")
            suggestions.append("Lower `temperature` (e.g., 0.8-1.0) for more conservative generation")
            suggestions.append("Ensure BART fine-tuning is enabled and has sufficient training data")

        # Check empty values
        if empty_rate > 0.1:
            issues.append("⚠️ **High Empty Rate**: Many attributes have missing values")
            suggestions.append("Check attribute matching configuration")
            suggestions.append("Enable `generate_unmatched` for unmatched attributes")

        # Check consistency
        if consistency < 0.4:
            issues.append("⚠️ **Low Consistency**: Aligned pairs are not semantically consistent")
            suggestions.append("Review predicate matching (check `match_attr` files)")
            suggestions.append("Adjust interpolation parameters for better alignment preservation")

        if not issues:
            f.write("✅ **All metrics are within acceptable ranges!**\n\n")
            f.write("The augmented dataset shows good balance between diversity and realism.\n\n")
        else:
            f.write("### Issues Detected\n\n")
            for issue in issues:
                f.write(f"{issue}\n\n")

            f.write("### Suggested Actions\n\n")
            for i, suggestion in enumerate(suggestions, 1):
                f.write(f"{i}. {suggestion}\n\n")


def generate_quality_report(
    original_path: Path,
    augmented_path: Path,
    output_dir: Path,
    stage: str = "augmentation",
    n_samples: int = 20,
) -> Dict:
    """Generate quality report (convenience function).

    Args:
        original_path: Path to original dataset
        augmented_path: Path to augmented dataset
        output_dir: Directory to save reports
        stage: Stage name
        n_samples: Number of samples for evaluation

    Returns:
        Report dictionary
    """
    generator = QualityReportGenerator()
    return generator.generate_report(
        original_path,
        augmented_path,
        output_dir,
        stage=stage,
        n_samples=n_samples
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate quality report for augmented dataset")
    parser.add_argument("--original", type=str, required=True, help="Path to original dataset")
    parser.add_argument("--augmented", type=str, required=True, help="Path to augmented dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save reports")
    parser.add_argument("--stage", type=str, default="augmentation", choices=["reduction", "augmentation"])
    parser.add_argument("--n-samples", type=int, default=20, help="Number of samples for evaluation")

    args = parser.parse_args()

    print("=" * 80)
    print("DAKGEA Quality Analysis")
    print("=" * 80)

    report = generate_quality_report(
        Path(args.original),
        Path(args.augmented),
        Path(args.output_dir),
        stage=args.stage,
        n_samples=args.n_samples
    )

    print("\n" + "=" * 80)
    print("✅ Quality report generated successfully!")
    print("=" * 80)
    print(f"\nView reports in: {args.output_dir}")
    print(f"  - quality_report.md (human-readable)")
    print(f"  - quality_report.json (machine-readable)")
    print(f"  - samples/ (for human evaluation)")
