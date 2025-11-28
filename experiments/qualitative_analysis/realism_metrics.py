#!/usr/bin/env python3
"""Realism metrics for evaluating quality of generated entities.

This module provides metrics to assess how "realistic" or "plausible" the
generated entities are. High realism means entities look like they could
be real entries in the knowledge graph.

Metrics:
    1. Semantic Coherence: Do attributes make sense together?
    2. Attribute Validity: Are values well-formed (dates, numbers, etc.)?
    3. Fluency: Is the text grammatically correct?
    4. Consistency: Do aligned pairs have consistent semantics?
    5. Hallucination Detection: Do entities contain factually wrong info?
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.data_structures import Dataset, Entity, KnowledgeGraph


class RealismAnalyzer:
    """Analyzes realism of generated entities."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize realism analyzer.

        Args:
            embedding_model: Sentence transformer for semantic analysis
        """
        self.embedding_model_name = embedding_model
        self.encoder = None  # Lazy loading

    def analyze(
        self,
        original_dataset: Dataset,
        augmented_dataset: Dataset,
        stage: str = "augmentation"
    ) -> Dict[str, float]:
        """Compute all realism metrics.

        Args:
            original_dataset: Original dataset before augmentation
            augmented_dataset: Dataset after augmentation
            stage: Stage name ('reduction' or 'augmentation')

        Returns:
            Dictionary of metric_name -> value
        """
        # Extract KGs based on stage
        if stage == "augmentation":
            orig_kg = original_dataset.kg1
            aug_kg = augmented_dataset.kg1
            paired_kg = augmented_dataset.kg2
        else:
            orig_kg = original_dataset.kg2
            aug_kg = augmented_dataset.kg2
            paired_kg = augmented_dataset.kg1

        # Identify synthetic entities
        orig_uris = set(orig_kg.entities.keys())
        aug_uris = set(aug_kg.entities.keys())
        synthetic_uris = aug_uris - orig_uris

        if not synthetic_uris:
            return {"error": "No synthetic entities found"}

        metrics = {}

        # 1. Attribute Validity
        metrics.update(self._compute_attribute_validity(aug_kg, synthetic_uris))

        # 2. Fluency (text quality)
        metrics.update(self._compute_fluency(aug_kg, synthetic_uris))

        # 3. Semantic Coherence
        metrics.update(self._compute_semantic_coherence(aug_kg, synthetic_uris))

        # 4. Alignment Consistency (if pairs available)
        metrics.update(self._compute_alignment_consistency(
            augmented_dataset, aug_kg, paired_kg, synthetic_uris
        ))

        # 5. Length Statistics (sanity check)
        metrics.update(self._compute_length_statistics(orig_kg, aug_kg, orig_uris, synthetic_uris))

        # Summary
        metrics["num_synthetic_entities_analyzed"] = len(synthetic_uris)

        return metrics

    def _compute_attribute_validity(
        self,
        kg: KnowledgeGraph,
        entity_uris: Set[str]
    ) -> Dict[str, float]:
        """Check if attribute values are well-formed.

        Checks:
            - Dates are valid
            - Numbers are numeric
            - No empty/null values
            - No excessive repetition
        """
        total_attrs = 0
        valid_dates = 0
        invalid_dates = 0
        valid_numbers = 0
        invalid_numbers = 0
        empty_values = 0
        repetitive_values = 0

        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY or MM/DD/YYYY
            r'\d{4}',              # YYYY
        ]

        for uri in entity_uris:
            entity = kg.entities.get(uri)
            if not entity:
                continue

            for attr in entity.attributes:
                total_attrs += 1
                value = str(attr.value) if attr.value else ""

                # Check for empty
                if not value or value.strip() == "":
                    empty_values += 1
                    continue

                # Check for dates (if predicate suggests it's a date)
                pred_lower = str(attr.predicate).lower()
                if any(k in pred_lower for k in ["date", "year", "birth", "death", "time"]):
                    is_valid_date = any(re.search(pattern, value) for pattern in date_patterns)
                    if is_valid_date:
                        valid_dates += 1
                    else:
                        invalid_dates += 1

                # Check for numbers (if predicate suggests numeric)
                if any(k in pred_lower for k in ["count", "number", "population", "age", "year"]):
                    try:
                        # Remove commas and parse
                        float(value.replace(",", ""))
                        valid_numbers += 1
                    except ValueError:
                        invalid_numbers += 1

                # Check for repetition (e.g., "Paris Paris Paris")
                words = value.split()
                if len(words) > 2:
                    unique_words = set(words)
                    if len(unique_words) / len(words) < 0.5:  # More than 50% repetition
                        repetitive_values += 1

        date_checks = valid_dates + invalid_dates
        number_checks = valid_numbers + invalid_numbers

        return {
            "total_attributes": total_attrs,
            "empty_value_rate": empty_values / total_attrs if total_attrs > 0 else 0.0,
            "date_validity_rate": valid_dates / date_checks if date_checks > 0 else 1.0,
            "number_validity_rate": valid_numbers / number_checks if number_checks > 0 else 1.0,
            "repetition_rate": repetitive_values / total_attrs if total_attrs > 0 else 0.0,
        }

    def _compute_fluency(
        self,
        kg: KnowledgeGraph,
        entity_uris: Set[str]
    ) -> Dict[str, float]:
        """Measure text fluency of attribute values.

        Uses heuristics:
            - No excessive punctuation
            - Proper capitalization
            - No truncated words
            - Reasonable sentence length
        """
        total_text_attrs = 0
        fluent_count = 0

        for uri in entity_uris:
            entity = kg.entities.get(uri)
            if not entity:
                continue

            for attr in entity.attributes:
                value = str(attr.value) if attr.value else ""
                if not value or len(value) < 5:
                    continue

                # Only analyze text attributes (not numbers/dates)
                if not any(c.isalpha() for c in value):
                    continue

                total_text_attrs += 1
                is_fluent = True

                # Check 1: No excessive punctuation (>20% of chars)
                punct_count = sum(1 for c in value if c in ".,;:!?")
                if punct_count / len(value) > 0.2:
                    is_fluent = False

                # Check 2: First letter capitalized (if it's a sentence)
                if len(value) > 20 and value[0].islower():
                    is_fluent = False

                # Check 3: No truncated words (ends with incomplete word)
                if value.endswith("-") or value.endswith("..."):
                    is_fluent = False

                # Check 4: Reasonable length (not too short or too long)
                words = value.split()
                if len(words) > 100:  # Too long
                    is_fluent = False

                if is_fluent:
                    fluent_count += 1

        return {
            "text_attributes_count": total_text_attrs,
            "fluency_rate": fluent_count / total_text_attrs if total_text_attrs > 0 else 1.0,
        }

    def _compute_semantic_coherence(
        self,
        kg: KnowledgeGraph,
        entity_uris: Set[str]
    ) -> Dict[str, float]:
        """Measure semantic coherence within each entity.

        For each entity, check if attributes are semantically related.
        High coherence = attributes "make sense" together.
        """
        # Lazy load encoder
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.embedding_model_name)

        coherence_scores = []

        # Sample entities for performance
        sampled_uris = list(entity_uris)
        if len(sampled_uris) > 100:
            sampled_uris = np.random.choice(sampled_uris, 100, replace=False).tolist()

        for uri in sampled_uris:
            entity = kg.entities.get(uri)
            if not entity or len(entity.attributes) < 2:
                continue

            # Extract text values
            text_values = []
            for attr in entity.attributes:
                value = str(attr.value) if attr.value else ""
                if value and len(value) > 3 and any(c.isalpha() for c in value):
                    text_values.append(value)

            if len(text_values) < 2:
                continue

            # Encode all values
            embeddings = self.encoder.encode(text_values, convert_to_tensor=False, show_progress_bar=False)

            # Compute pairwise cosine similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(sim)

            if similarities:
                coherence_scores.append(np.mean(similarities))

        return {
            "semantic_coherence_mean": float(np.mean(coherence_scores)) if coherence_scores else 0.0,
            "semantic_coherence_std": float(np.std(coherence_scores)) if coherence_scores else 0.0,
        }

    def _compute_alignment_consistency(
        self,
        dataset: Dataset,
        kg1: KnowledgeGraph,
        kg2: KnowledgeGraph,
        synthetic_uris: Set[str]
    ) -> Dict[str, float]:
        """Check if aligned pairs are semantically consistent.

        For synthetic aligned pairs (src_synth, tgt_synth), check if
        their attributes are semantically similar.
        """
        # Lazy load encoder
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.embedding_model_name)

        # Find aligned pairs that include synthetic entities
        synthetic_pairs = []
        for src_uri, tgt_uri in dataset.alignment_pairs:
            if src_uri in synthetic_uris:
                synthetic_pairs.append((src_uri, tgt_uri))

        if not synthetic_pairs:
            return {
                "alignment_consistency_mean": 0.0,
                "synthetic_aligned_pairs": 0,
            }

        # Sample for performance
        if len(synthetic_pairs) > 50:
            synthetic_pairs = [synthetic_pairs[i] for i in np.random.choice(
                len(synthetic_pairs), 50, replace=False
            )]

        consistency_scores = []

        for src_uri, tgt_uri in synthetic_pairs:
            src_entity = kg1.entities.get(src_uri)
            tgt_entity = kg2.entities.get(tgt_uri)

            if not src_entity or not tgt_entity:
                continue

            # Extract text values from both entities
            src_texts = [str(attr.value) for attr in src_entity.attributes if attr.value]
            tgt_texts = [str(attr.value) for attr in tgt_entity.attributes if attr.value]

            if not src_texts or not tgt_texts:
                continue

            # Compute cross-similarity
            src_embs = self.encoder.encode(src_texts, convert_to_tensor=False, show_progress_bar=False)
            tgt_embs = self.encoder.encode(tgt_texts, convert_to_tensor=False, show_progress_bar=False)

            # Average max similarity (for each src, find best match in tgt)
            max_sims = []
            for src_emb in src_embs:
                sims = [
                    np.dot(src_emb, tgt_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(tgt_emb))
                    for tgt_emb in tgt_embs
                ]
                max_sims.append(max(sims))

            consistency_scores.append(np.mean(max_sims))

        return {
            "alignment_consistency_mean": float(np.mean(consistency_scores)) if consistency_scores else 0.0,
            "alignment_consistency_std": float(np.std(consistency_scores)) if consistency_scores else 0.0,
            "synthetic_aligned_pairs": len(synthetic_pairs),
        }

    def _compute_length_statistics(
        self,
        orig_kg: KnowledgeGraph,
        aug_kg: KnowledgeGraph,
        orig_uris: Set[str],
        synth_uris: Set[str]
    ) -> Dict[str, float]:
        """Compare length statistics between original and synthetic.

        Synthetic entities should have similar length distributions
        to originals (not too short, not too long).
        """
        def get_avg_lengths(kg: KnowledgeGraph, uris: Set[str]) -> List[float]:
            lengths = []
            for uri in uris:
                entity = kg.entities.get(uri)
                if not entity:
                    continue

                total_length = sum(
                    len(str(attr.value)) if attr.value else 0
                    for attr in entity.attributes
                )
                lengths.append(total_length)
            return lengths

        orig_lengths = get_avg_lengths(orig_kg, orig_uris)
        synth_lengths = get_avg_lengths(aug_kg, synth_uris)

        return {
            "avg_entity_length_original": float(np.mean(orig_lengths)) if orig_lengths else 0.0,
            "avg_entity_length_synthetic": float(np.mean(synth_lengths)) if synth_lengths else 0.0,
            "std_entity_length_original": float(np.std(orig_lengths)) if orig_lengths else 0.0,
            "std_entity_length_synthetic": float(np.std(synth_lengths)) if synth_lengths else 0.0,
        }


def analyze_realism(
    original_path: Path,
    augmented_path: Path,
    output_path: Path = None,
    stage: str = "augmentation"
) -> Dict[str, float]:
    """Analyze realism of augmented dataset.

    Args:
        original_path: Path to original dataset
        augmented_path: Path to augmented dataset
        output_path: Optional path to save metrics JSON
        stage: Stage name ('reduction' or 'augmentation')

    Returns:
        Dictionary of realism metrics
    """
    from src.core.data_io import load_dataset

    # Load datasets
    orig_dataset = load_dataset(original_path)
    aug_dataset = load_dataset(augmented_path)

    # Analyze
    analyzer = RealismAnalyzer()
    metrics = analyzer.analyze(orig_dataset, aug_dataset, stage=stage)

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze realism of augmented entities")
    parser.add_argument("--original", type=str, required=True, help="Path to original dataset")
    parser.add_argument("--augmented", type=str, required=True, help="Path to augmented dataset")
    parser.add_argument("--output", type=str, help="Path to save metrics JSON")
    parser.add_argument("--stage", type=str, default="augmentation", choices=["reduction", "augmentation"])

    args = parser.parse_args()

    metrics = analyze_realism(
        Path(args.original),
        Path(args.augmented),
        Path(args.output) if args.output else None,
        stage=args.stage
    )

    print("\n=== Realism Metrics ===")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"{key:40s}: {value:.4f}")
        else:
            print(f"{key:40s}: {value}")
