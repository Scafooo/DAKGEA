#!/usr/bin/env python3
"""Realism metrics for analyzing generated entities.

This module evaluates how realistic and plausible the generated entities are.
High realism indicates that augmented entities look like they could be real.

Metrics:
    1. Attribute Validity: Proper formatting of dates, numbers, etc.
    2. Fluency: Text quality and grammaticality
    3. Semantic Coherence: Attributes make sense together
    4. Alignment Consistency: Paired entities are semantically related
    5. Length Statistics: Value length distributions
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from rdflib import Literal, URIRef
from sentence_transformers import SentenceTransformer

from src.core import Dataset, KnowledgeGraph


class RealismAnalyzer:
    """Analyzes realism and quality of generated entities."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize realism analyzer.

        Args:
            embedding_model: Sentence transformer model for semantic analysis
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
            original_dataset: Original dataset
            augmented_dataset: Augmented dataset
            stage: Stage name ('reduction' or 'augmentation')

        Returns:
            Dictionary of metric_name -> value
        """
        # Extract appropriate KGs
        if stage == "augmentation":
            orig_kg = original_dataset.knowledge_graph_source
            aug_kg = augmented_dataset.knowledge_graph_source
        else:
            orig_kg = original_dataset.knowledge_graph_target
            aug_kg = augmented_dataset.knowledge_graph_target

        # Get entity URIs
        orig_uris = self._get_entity_uris(orig_kg)
        aug_uris = self._get_entity_uris(aug_kg)
        synthetic_uris = aug_uris - orig_uris

        if not synthetic_uris:
            return {"error": "No synthetic entities found"}

        metrics = {}

        # 1. Attribute Validity
        metrics.update(self._compute_attribute_validity(aug_kg, synthetic_uris))

        # 2. Fluency (text quality)
        metrics.update(self._compute_fluency(aug_kg, synthetic_uris))

        # 3. Semantic Coherence (attributes within entity)
        metrics.update(self._compute_semantic_coherence(aug_kg, synthetic_uris))

        # 4. Alignment Consistency (paired entities)
        metrics.update(self._compute_alignment_consistency(
            augmented_dataset, aug_kg, stage
        ))

        # 5. Length Statistics
        metrics.update(self._compute_length_statistics(orig_kg, aug_kg, orig_uris, synthetic_uris))

        return metrics

    def _get_entity_uris(self, kg: KnowledgeGraph) -> Set[str]:
        """Extract all entity URIs from knowledge graph."""
        entities = set()
        for s, p, o in kg.triples((None, None, None)):
            if isinstance(s, URIRef):
                entities.add(str(s))
        return entities

    def _compute_attribute_validity(
        self,
        kg: KnowledgeGraph,
        entity_uris: Set[str]
    ) -> Dict[str, float]:
        """Check validity of attribute values (dates, numbers, etc.)."""
        metrics = {}

        # Patterns
        date_pattern = re.compile(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}')
        number_pattern = re.compile(r'^-?\d+(\.\d+)?$')

        date_checks = 0
        valid_dates = 0
        number_checks = 0
        valid_numbers = 0
        empty_values = 0
        total_values = 0
        repetitive_values = 0

        for uri in entity_uris:
            values_in_entity = []
            for s, p, o in kg.triples((URIRef(uri), None, None)):
                if isinstance(o, Literal):
                    value = str(o).strip()
                    total_values += 1
                    values_in_entity.append(value)

                    if not value:
                        empty_values += 1
                        continue

                    # Check dates
                    if 'date' in str(p).lower() or 'time' in str(p).lower():
                        date_checks += 1
                        if date_pattern.search(value):
                            valid_dates += 1

                    # Check numbers
                    if any(x in str(p).lower() for x in ['count', 'number', 'age', 'year']):
                        number_checks += 1
                        if number_pattern.match(value):
                            valid_numbers += 1

            # Check repetition within entity
            if len(values_in_entity) > 1 and len(set(values_in_entity)) < len(values_in_entity):
                repetitive_values += len(values_in_entity) - len(set(values_in_entity))

        metrics["date_validity_rate"] = valid_dates / date_checks if date_checks > 0 else 1.0
        metrics["number_validity_rate"] = valid_numbers / number_checks if number_checks > 0 else 1.0
        metrics["empty_value_rate"] = empty_values / total_values if total_values > 0 else 0.0
        metrics["repetition_rate"] = repetitive_values / total_values if total_values > 0 else 0.0

        return metrics

    def _compute_fluency(
        self,
        kg: KnowledgeGraph,
        entity_uris: Set[str]
    ) -> Dict[str, float]:
        """Estimate text fluency using heuristics."""
        fluent_count = 0
        total_text_attrs = 0

        for uri in entity_uris:
            for s, p, o in kg.triples((URIRef(uri), None, None)):
                if isinstance(o, Literal):
                    value = str(o).strip()

                    # Only check text attributes (not numbers/dates)
                    if not value or value.replace('.', '').replace('-', '').isdigit():
                        continue

                    total_text_attrs += 1

                    # Heuristic checks for fluency
                    is_fluent = True

                    # 1. Not too short
                    if len(value) < 2:
                        is_fluent = False

                    # 2. Has reasonable characters
                    if not any(c.isalnum() for c in value):
                        is_fluent = False

                    # 3. Not excessive punctuation
                    punct_ratio = sum(1 for c in value if not c.isalnum() and not c.isspace()) / len(value)
                    if punct_ratio > 0.5:
                        is_fluent = False

                    # 4. No excessive repetition (e.g., "aaaa" or "the the the")
                    words = value.lower().split()
                    if len(words) > 1:
                        unique_ratio = len(set(words)) / len(words)
                        if unique_ratio < 0.5:
                            is_fluent = False

                    if is_fluent:
                        fluent_count += 1

        return {
            "fluency_rate": fluent_count / total_text_attrs if total_text_attrs > 0 else 0.0,
            "total_text_attributes": total_text_attrs,
        }

    def _compute_semantic_coherence(
        self,
        kg: KnowledgeGraph,
        entity_uris: Set[str]
    ) -> Dict[str, float]:
        """Check if attributes within an entity are semantically coherent."""
        # Lazy load encoder
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.embedding_model_name)

        coherence_scores = []

        # Sample entities for performance
        sampled_uris = list(entity_uris)
        if len(sampled_uris) > 100:
            sampled_uris = np.random.choice(sampled_uris, 100, replace=False).tolist()

        for uri in sampled_uris:
            # Get all text values for this entity
            values = []
            for s, p, o in kg.triples((URIRef(uri), None, None)):
                if isinstance(o, Literal):
                    value = str(o).strip()
                    if value and len(value) > 2:
                        values.append(value)

            if len(values) < 2:
                continue

            # Encode values
            embeddings = self.encoder.encode(values, convert_to_tensor=False, show_progress_bar=False)

            # Compute pairwise similarities
            sims = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    sims.append(sim)

            if sims:
                coherence_scores.append(np.mean(sims))

        return {
            "semantic_coherence_mean": float(np.mean(coherence_scores)) if coherence_scores else 0.0,
            "semantic_coherence_std": float(np.std(coherence_scores)) if coherence_scores else 0.0,
        }

    def _compute_alignment_consistency(
        self,
        dataset: Dataset,
        aug_kg: KnowledgeGraph,
        stage: str
    ) -> Dict[str, float]:
        """Check if aligned entity pairs are semantically consistent."""
        # Lazy load encoder
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.embedding_model_name)

        similarities = []

        # Sample aligned pairs
        aligned_pairs = list(dataset.aligned_entities)
        if len(aligned_pairs) > 100:
            aligned_pairs = np.random.choice(len(aligned_pairs), 100, replace=False)
            aligned_pairs = [dataset.aligned_entities[i] for i in aligned_pairs]

        for src_uri, tgt_uri in aligned_pairs:
            # Get text representation of source entity
            src_values = []
            for s, p, o in dataset.knowledge_graph_source.triples((URIRef(src_uri), None, None)):
                if isinstance(o, Literal):
                    src_values.append(str(o))

            # Get text representation of target entity
            tgt_values = []
            for s, p, o in dataset.knowledge_graph_target.triples((URIRef(tgt_uri), None, None)):
                if isinstance(o, Literal):
                    tgt_values.append(str(o))

            if not src_values or not tgt_values:
                continue

            # Combine into text
            src_text = " ".join(src_values[:10])  # Limit for performance
            tgt_text = " ".join(tgt_values[:10])

            # Encode and compare
            embeddings = self.encoder.encode([src_text, tgt_text], convert_to_tensor=False, show_progress_bar=False)
            sim = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            similarities.append(sim)

        return {
            "alignment_consistency_mean": float(np.mean(similarities)) if similarities else 0.0,
            "alignment_consistency_std": float(np.std(similarities)) if similarities else 0.0,
        }

    def _compute_length_statistics(
        self,
        orig_kg: KnowledgeGraph,
        aug_kg: KnowledgeGraph,
        orig_uris: Set[str],
        synth_uris: Set[str]
    ) -> Dict[str, float]:
        """Compare length distributions of attribute values."""
        def get_lengths(kg: KnowledgeGraph, uris: Set[str]) -> List[int]:
            lengths = []
            for uri in uris:
                for s, p, o in kg.triples((URIRef(uri), None, None)):
                    if isinstance(o, Literal):
                        lengths.append(len(str(o)))
            return lengths

        orig_lengths = get_lengths(orig_kg, orig_uris)
        synth_lengths = get_lengths(aug_kg, synth_uris)

        return {
            "avg_value_length_original": float(np.mean(orig_lengths)) if orig_lengths else 0.0,
            "avg_value_length_synthetic": float(np.mean(synth_lengths)) if synth_lengths else 0.0,
            "std_value_length_original": float(np.std(orig_lengths)) if orig_lengths else 0.0,
            "std_value_length_synthetic": float(np.std(synth_lengths)) if synth_lengths else 0.0,
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
        stage: Stage name

    Returns:
        Dictionary of realism metrics
    """
    from src.core import DatasetReaderFactory

    # Load datasets
    reader = DatasetReaderFactory.create_reader("openea")
    orig_dataset = reader.read(original_path)
    aug_dataset = reader.read(augmented_path)

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
    parser.add_argument("--original", type=str, required=True)
    parser.add_argument("--augmented", type=str, required=True)
    parser.add_argument("--output", type=str)
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
