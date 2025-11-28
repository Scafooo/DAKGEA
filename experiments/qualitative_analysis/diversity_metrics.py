#!/usr/bin/env python3
"""Diversity metrics for analyzing generated entities.

This module provides metrics to measure how diverse the generated entities are
compared to the original dataset. High diversity indicates the augmentation
is creating varied, non-redundant entities.

Metrics:
    1. Attribute Value Diversity: Unique values per attribute
    2. Lexical Diversity (TTR): Type-Token Ratio for text attributes
    3. Embedding Diversity: Cosine distance in semantic space
    4. Self-BLEU: Measures n-gram overlap (lower = more diverse)
    5. Structural Diversity: Variance in attribute patterns
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from rdflib import Literal, URIRef
from sentence_transformers import SentenceTransformer

from src.core import Dataset, KnowledgeGraph


class DiversityAnalyzer:
    """Analyzes diversity of generated entities compared to originals."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize diversity analyzer.

        Args:
            embedding_model: Sentence transformer model for semantic diversity
        """
        self.embedding_model_name = embedding_model
        self.encoder = None  # Lazy loading

    def analyze(
        self,
        original_dataset: Dataset,
        augmented_dataset: Dataset,
        stage: str = "augmentation"
    ) -> Dict[str, float]:
        """Compute all diversity metrics.

        Args:
            original_dataset: Original dataset before augmentation
            augmented_dataset: Dataset after augmentation
            stage: Stage name ('reduction' or 'augmentation')

        Returns:
            Dictionary of metric_name -> value
        """
        # Extract entities from appropriate KG based on stage
        if stage == "augmentation":
            orig_kg = original_dataset.knowledge_graph_source
            aug_kg = augmented_dataset.knowledge_graph_source
        else:
            orig_kg = original_dataset.knowledge_graph_target
            aug_kg = augmented_dataset.knowledge_graph_target

        # Identify entity URIs from triples
        orig_uris = self._get_entity_uris(orig_kg)
        aug_uris = self._get_entity_uris(aug_kg)
        synthetic_uris = aug_uris - orig_uris

        if not synthetic_uris:
            return {"error": "No synthetic entities found"}

        # Extract attribute values
        orig_values = self._extract_attribute_values(orig_kg, orig_uris)
        synth_values = self._extract_attribute_values(aug_kg, synthetic_uris)

        metrics = {}

        # 1. Attribute Value Diversity
        metrics.update(self._compute_attribute_diversity(orig_values, synth_values))

        # 2. Lexical Diversity (Type-Token Ratio)
        metrics.update(self._compute_lexical_diversity(orig_values, synth_values))

        # 3. Embedding Diversity
        metrics.update(self._compute_embedding_diversity(orig_values, synth_values))

        # 4. Self-BLEU (lower = more diverse)
        metrics["self_bleu_synthetic"] = self._compute_self_bleu(synth_values)
        metrics["self_bleu_original"] = self._compute_self_bleu(orig_values)

        # 5. Structural Diversity
        metrics.update(self._compute_structural_diversity(orig_kg, aug_kg, orig_uris, synthetic_uris))

        # Summary statistics
        metrics["num_synthetic_entities"] = len(synthetic_uris)
        metrics["num_original_entities"] = len(orig_uris)
        metrics["augmentation_ratio"] = len(synthetic_uris) / len(orig_uris) if orig_uris else 0

        return metrics

    def _get_entity_uris(self, kg: KnowledgeGraph) -> Set[str]:
        """Extract all entity URIs from knowledge graph.

        Args:
            kg: Knowledge graph

        Returns:
            Set of entity URI strings
        """
        entities = set()
        for s, p, o in kg.triples((None, None, None)):
            if isinstance(s, URIRef):
                entities.add(str(s))
        return entities

    def _extract_attribute_values(
        self,
        kg: KnowledgeGraph,
        entity_uris: Set[str]
    ) -> Dict[str, List[str]]:
        """Extract all attribute values from entities.

        Args:
            kg: Knowledge graph
            entity_uris: URIs of entities to extract

        Returns:
            Dict mapping predicate -> list of values
        """
        values_by_predicate = defaultdict(list)

        for uri in entity_uris:
            # Query all triples for this entity
            for s, p, o in kg.triples((URIRef(uri), None, None)):
                # Only consider attribute triples (with Literal objects)
                if isinstance(o, Literal):
                    values_by_predicate[str(p)].append(str(o))

        return dict(values_by_predicate)

    def _compute_attribute_diversity(
        self,
        orig_values: Dict[str, List[str]],
        synth_values: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Compute diversity of attribute values.

        Measures: Unique values, overlap, new values introduced
        """
        metrics = {}

        all_predicates = set(orig_values.keys()) | set(synth_values.keys())

        unique_orig_total = 0
        unique_synth_total = 0
        overlap_total = 0
        new_values_total = 0

        for pred in all_predicates:
            orig_set = set(orig_values.get(pred, []))
            synth_set = set(synth_values.get(pred, []))

            unique_orig_total += len(orig_set)
            unique_synth_total += len(synth_set)
            overlap = len(orig_set & synth_set)
            overlap_total += overlap
            new_values = len(synth_set - orig_set)
            new_values_total += new_values

        metrics["unique_values_original"] = unique_orig_total
        metrics["unique_values_synthetic"] = unique_synth_total
        metrics["value_overlap_count"] = overlap_total
        metrics["new_values_count"] = new_values_total

        # Novelty ratio: how many synthetic values are new?
        if unique_synth_total > 0:
            metrics["novelty_ratio"] = new_values_total / unique_synth_total
        else:
            metrics["novelty_ratio"] = 0.0

        return metrics

    def _compute_lexical_diversity(
        self,
        orig_values: Dict[str, List[str]],
        synth_values: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Compute Type-Token Ratio (TTR) for text attributes.

        TTR = unique_words / total_words
        Higher TTR = more diverse vocabulary
        """
        def get_ttr(values_dict: Dict[str, List[str]]) -> float:
            all_words = []
            for vals in values_dict.values():
                for v in vals:
                    # Simple tokenization
                    words = v.lower().split()
                    all_words.extend(words)

            if not all_words:
                return 0.0

            unique_words = len(set(all_words))
            total_words = len(all_words)
            return unique_words / total_words if total_words > 0 else 0.0

        return {
            "lexical_diversity_original": get_ttr(orig_values),
            "lexical_diversity_synthetic": get_ttr(synth_values),
        }

    def _compute_embedding_diversity(
        self,
        orig_values: Dict[str, List[str]],
        synth_values: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Compute diversity in semantic embedding space.

        Measures:
            - Avg pairwise cosine distance within synthetic entities
            - Avg distance from synthetic to original
        """
        # Lazy load encoder
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.embedding_model_name)

        # Flatten all values
        orig_texts = [v for vals in orig_values.values() for v in vals]
        synth_texts = [v for vals in synth_values.values() for v in vals]

        if not synth_texts or not orig_texts:
            return {
                "embedding_diversity_synthetic": 0.0,
                "embedding_distance_orig_to_synth": 0.0,
            }

        # Sample if too many (for performance)
        max_samples = 500
        if len(orig_texts) > max_samples:
            orig_texts = np.random.choice(orig_texts, max_samples, replace=False).tolist()
        if len(synth_texts) > max_samples:
            synth_texts = np.random.choice(synth_texts, max_samples, replace=False).tolist()

        # Encode
        orig_embs = self.encoder.encode(orig_texts, convert_to_tensor=False, show_progress_bar=False)
        synth_embs = self.encoder.encode(synth_texts, convert_to_tensor=False, show_progress_bar=False)

        # Compute pairwise distances within synthetic
        synth_distances = []
        for i in range(len(synth_embs)):
            for j in range(i + 1, len(synth_embs)):
                dist = 1 - np.dot(synth_embs[i], synth_embs[j]) / (
                    np.linalg.norm(synth_embs[i]) * np.linalg.norm(synth_embs[j])
                )
                synth_distances.append(dist)

        # Compute distances from synthetic to original (closest)
        cross_distances = []
        for synth_emb in synth_embs:
            distances_to_orig = [
                1 - np.dot(synth_emb, orig_emb) / (
                    np.linalg.norm(synth_emb) * np.linalg.norm(orig_emb)
                )
                for orig_emb in orig_embs
            ]
            cross_distances.append(min(distances_to_orig))

        return {
            "embedding_diversity_synthetic": float(np.mean(synth_distances)) if synth_distances else 0.0,
            "embedding_distance_orig_to_synth": float(np.mean(cross_distances)) if cross_distances else 0.0,
        }

    def _compute_self_bleu(self, values_dict: Dict[str, List[str]]) -> float:
        """Compute Self-BLEU score.

        Self-BLEU measures n-gram overlap between generated sentences.
        Lower Self-BLEU = more diverse (less repetitive).

        Simplified version: counts 2-gram overlap.
        """
        all_texts = [v for vals in values_dict.values() for v in vals]

        if len(all_texts) < 2:
            return 0.0

        # Sample for performance
        if len(all_texts) > 200:
            all_texts = np.random.choice(all_texts, 200, replace=False).tolist()

        def get_ngrams(text: str, n: int = 2) -> Set[Tuple[str, ...]]:
            words = text.lower().split()
            if len(words) < n:
                return set()
            return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

        # Compute average n-gram overlap
        overlaps = []
        for i, text in enumerate(all_texts):
            ref_ngrams = get_ngrams(text)
            if not ref_ngrams:
                continue

            # Compare with other texts
            other_texts = all_texts[:i] + all_texts[i+1:]
            for other in other_texts:
                other_ngrams = get_ngrams(other)
                if not other_ngrams:
                    continue

                overlap = len(ref_ngrams & other_ngrams) / len(ref_ngrams)
                overlaps.append(overlap)

        return float(np.mean(overlaps)) if overlaps else 0.0

    def _compute_structural_diversity(
        self,
        orig_kg: KnowledgeGraph,
        aug_kg: KnowledgeGraph,
        orig_uris: Set[str],
        synth_uris: Set[str]
    ) -> Dict[str, float]:
        """Compute diversity in entity structure.

        Measures:
            - Variance in number of attributes per entity
            - Variance in predicate usage patterns
        """
        def get_attribute_counts(kg: KnowledgeGraph, uris: Set[str]) -> List[int]:
            counts = []
            for uri in uris:
                # Count literal triples for this entity
                attr_count = sum(
                    1 for s, p, o in kg.triples((URIRef(uri), None, None))
                    if isinstance(o, Literal)
                )
                counts.append(attr_count)
            return counts

        orig_counts = get_attribute_counts(orig_kg, orig_uris)
        synth_counts = get_attribute_counts(aug_kg, synth_uris)

        return {
            "structural_variance_original": float(np.var(orig_counts)) if orig_counts else 0.0,
            "structural_variance_synthetic": float(np.var(synth_counts)) if synth_counts else 0.0,
            "avg_attributes_original": float(np.mean(orig_counts)) if orig_counts else 0.0,
            "avg_attributes_synthetic": float(np.mean(synth_counts)) if synth_counts else 0.0,
        }


def analyze_diversity(
    original_path: Path,
    augmented_path: Path,
    output_path: Path = None,
    stage: str = "augmentation"
) -> Dict[str, float]:
    """Analyze diversity of augmented dataset.

    Args:
        original_path: Path to original dataset
        augmented_path: Path to augmented dataset
        output_path: Optional path to save metrics JSON
        stage: Stage name ('reduction' or 'augmentation')

    Returns:
        Dictionary of diversity metrics
    """
    from src.core import DatasetReaderFactory

    # Load datasets
    reader = DatasetReaderFactory.create_reader("openea")
    orig_dataset = reader.read(original_path)
    aug_dataset = reader.read(augmented_path)

    # Analyze
    analyzer = DiversityAnalyzer()
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

    parser = argparse.ArgumentParser(description="Analyze diversity of augmented entities")
    parser.add_argument("--original", type=str, required=True, help="Path to original dataset")
    parser.add_argument("--augmented", type=str, required=True, help="Path to augmented dataset")
    parser.add_argument("--output", type=str, help="Path to save metrics JSON")
    parser.add_argument("--stage", type=str, default="augmentation", choices=["reduction", "augmentation"])

    args = parser.parse_args()

    metrics = analyze_diversity(
        Path(args.original),
        Path(args.augmented),
        Path(args.output) if args.output else None,
        stage=args.stage
    )

    print("\n=== Diversity Metrics ===")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"{key:40s}: {value:.4f}")
        else:
            print(f"{key:40s}: {value}")
