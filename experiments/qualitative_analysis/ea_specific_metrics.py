#!/usr/bin/env python3
"""Entity Alignment-specific quality metrics for synthetic entities.

This module provides metrics to evaluate synthetic entities specifically
in the context of Entity Alignment tasks. These metrics go beyond general
diversity/realism and assess properties critical for EA.

The 6 EA-Specific Metrics:
    1. Alignment Preservation Score: Do synthetic pairs remain alignable?
    2. Structural Consistency Score: Is KG structure preserved?
    3. Predicate Co-occurrence Preservation: Are attribute patterns maintained?
    4. Cross-KG Style Consistency: Do KGs maintain distinct styles?
    5. Nearest Neighbor Distance Ratio (NNDR): Balance diversity vs realism
    6. Alignment Model Performance Gain: Does augmentation improve EA models?
       (This requires training an EA model, so it's a placeholder here)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from rdflib import Literal, URIRef
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer

from src.core import Dataset, KnowledgeGraph


class EntityAlignmentMetrics:
    """Compute EA-specific quality metrics for synthetic entities."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize EA metrics analyzer.

        Args:
            embedding_model: Sentence transformer for semantic analysis
        """
        self.embedding_model_name = embedding_model
        self.encoder = None

    def analyze_all(
        self,
        original_dataset: Dataset,
        augmented_dataset: Dataset,
        stage: str = "augmentation"
    ) -> Dict[str, float]:
        """Compute all 6 EA-specific metrics.

        Args:
            original_dataset: Original dataset
            augmented_dataset: Augmented dataset
            stage: Stage name

        Returns:
            Dictionary of EA metrics
        """
        metrics = {}

        # Get appropriate KGs
        if stage == "augmentation":
            orig_src_kg = original_dataset.knowledge_graph_source
            orig_tgt_kg = original_dataset.knowledge_graph_target
            aug_src_kg = augmented_dataset.knowledge_graph_source
            aug_tgt_kg = augmented_dataset.knowledge_graph_target
        else:
            orig_src_kg = original_dataset.knowledge_graph_target
            orig_tgt_kg = original_dataset.knowledge_graph_source
            aug_src_kg = augmented_dataset.knowledge_graph_target
            aug_tgt_kg = augmented_dataset.knowledge_graph_source

        # 1. Alignment Preservation Score
        metrics.update(self.alignment_preservation_score(
            augmented_dataset, stage
        ))

        # 2. Structural Consistency Score
        metrics.update(self.structural_consistency_score(
            orig_src_kg, aug_src_kg
        ))

        # 3. Predicate Co-occurrence Preservation
        metrics.update(self.predicate_cooccurrence_preservation(
            orig_src_kg, aug_src_kg
        ))

        # 4. Cross-KG Style Consistency
        metrics.update(self.cross_kg_style_consistency(
            augmented_dataset, stage
        ))

        # 5. Nearest Neighbor Distance Ratio
        metrics.update(self.nearest_neighbor_distance_ratio(
            orig_src_kg, aug_src_kg
        ))

        # 6. Alignment Model Performance Gain
        # Note: This requires training EA models, which is expensive
        # We provide a placeholder that can be filled externally
        metrics["alignment_model_performance_gain"] = None

        return metrics

    def alignment_preservation_score(
        self,
        dataset: Dataset,
        stage: str
    ) -> Dict[str, float]:
        """Metric #1: Alignment Preservation Score.

        Measures if synthetic entity pairs remain alignable.
        If (A, B) are aligned and we generate (A', B'), then A' should be
        more similar to B' than to random entities.
        """
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.embedding_model_name)

        src_kg = dataset.knowledge_graph_source
        tgt_kg = dataset.knowledge_graph_target

        preserved_count = 0
        total_pairs = 0

        # Sample aligned pairs
        aligned_pairs = list(dataset.aligned_entities)
        sampled_pairs = aligned_pairs
        if len(aligned_pairs) > 100:
            indices = np.random.choice(len(aligned_pairs), 100, replace=False)
            sampled_pairs = [aligned_pairs[i] for i in indices]

        # Get all target entities for random sampling
        tgt_uris = self._get_entity_uris(tgt_kg)

        for src_uri, tgt_uri in sampled_pairs:
            # Get text representations
            src_text = self._entity_to_text(src_kg, str(src_uri))
            tgt_text = self._entity_to_text(tgt_kg, str(tgt_uri))

            if not src_text or not tgt_text:
                continue

            # Encode
            embeddings = self.encoder.encode([src_text, tgt_text], convert_to_tensor=False, show_progress_bar=False)
            src_emb, tgt_emb = embeddings[0], embeddings[1]

            # Similarity to aligned pair
            aligned_sim = 1 - cosine(src_emb, tgt_emb)

            # Similarity to random target entity
            random_tgt_uri = np.random.choice(list(tgt_uris - {str(tgt_uri)}))
            random_text = self._entity_to_text(tgt_kg, random_tgt_uri)

            if random_text:
                random_emb = self.encoder.encode([random_text], convert_to_tensor=False, show_progress_bar=False)[0]
                random_sim = 1 - cosine(src_emb, random_emb)

                # Check if aligned pair is more similar than random
                if aligned_sim > random_sim:
                    preserved_count += 1

                total_pairs += 1

        preservation_score = preserved_count / total_pairs if total_pairs > 0 else 0.0

        return {
            "alignment_preservation_score": preservation_score,
            "preserved_pairs": preserved_count,
            "total_evaluated_pairs": total_pairs,
        }

    def structural_consistency_score(
        self,
        orig_kg: KnowledgeGraph,
        aug_kg: KnowledgeGraph
    ) -> Dict[str, float]:
        """Metric #2: Structural Consistency Score.

        Measures if KG structural patterns (predicates, frequencies) are preserved.
        """
        # Get entity URIs
        orig_uris = self._get_entity_uris(orig_kg)
        aug_uris = self._get_entity_uris(aug_kg)
        synth_uris = aug_uris - orig_uris

        if not synth_uris:
            return {"structural_consistency_score": 0.0}

        # Get predicate statistics
        orig_stats = self._get_structural_stats(orig_kg, orig_uris)
        synth_stats = self._get_structural_stats(aug_kg, synth_uris)

        # 1. Jaccard similarity of predicate sets
        orig_preds = set(orig_stats["predicate_freq"].keys())
        synth_preds = set(synth_stats["predicate_freq"].keys())
        jaccard = len(orig_preds & synth_preds) / len(orig_preds | synth_preds) if (orig_preds | synth_preds) else 0.0

        # 2. KL divergence of predicate frequencies
        all_preds = orig_preds | synth_preds
        orig_freq = np.array([orig_stats["predicate_freq"].get(p, 0) for p in all_preds])
        synth_freq = np.array([synth_stats["predicate_freq"].get(p, 0) for p in all_preds])

        # Add smoothing
        orig_freq = orig_freq + 1e-10
        synth_freq = synth_freq + 1e-10

        # Normalize
        orig_freq = orig_freq / orig_freq.sum()
        synth_freq = synth_freq / synth_freq.sum()

        kl_div = entropy(orig_freq, synth_freq)

        # 3. Average attributes per entity
        attr_diff = abs(orig_stats["avg_attributes"] - synth_stats["avg_attributes"])

        # Combine into score (higher = better)
        structural_score = (jaccard + (1 / (1 + kl_div)) + (1 / (1 + attr_diff))) / 3

        return {
            "structural_consistency_score": structural_score,
            "predicate_overlap_jaccard": jaccard,
            "kl_divergence": kl_div,
            "avg_attributes_diff": attr_diff,
        }

    def predicate_cooccurrence_preservation(
        self,
        orig_kg: KnowledgeGraph,
        aug_kg: KnowledgeGraph
    ) -> Dict[str, float]:
        """Metric #3: Predicate Co-occurrence Preservation.

        Measures if patterns of co-occurring predicates are maintained.
        """
        orig_uris = self._get_entity_uris(orig_kg)
        aug_uris = self._get_entity_uris(aug_kg)
        synth_uris = aug_uris - orig_uris

        if not synth_uris:
            return {"cooccurrence_preservation_score": 0.0}

        # Build co-occurrence matrices
        orig_cooccur = self._build_cooccurrence_matrix(orig_kg, orig_uris)
        synth_cooccur = self._build_cooccurrence_matrix(aug_kg, synth_uris)

        # Get all predicates
        all_preds = set(orig_cooccur.keys()) | set(synth_cooccur.keys())

        if not all_preds:
            return {"cooccurrence_preservation_score": 0.0}

        # Create vectors
        orig_vec = []
        synth_vec = []

        for p1 in all_preds:
            for p2 in all_preds:
                if p1 < p2:  # Avoid duplicates
                    orig_vec.append(orig_cooccur.get((p1, p2), 0))
                    synth_vec.append(synth_cooccur.get((p1, p2), 0))

        # Compute cosine similarity
        if len(orig_vec) == 0 or sum(orig_vec) == 0 or sum(synth_vec) == 0:
            similarity = 0.0
        else:
            similarity = 1 - cosine(np.array(orig_vec), np.array(synth_vec))

        return {
            "cooccurrence_preservation_score": similarity,
        }

    def cross_kg_style_consistency(
        self,
        dataset: Dataset,
        stage: str
    ) -> Dict[str, float]:
        """Metric #4: Cross-KG Style Consistency.

        Measures if each KG maintains its distinct style.
        within-KG similarity should be higher than cross-KG similarity.
        """
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.embedding_model_name)

        src_kg = dataset.knowledge_graph_source
        tgt_kg = dataset.knowledge_graph_target

        src_uris = list(self._get_entity_uris(src_kg))
        tgt_uris = list(self._get_entity_uris(tgt_kg))

        # Sample for performance
        src_sample = src_uris[:50] if len(src_uris) <= 50 else list(np.random.choice(src_uris, 50, replace=False))
        tgt_sample = tgt_uris[:50] if len(tgt_uris) <= 50 else list(np.random.choice(tgt_uris, 50, replace=False))

        # Get embeddings
        src_texts = [self._entity_to_text(src_kg, uri) for uri in src_sample]
        tgt_texts = [self._entity_to_text(tgt_kg, uri) for uri in tgt_sample]

        src_texts = [t for t in src_texts if t]
        tgt_texts = [t for t in tgt_texts if t]

        if not src_texts or not tgt_texts:
            return {"style_consistency_score": 0.0}

        src_embs = self.encoder.encode(src_texts, convert_to_tensor=False, show_progress_bar=False)
        tgt_embs = self.encoder.encode(tgt_texts, convert_to_tensor=False, show_progress_bar=False)

        # Within-KG similarities
        within_src_sims = []
        for i in range(len(src_embs)):
            for j in range(i + 1, min(i + 10, len(src_embs))):  # Limit for performance
                sim = 1 - cosine(src_embs[i], src_embs[j])
                within_src_sims.append(sim)

        within_tgt_sims = []
        for i in range(len(tgt_embs)):
            for j in range(i + 1, min(i + 10, len(tgt_embs))):
                sim = 1 - cosine(tgt_embs[i], tgt_embs[j])
                within_tgt_sims.append(sim)

        # Cross-KG similarities
        cross_sims = []
        for i in range(min(len(src_embs), 20)):
            for j in range(min(len(tgt_embs), 20)):
                sim = 1 - cosine(src_embs[i], tgt_embs[j])
                cross_sims.append(sim)

        within_sim = np.mean(within_src_sims + within_tgt_sims)
        cross_sim = np.mean(cross_sims)

        # Style score: within should be higher than cross
        style_score = within_sim - cross_sim

        return {
            "style_consistency_score": style_score,
            "within_kg_similarity": within_sim,
            "cross_kg_similarity": cross_sim,
        }

    def nearest_neighbor_distance_ratio(
        self,
        orig_kg: KnowledgeGraph,
        aug_kg: KnowledgeGraph
    ) -> Dict[str, float]:
        """Metric #5: Nearest Neighbor Distance Ratio (NNDR).

        Measures balance between diversity and realism.
        Ratio of distance to nearest original vs nearest synthetic.
        Target: 0.8-1.2 (synthetic entities are "between" originals).
        """
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.embedding_model_name)

        orig_uris = list(self._get_entity_uris(orig_kg))
        aug_uris = list(self._get_entity_uris(aug_kg))
        synth_uris = list(set(aug_uris) - set(orig_uris))

        if not synth_uris or not orig_uris:
            return {"nndr_mean": 0.0}

        # Sample for performance
        synth_sample = synth_uris[:50] if len(synth_uris) <= 50 else list(np.random.choice(synth_uris, 50, replace=False))
        orig_sample = orig_uris[:200] if len(orig_uris) <= 200 else list(np.random.choice(orig_uris, 200, replace=False))

        # Get embeddings
        synth_texts = [self._entity_to_text(aug_kg, uri) for uri in synth_sample]
        synth_texts = [t for t in synth_texts if t]

        orig_texts = [self._entity_to_text(orig_kg, uri) for uri in orig_sample]
        orig_texts = [t for t in orig_texts if t]

        if not synth_texts or not orig_texts:
            return {"nndr_mean": 0.0}

        synth_embs = self.encoder.encode(synth_texts, convert_to_tensor=False, show_progress_bar=False)
        orig_embs = self.encoder.encode(orig_texts, convert_to_tensor=False, show_progress_bar=False)

        # Compute NNDR for each synthetic entity
        nndr_values = []
        for synth_emb in synth_embs:
            # Distance to nearest original
            orig_dists = [cosine(synth_emb, orig_emb) for orig_emb in orig_embs]
            nearest_orig_dist = min(orig_dists)

            # Distance to nearest other synthetic
            synth_dists = [cosine(synth_emb, other_emb) for other_emb in synth_embs]
            synth_dists = [d for d in synth_dists if d > 0]  # Exclude self
            if synth_dists:
                nearest_synth_dist = min(synth_dists)

                # NNDR = dist_to_orig / dist_to_synth
                if nearest_synth_dist > 0:
                    nndr = nearest_orig_dist / nearest_synth_dist
                    nndr_values.append(nndr)

        return {
            "nndr_mean": float(np.mean(nndr_values)) if nndr_values else 0.0,
            "nndr_std": float(np.std(nndr_values)) if nndr_values else 0.0,
        }

    # Helper methods

    def _get_entity_uris(self, kg: KnowledgeGraph) -> Set[str]:
        """Extract all entity URIs from knowledge graph."""
        entities = set()
        for s, p, o in kg.triples((None, None, None)):
            if isinstance(s, URIRef):
                entities.add(str(s))
        return entities

    def _entity_to_text(self, kg: KnowledgeGraph, uri: str) -> str:
        """Convert entity to text representation."""
        texts = []
        for s, p, o in kg.triples((URIRef(uri), None, None)):
            if isinstance(o, Literal):
                texts.append(str(o))

        return " ".join(texts[:10]) if texts else ""

    def _get_structural_stats(self, kg: KnowledgeGraph, entity_uris: Set[str]) -> Dict:
        """Get structural statistics for entities."""
        predicate_counts = defaultdict(int)
        total_attributes = 0
        num_entities = 0

        for uri in entity_uris:
            entity_attrs = 0
            for s, p, o in kg.triples((URIRef(uri), None, None)):
                if isinstance(o, Literal):
                    predicate_counts[str(p)] += 1
                    entity_attrs += 1

            if entity_attrs > 0:
                num_entities += 1
                total_attributes += entity_attrs

        # Normalize to frequencies
        total = sum(predicate_counts.values())
        predicate_freq = {
            pred: count / total for pred, count in predicate_counts.items()
        } if total > 0 else {}

        return {
            "predicate_freq": predicate_freq,
            "avg_attributes": total_attributes / num_entities if num_entities > 0 else 0,
        }

    def _build_cooccurrence_matrix(self, kg: KnowledgeGraph, entity_uris: Set[str]) -> Dict[Tuple[str, str], int]:
        """Build predicate co-occurrence matrix."""
        cooccur = defaultdict(int)

        for uri in entity_uris:
            # Get all predicates for this entity
            preds = set()
            for s, p, o in kg.triples((URIRef(uri), None, None)):
                if isinstance(o, Literal):
                    preds.add(str(p))

            # Count co-occurrences
            pred_list = sorted(preds)
            for i, p1 in enumerate(pred_list):
                for p2 in pred_list[i + 1:]:
                    cooccur[(p1, p2)] += 1

        return dict(cooccur)


def analyze_ea_metrics(
    original_path: Path,
    augmented_path: Path,
    output_path: Path = None,
    stage: str = "augmentation"
) -> Dict[str, float]:
    """Analyze EA-specific metrics for augmented dataset.

    Args:
        original_path: Path to original dataset
        augmented_path: Path to augmented dataset
        output_path: Optional path to save metrics JSON
        stage: Stage name

    Returns:
        Dictionary of EA metrics
    """
    from src.core import DatasetReaderFactory

    # Load datasets
    reader = DatasetReaderFactory.create_reader("openea")
    orig_dataset = reader.read(original_path)
    aug_dataset = reader.read(augmented_path)

    # Analyze
    analyzer = EntityAlignmentMetrics()
    metrics = analyzer.analyze_all(orig_dataset, aug_dataset, stage=stage)

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, required=True)
    parser.add_argument("--augmented", type=str, required=True)
    parser.add_argument("--output", type=str)
    parser.add_argument("--stage", type=str, default="augmentation")

    args = parser.parse_args()

    metrics = analyze_ea_metrics(
        Path(args.original),
        Path(args.augmented),
        Path(args.output) if args.output else None,
        stage=args.stage
    )

    print("\n=== Entity Alignment-Specific Metrics ===")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"{key:40s}: {value:.4f}")
        elif value is None:
            print(f"{key:40s}: N/A")
        else:
            print(f"{key:40s}: {value}")
