#!/usr/bin/env python3
"""Entity Alignment-specific quality metrics for synthetic entities.

This module provides metrics tailored for evaluating synthetic entities
in the context of Entity Alignment tasks. These metrics go beyond general
diversity/realism to assess EA-specific properties.

Metrics:
    1. Alignment Preservation Score: Do synthetic pairs remain alignable?
    2. Structural Consistency Score: Are KG structural patterns preserved?
    3. Predicate Co-occurrence Preservation: Are attribute patterns maintained?
    4. Cross-KG Style Consistency: Do source/target maintain distinct styles?
    5. Nearest Neighbor Distance Ratio (NNDR): Balance diversity vs realism
    6. Alignment Model Performance Gain: Ultimate downstream task metric
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.data_structures import Dataset, Entity, KnowledgeGraph


class EntityAlignmentMetrics:
    """EA-specific quality metrics for synthetic entities."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize EA metrics analyzer.

        Args:
            embedding_model: Sentence transformer for semantic analysis
        """
        self.embedding_model_name = embedding_model
        self.encoder = None  # Lazy loading

    def analyze_all(
        self,
        original_dataset: Dataset,
        augmented_dataset: Dataset,
        stage: str = "augmentation"
    ) -> Dict[str, float]:
        """Compute all EA-specific metrics.

        Args:
            original_dataset: Original dataset before augmentation
            augmented_dataset: Dataset after augmentation
            stage: Stage name ('reduction' or 'augmentation')

        Returns:
            Dictionary of metric_name -> value
        """
        metrics = {}

        # 1. Alignment Preservation Score
        metrics.update(self.alignment_preservation_score(
            original_dataset, augmented_dataset, stage
        ))

        # 2. Structural Consistency Score
        if stage == "augmentation":
            orig_kg = original_dataset.kg1
            aug_kg = augmented_dataset.kg1
        else:
            orig_kg = original_dataset.kg2
            aug_kg = augmented_dataset.kg2

        metrics.update(self.structural_consistency_score(orig_kg, aug_kg))

        # 3. Predicate Co-occurrence Preservation
        metrics.update(self.predicate_cooccurrence_preservation(orig_kg, aug_kg))

        # 4. Cross-KG Style Consistency
        metrics.update(self.cross_kg_style_consistency(
            augmented_dataset, stage
        ))

        # 5. Nearest Neighbor Distance Ratio
        metrics.update(self.nearest_neighbor_distance_ratio(orig_kg, aug_kg))

        return metrics

    def alignment_preservation_score(
        self,
        original_dataset: Dataset,
        augmented_dataset: Dataset,
        stage: str = "augmentation"
    ) -> Dict[str, float]:
        """Measure if synthetic pairs remain alignable.

        Idea: If (A, B) are aligned entities, and we generate synthetic (A', B'),
        then A' should be more similar to B' than to any other entity in the
        opposite KG. This measures if the augmentation preserves alignability.

        Returns:
            - alignment_preservation_score: % of synthetic pairs where
              sim(A', B') > sim(A', B_random)
            - avg_alignment_similarity: Average similarity of aligned pairs
            - avg_random_similarity: Average similarity to random non-aligned entities
        """
        # Lazy load encoder
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.embedding_model_name)

        # Get KGs
        if stage == "augmentation":
            kg1 = augmented_dataset.kg1
            kg2 = augmented_dataset.kg2
            orig_kg1_uris = set(original_dataset.kg1.entities.keys())
        else:
            kg1 = augmented_dataset.kg2
            kg2 = augmented_dataset.kg1
            orig_kg1_uris = set(original_dataset.kg2.entities.keys())

        # Find synthetic aligned pairs
        synthetic_pairs = []
        for src_uri, tgt_uri in augmented_dataset.alignment_pairs:
            if src_uri not in orig_kg1_uris:  # Synthetic entity
                synthetic_pairs.append((src_uri, tgt_uri))

        if not synthetic_pairs:
            return {
                "alignment_preservation_score": 0.0,
                "avg_alignment_similarity": 0.0,
                "avg_random_similarity": 0.0,
                "num_synthetic_pairs": 0,
            }

        # Sample for performance
        if len(synthetic_pairs) > 50:
            synthetic_pairs = [synthetic_pairs[i] for i in np.random.choice(
                len(synthetic_pairs), 50, replace=False
            )]

        preserved_count = 0
        alignment_sims = []
        random_sims = []

        for src_uri, tgt_uri in synthetic_pairs:
            src_entity = kg1.entities.get(src_uri)
            tgt_entity = kg2.entities.get(tgt_uri)

            if not src_entity or not tgt_entity:
                continue

            # Get text representations
            src_text = self._entity_to_text(src_entity)
            tgt_text = self._entity_to_text(tgt_entity)

            if not src_text or not tgt_text:
                continue

            # Encode
            src_emb = self.encoder.encode([src_text], convert_to_tensor=False, show_progress_bar=False)[0]
            tgt_emb = self.encoder.encode([tgt_text], convert_to_tensor=False, show_progress_bar=False)[0]

            # Similarity with aligned pair
            aligned_sim = np.dot(src_emb, tgt_emb) / (
                np.linalg.norm(src_emb) * np.linalg.norm(tgt_emb)
            )
            alignment_sims.append(aligned_sim)

            # Similarity with random non-aligned entity
            random_tgt = self._get_random_entity(kg2, exclude=tgt_uri)
            if random_tgt:
                random_text = self._entity_to_text(random_tgt)
                if random_text:
                    random_emb = self.encoder.encode([random_text], convert_to_tensor=False, show_progress_bar=False)[0]
                    random_sim = np.dot(src_emb, random_emb) / (
                        np.linalg.norm(src_emb) * np.linalg.norm(random_emb)
                    )
                    random_sims.append(random_sim)

                    # Check if aligned is more similar than random
                    if aligned_sim > random_sim:
                        preserved_count += 1

        return {
            "alignment_preservation_score": preserved_count / len(synthetic_pairs) if synthetic_pairs else 0.0,
            "avg_alignment_similarity": float(np.mean(alignment_sims)) if alignment_sims else 0.0,
            "avg_random_similarity": float(np.mean(random_sims)) if random_sims else 0.0,
            "num_synthetic_pairs": len(synthetic_pairs),
        }

    def structural_consistency_score(
        self,
        orig_kg: KnowledgeGraph,
        aug_kg: KnowledgeGraph
    ) -> Dict[str, float]:
        """Check if structural patterns are preserved.

        Measures:
            - Predicate frequency distribution similarity
            - Average attributes per entity consistency
            - Predicate set overlap

        Synthetic entities should have similar structural properties to originals.
        """
        # Extract structural statistics
        orig_stats = self._get_structural_stats(orig_kg, set(orig_kg.entities.keys()))

        # Identify synthetic entities
        synth_uris = set(aug_kg.entities.keys()) - set(orig_kg.entities.keys())
        if not synth_uris:
            return {
                "structural_consistency_score": 0.0,
                "predicate_overlap": 0.0,
                "avg_attributes_similarity": 0.0,
            }

        synth_stats = self._get_structural_stats(aug_kg, synth_uris)

        # 1. Predicate frequency distribution (KL divergence)
        kl_div = self._compute_kl_divergence(
            orig_stats["predicate_freq"],
            synth_stats["predicate_freq"]
        )

        # 2. Predicate set overlap (Jaccard)
        orig_preds = set(orig_stats["predicate_freq"].keys())
        synth_preds = set(synth_stats["predicate_freq"].keys())
        jaccard = len(orig_preds & synth_preds) / len(orig_preds | synth_preds) if orig_preds | synth_preds else 0

        # 3. Average attributes similarity
        avg_attr_diff = abs(orig_stats["avg_attributes"] - synth_stats["avg_attributes"])
        avg_attr_sim = 1.0 / (1.0 + avg_attr_diff)  # Normalize to [0, 1]

        # Overall score (lower KL = better, higher Jaccard = better)
        overall_score = (jaccard + avg_attr_sim) / 2

        return {
            "structural_consistency_score": float(overall_score),
            "predicate_overlap_jaccard": float(jaccard),
            "avg_attributes_similarity": float(avg_attr_sim),
            "kl_divergence": float(kl_div),
            "orig_avg_attributes": orig_stats["avg_attributes"],
            "synth_avg_attributes": synth_stats["avg_attributes"],
        }

    def predicate_cooccurrence_preservation(
        self,
        orig_kg: KnowledgeGraph,
        aug_kg: KnowledgeGraph
    ) -> Dict[str, float]:
        """Measure predicate co-occurrence pattern consistency.

        If in original KG, predicates P1 and P2 co-occur in X% of entities,
        they should co-occur similarly in synthetic entities.
        """
        # Get co-occurrence patterns
        orig_uris = set(orig_kg.entities.keys())
        synth_uris = set(aug_kg.entities.keys()) - orig_uris

        if not synth_uris:
            return {
                "cooccurrence_preservation_score": 0.0,
                "cooccurrence_similarity": 0.0,
            }

        orig_cooccur = self._get_cooccurrence_matrix(orig_kg, orig_uris)
        synth_cooccur = self._get_cooccurrence_matrix(aug_kg, synth_uris)

        # Compute similarity (cosine similarity of flattened matrices)
        all_preds = set(orig_cooccur.keys()) | set(synth_cooccur.keys())

        if not all_preds:
            return {
                "cooccurrence_preservation_score": 0.0,
                "cooccurrence_similarity": 0.0,
            }

        # Build vectors
        orig_vec = []
        synth_vec = []
        for p1 in sorted(all_preds):
            for p2 in sorted(all_preds):
                if p1 >= p2:  # Avoid duplicates
                    continue
                orig_vec.append(orig_cooccur.get(p1, {}).get(p2, 0))
                synth_vec.append(synth_cooccur.get(p1, {}).get(p2, 0))

        if not orig_vec:
            return {
                "cooccurrence_preservation_score": 0.0,
                "cooccurrence_similarity": 0.0,
            }

        # Cosine similarity
        similarity = np.dot(orig_vec, synth_vec) / (
            np.linalg.norm(orig_vec) * np.linalg.norm(synth_vec) + 1e-10
        )

        return {
            "cooccurrence_preservation_score": float(similarity),
            "cooccurrence_similarity": float(similarity),
        }

    def cross_kg_style_consistency(
        self,
        dataset: Dataset,
        stage: str = "augmentation"
    ) -> Dict[str, float]:
        """Verify KG-specific styles are maintained.

        Each KG has a distinct "style" (e.g., DBpedia: "Paris, France" vs
        Wikidata: "Paris (city in France)"). Check if synthetic entities
        maintain these style differences.

        Measures:
            - Within-KG similarity (should be high)
            - Cross-KG similarity (should be lower)
            - Style separation score
        """
        # Lazy load encoder
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.embedding_model_name)

        # Get synthetic entities from both KGs
        if stage == "augmentation":
            kg1 = dataset.kg1
            kg2 = dataset.kg2
        else:
            kg1 = dataset.kg2
            kg2 = dataset.kg1

        # Sample synthetic entities
        synth_kg1_uris = list(set(kg1.entities.keys()))[:50]
        synth_kg2_uris = list(set(kg2.entities.keys()))[:50]

        # Get texts
        kg1_texts = [self._entity_to_text(kg1.entities[uri]) for uri in synth_kg1_uris
                     if uri in kg1.entities]
        kg2_texts = [self._entity_to_text(kg2.entities[uri]) for uri in synth_kg2_uris
                     if uri in kg2.entities]

        kg1_texts = [t for t in kg1_texts if t]
        kg2_texts = [t for t in kg2_texts if t]

        if not kg1_texts or not kg2_texts:
            return {
                "style_consistency_score": 0.0,
                "within_kg_similarity": 0.0,
                "cross_kg_similarity": 0.0,
            }

        # Encode
        kg1_embs = self.encoder.encode(kg1_texts, convert_to_tensor=False, show_progress_bar=False)
        kg2_embs = self.encoder.encode(kg2_texts, convert_to_tensor=False, show_progress_bar=False)

        # Within-KG similarity
        within_kg1 = self._avg_pairwise_similarity(kg1_embs)
        within_kg2 = self._avg_pairwise_similarity(kg2_embs)
        within_kg_sim = (within_kg1 + within_kg2) / 2

        # Cross-KG similarity
        cross_sims = []
        for emb1 in kg1_embs[:20]:  # Sample for efficiency
            for emb2 in kg2_embs[:20]:
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                cross_sims.append(sim)
        cross_kg_sim = np.mean(cross_sims)

        # Style separation: within-KG should be > cross-KG
        style_score = max(0, within_kg_sim - cross_kg_sim)

        return {
            "style_consistency_score": float(style_score),
            "within_kg_similarity": float(within_kg_sim),
            "cross_kg_similarity": float(cross_kg_sim),
        }

    def nearest_neighbor_distance_ratio(
        self,
        orig_kg: KnowledgeGraph,
        aug_kg: KnowledgeGraph
    ) -> Dict[str, float]:
        """NNDR: Nearest Neighbor Distance Ratio.

        Measures the ratio between:
            - Distance to closest original entity
            - Distance to closest synthetic entity

        Ideal: synthetic entities are "between" originals
        Too low ratio = too similar to originals (low diversity)
        Too high ratio = too far from originals (possible hallucination)
        """
        # Lazy load encoder
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.embedding_model_name)

        # Identify synthetic entities
        orig_uris = set(orig_kg.entities.keys())
        synth_uris = list(set(aug_kg.entities.keys()) - orig_uris)

        if not synth_uris:
            return {
                "nndr_mean": 0.0,
                "nndr_std": 0.0,
            }

        # Sample for efficiency
        if len(synth_uris) > 50:
            synth_uris = [synth_uris[i] for i in np.random.choice(len(synth_uris), 50, replace=False)]

        orig_uris_list = list(orig_uris)
        if len(orig_uris_list) > 200:
            orig_uris_list = [orig_uris_list[i] for i in np.random.choice(len(orig_uris_list), 200, replace=False)]

        # Get embeddings
        synth_texts = [self._entity_to_text(aug_kg.entities[uri]) for uri in synth_uris]
        orig_texts = [self._entity_to_text(orig_kg.entities[uri]) for uri in orig_uris_list if uri in orig_kg.entities]

        synth_texts = [t for t in synth_texts if t]
        orig_texts = [t for t in orig_texts if t]

        if not synth_texts or not orig_texts:
            return {
                "nndr_mean": 0.0,
                "nndr_std": 0.0,
            }

        synth_embs = self.encoder.encode(synth_texts, convert_to_tensor=False, show_progress_bar=False)
        orig_embs = self.encoder.encode(orig_texts, convert_to_tensor=False, show_progress_bar=False)

        # For each synthetic, compute NNDR
        nndrs = []
        for synth_emb in synth_embs:
            # Distance to closest original
            dists_to_orig = [
                1 - np.dot(synth_emb, orig_emb) / (np.linalg.norm(synth_emb) * np.linalg.norm(orig_emb))
                for orig_emb in orig_embs
            ]
            nearest_orig_dist = min(dists_to_orig)

            # Distance to closest other synthetic
            dists_to_synth = [
                1 - np.dot(synth_emb, other_emb) / (np.linalg.norm(synth_emb) * np.linalg.norm(other_emb))
                for other_emb in synth_embs
            ]
            dists_to_synth = [d for d in dists_to_synth if d > 0]  # Exclude self

            if dists_to_synth:
                nearest_synth_dist = min(dists_to_synth)

                # NNDR
                if nearest_synth_dist > 0:
                    nndr = nearest_orig_dist / nearest_synth_dist
                    nndrs.append(nndr)

        return {
            "nndr_mean": float(np.mean(nndrs)) if nndrs else 0.0,
            "nndr_std": float(np.std(nndrs)) if nndrs else 0.0,
            "nndr_samples": len(nndrs),
        }

    # Helper methods

    def _entity_to_text(self, entity: Entity) -> str:
        """Convert entity to text representation."""
        if not entity:
            return ""
        texts = [str(attr.value) for attr in entity.attributes if attr.value]
        return " ".join(texts)

    def _get_random_entity(self, kg: KnowledgeGraph, exclude: str = None) -> Entity:
        """Get random entity from KG."""
        uris = [uri for uri in kg.entities.keys() if uri != exclude]
        if not uris:
            return None
        random_uri = np.random.choice(uris)
        return kg.entities.get(random_uri)

    def _get_structural_stats(self, kg: KnowledgeGraph, entity_uris: Set[str]) -> Dict:
        """Get structural statistics for entities."""
        predicate_counts = defaultdict(int)
        total_attributes = 0
        num_entities = 0

        for uri in entity_uris:
            entity = kg.entities.get(uri)
            if not entity:
                continue

            num_entities += 1
            total_attributes += len(entity.attributes)

            for attr in entity.attributes:
                predicate_counts[str(attr.predicate)] += 1

        # Normalize to frequencies
        total = sum(predicate_counts.values())
        predicate_freq = {
            pred: count / total
            for pred, count in predicate_counts.items()
        } if total > 0 else {}

        return {
            "predicate_freq": predicate_freq,
            "avg_attributes": total_attributes / num_entities if num_entities > 0 else 0,
        }

    def _compute_kl_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """Compute KL divergence between two distributions."""
        all_keys = set(p.keys()) | set(q.keys())
        if not all_keys:
            return 0.0

        kl = 0.0
        for key in all_keys:
            p_val = p.get(key, 1e-10)
            q_val = q.get(key, 1e-10)
            kl += p_val * np.log(p_val / q_val)

        return max(0, kl)  # Ensure non-negative

    def _get_cooccurrence_matrix(
        self,
        kg: KnowledgeGraph,
        entity_uris: Set[str]
    ) -> Dict[str, Dict[str, float]]:
        """Get predicate co-occurrence matrix."""
        cooccur = defaultdict(lambda: defaultdict(int))

        for uri in entity_uris:
            entity = kg.entities.get(uri)
            if not entity:
                continue

            predicates = [str(attr.predicate) for attr in entity.attributes]

            # Count co-occurrences
            for i, p1 in enumerate(predicates):
                for p2 in predicates[i+1:]:
                    if p1 != p2:
                        cooccur[p1][p2] += 1
                        cooccur[p2][p1] += 1

        # Normalize
        total_entities = len(entity_uris)
        normalized = {}
        for p1, p2_dict in cooccur.items():
            normalized[p1] = {
                p2: count / total_entities
                for p2, count in p2_dict.items()
            }

        return normalized

    def _avg_pairwise_similarity(self, embeddings: np.ndarray) -> float:
        """Compute average pairwise cosine similarity."""
        sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, min(i + 20, len(embeddings))):  # Sample for efficiency
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                sims.append(sim)
        return float(np.mean(sims)) if sims else 0.0


def analyze_ea_metrics(
    original_path: Path,
    augmented_path: Path,
    output_path: Path = None,
    stage: str = "augmentation"
) -> Dict[str, float]:
    """Analyze EA-specific metrics of augmented dataset.

    Args:
        original_path: Path to original dataset
        augmented_path: Path to augmented dataset
        output_path: Optional path to save metrics JSON
        stage: Stage name ('reduction' or 'augmentation')

    Returns:
        Dictionary of EA-specific metrics
    """
    from src.core.data_io import load_dataset

    # Load datasets
    orig_dataset = load_dataset(original_path)
    aug_dataset = load_dataset(augmented_path)

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

    parser = argparse.ArgumentParser(description="Analyze EA-specific metrics")
    parser.add_argument("--original", type=str, required=True, help="Path to original dataset")
    parser.add_argument("--augmented", type=str, required=True, help="Path to augmented dataset")
    parser.add_argument("--output", type=str, help="Path to save metrics JSON")
    parser.add_argument("--stage", type=str, default="augmentation", choices=["reduction", "augmentation"])

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
            print(f"{key:50s}: {value:.4f}")
        else:
            print(f"{key:50s}: {value}")
