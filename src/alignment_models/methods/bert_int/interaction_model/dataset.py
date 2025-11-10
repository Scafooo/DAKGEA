"""Dataset utilities for BERT-INT interaction model."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np
import torch

from src.alignment_models.methods.bert_int.basic_unit.metrics import (
    batch_cosine_similarity,
    batch_topk,
)
from src.logger import get_logger

logger = get_logger(__name__)


class AttributeValueCleaner:
    """Clean attribute triples by removing noisy one-to-many mappings.

    Filters out attribute triples where an (entity, attribute) pair has more
    than a threshold number of values, as these are typically noisy.
    """

    def __init__(self, threshold: int = 3):
        """Initialize cleaner.

        Args:
            threshold: Maximum number of values allowed for an (entity, attribute) pair
        """
        self.threshold = threshold

    def clean(
        self, attribute_triples: List[Tuple[int, int, str]]
    ) -> List[Tuple[int, int, str]]:
        """Clean attribute triples by removing noisy mappings.

        Args:
            attribute_triples: List of (entity, attribute, value) triples

        Returns:
            Cleaned list of attribute triples
        """
        # Count values for each (entity, attribute) pair
        entity_attr_values: Dict[Tuple[int, int], Set[str]] = defaultdict(set)

        for entity, attribute, value in attribute_triples:
            entity_attr_values[(entity, attribute)].add(value)

        # Filter out triples with too many values
        cleaned_triples = []
        for entity, attribute, value in attribute_triples:
            if len(entity_attr_values[(entity, attribute)]) <= self.threshold:
                cleaned_triples.append((entity, attribute, value))

        removed_count = len(attribute_triples) - len(cleaned_triples)
        logger.info(
            f"Cleaned attribute triples: {len(attribute_triples)} → {len(cleaned_triples)} "
            f"(removed {removed_count} noisy triples with >{self.threshold} values per (entity, attr) pair)"
        )

        return cleaned_triples


class CandidateGenerator:
    """Generate candidate entity pairs using cosine similarity."""

    def __init__(
        self,
        topk: int = 50,
        batch_size: int = 2048,
        device: int | str | torch.device | None = None,
    ):
        """Initialize candidate generator.

        Args:
            topk: Number of top candidates to generate per entity
            batch_size: Batch size for similarity computation
            device: Device to use for computation
        """
        self.topk = topk
        self.batch_size = batch_size
        self.device = device

    def generate(
        self,
        entities_source: List[int],
        entities_target: List[int],
        entity_embeddings: np.ndarray,
    ) -> Dict[int, List[int]]:
        """Generate top-k candidate mappings from source to target entities.

        Args:
            entities_source: List of source entity IDs
            entities_target: List of target entity IDs
            entity_embeddings: Entity embeddings array of shape [num_entities, embed_dim]

        Returns:
            Dictionary mapping source entity ID to list of top-k target entity IDs
        """
        logger.info(
            f"Generating top-{self.topk} candidates for {len(entities_source)} entities"
        )

        # Get embeddings for source and target entities
        emb_source = [entity_embeddings[e] for e in entities_source]
        emb_target = [entity_embeddings[e] for e in entities_target]

        # Compute cosine similarity matrix
        sim_matrix = batch_cosine_similarity(
            emb_source, emb_target, self.batch_size, self.device
        )

        # Get top-k indices
        scores, indices = batch_topk(sim_matrix, self.batch_size, self.topk, self.device)

        # Build candidate dictionary
        candidates = {}
        for i, source_id in enumerate(entities_source):
            # indices[i] contains indices into entities_target
            target_candidates = [entities_target[idx] for idx in indices[i].tolist()]
            candidates[source_id] = target_candidates

        logger.info(f"Generated {len(candidates)} candidate mappings")
        return candidates


class InteractionDataset:
    """Dataset for interaction model training and evaluation.

    Manages entity pairs, their features, and training data generation.
    """

    def __init__(
        self,
        entity_pairs: List[Tuple[int, int]],
        features: np.ndarray,
        train_ill: List[Tuple[int, int]],
        test_ill: List[Tuple[int, int]],
        train_candidates: Dict[int, List[int]],
        test_candidates: Dict[int, List[int]],
    ):
        """Initialize interaction dataset.

        Args:
            entity_pairs: List of all (entity1, entity2) candidate pairs
            features: Feature array of shape [num_pairs, feature_dim]
            train_ill: Training entity alignment pairs
            test_ill: Test entity alignment pairs
            train_candidates: Candidate mappings for training entities
            test_candidates: Candidate mappings for test entities
        """
        self.entity_pairs = entity_pairs
        self.features = torch.FloatTensor(features)
        self.train_ill = train_ill
        self.test_ill = test_ill
        self.train_candidates = train_candidates
        self.test_candidates = test_candidates

        # Build index from entity pair to feature index
        self.pair_to_feature_idx = {
            pair: idx for idx, pair in enumerate(entity_pairs)
        }

        logger.info(f"Interaction dataset initialized:")
        logger.info(f"  Entity pairs: {len(entity_pairs)}")
        logger.info(f"  Feature dimension: {features.shape[1]}")
        logger.info(f"  Train alignments: {len(train_ill)}")
        logger.info(f"  Test alignments: {len(test_ill)}")

    def get_feature_by_pair(self, entity_pair: Tuple[int, int]) -> torch.Tensor:
        """Get feature vector for an entity pair.

        Args:
            entity_pair: Tuple of (entity1, entity2)

        Returns:
            Feature vector
        """
        idx = self.pair_to_feature_idx[entity_pair]
        return self.features[idx]

    def get_features_by_pairs(
        self, entity_pairs: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Get feature vectors for multiple entity pairs.

        Args:
            entity_pairs: List of (entity1, entity2) tuples

        Returns:
            Feature tensor of shape [len(entity_pairs), feature_dim]
        """
        indices = [self.pair_to_feature_idx[pair] for pair in entity_pairs]
        return self.features[indices]

    def get_train_pairs(self) -> List[Tuple[int, int]]:
        """Get all training entity alignment pairs."""
        return self.train_ill

    def get_test_pairs(self) -> List[Tuple[int, int]]:
        """Get all test entity alignment pairs."""
        return self.test_ill

    def get_train_candidates_for(self, entity: int) -> List[int]:
        """Get candidate target entities for a source entity in training.

        Args:
            entity: Source entity ID

        Returns:
            List of candidate target entity IDs
        """
        return self.train_candidates.get(entity, [])

    def get_test_candidates_for(self, entity: int) -> List[int]:
        """Get candidate target entities for a source entity in testing.

        Args:
            entity: Source entity ID

        Returns:
            List of candidate target entity IDs
        """
        return self.test_candidates.get(entity, [])

    @staticmethod
    def generate_all_entity_pairs(
        candidate_dicts: List[Dict[int, List[int]]],
        ill_pairs: List[List[Tuple[int, int]]],
    ) -> List[Tuple[int, int]]:
        """Generate comprehensive list of entity pairs from candidates and ILL.

        Args:
            candidate_dicts: List of candidate dictionaries
            ill_pairs: List of entity alignment pair lists

        Returns:
            List of unique entity pairs
        """
        entity_pairs = set()

        # Add all candidate pairs
        for candidate_dict in candidate_dicts:
            for e1, candidates in candidate_dict.items():
                for e2 in candidates:
                    entity_pairs.add((e1, e2))

        # Add all ILL pairs
        for ill_list in ill_pairs:
            for e1, e2 in ill_list:
                entity_pairs.add((e1, e2))

        result = sorted(list(entity_pairs))
        logger.info(f"Generated {len(result)} unique entity pairs")
        return result
