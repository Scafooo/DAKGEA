"""Alignment-only reduction strategy.

This reducer reduces ONLY the aligned entity pairs without modifying the underlying
knowledge graphs. This is useful for experiments where we want to test augmentation
at various supervision levels while keeping the graph topology intact.

Key difference from RandomEntitiesReducer:
- RandomEntitiesReducer: Removes alignment pairs AND prunes triples from graphs
- AlignmentOnlyReducer: Removes alignment pairs only, graphs stay complete

Use case: Testing data augmentation effectiveness at different supervision levels (r%)
where the model should still have access to the full graph structure but only
a subset of alignment labels for training.
"""

from __future__ import annotations

import random
from typing import Iterable, Optional, Set, Tuple

from rdflib import URIRef

from src.core.dataset import Dataset
from src.logger import get_logger
from src.reduction.registry import REDUCTION_REGISTRY

logger = get_logger(__name__)


@REDUCTION_REGISTRY.register("alignment_only")
class AlignmentOnlyReducer:
    """Sample aligned entity pairs without modifying the knowledge graphs.

    Configuration:
        reduction:
            method: alignment_only
            target_entities: 100     # Number of aligned pairs to keep
            ratio: 0.5               # Alternative: keep 50% of pairs
            random_seed: 42          # For reproducibility

    Note: Either target_entities OR ratio should be specified. If both are given,
    target_entities takes precedence.
    """

    def __init__(self, config: dict):
        self.config = config
        reduction_cfg = self.config.get("reduction", {})

        # Support both target_entities (absolute) and ratio (relative)
        self.target_entities = reduction_cfg.get("target_entities")
        self.ratio = reduction_cfg.get("ratio")

        if self.target_entities is None and self.ratio is None:
            logger.warning("Neither target_entities nor ratio specified. Using 100% of pairs.")
            self.ratio = 1.0

        self.seed = reduction_cfg.get("random_seed")
        if self.seed is None:
            experiment_cfg = self.config.get("experiment", {})
            self.seed = experiment_cfg.get("seed")

    def reduce(self, dataset: Dataset) -> Dataset:
        """Reduce alignment pairs while keeping knowledge graphs intact.

        Args:
            dataset: Input dataset to reduce

        Returns:
            Dataset with reduced aligned_entities but unchanged KGs
        """
        logger.info("[STEP] AlignmentOnly reduction started")
        aligned_set = self._normalise_alignment(dataset.aligned_entities)
        total_pairs = len(aligned_set)

        if total_pairs == 0:
            logger.warning("Dataset has no aligned entities; skipping reduction.")
            dataset.aligned_entities = aligned_set
            return dataset

        # Determine target number of pairs
        if self.target_entities is not None:
            target_pairs = min(max(1, int(self.target_entities)), total_pairs)
        else:
            target_pairs = max(1, int(total_pairs * self.ratio))

        logger.info(
            "Reducing alignment pairs to %d (from %d) using random sampling. "
            "Knowledge graphs remain unchanged.",
            target_pairs,
            total_pairs,
        )
        logger.debug("Random seed: %s", self.seed)

        # Record graph sizes (should remain unchanged)
        source_size = len(dataset.knowledge_graph_source)
        target_size = len(dataset.knowledge_graph_target)

        # Sample pairs to KEEP (not remove)
        kept_pairs = self._sample_pairs_to_keep(aligned_set, target_pairs, self.seed)

        dataset.aligned_entities = kept_pairs

        # Verify graphs unchanged
        assert len(dataset.knowledge_graph_source) == source_size, "Source KG was modified!"
        assert len(dataset.knowledge_graph_target) == target_size, "Target KG was modified!"

        logger.info(
            "Reduction complete. Source triples: %d (unchanged); target triples: %d (unchanged); "
            "aligned pairs: %d -> %d.",
            source_size,
            target_size,
            total_pairs,
            len(dataset.aligned_entities),
        )
        logger.info("[SUCCESS] AlignmentOnly reduction finished")

        return dataset

    @staticmethod
    def _normalise_alignment(
        aligned_entities: Iterable[Tuple[object, object]]
    ) -> Set[Tuple[URIRef, URIRef]]:
        """Convert aligned entity pairs to a set of URIRefs."""
        normalised: Set[Tuple[URIRef, URIRef]] = set()
        for left, right in aligned_entities:
            normalised.add(
                (
                    AlignmentOnlyReducer._ensure_uri(left),
                    AlignmentOnlyReducer._ensure_uri(right),
                )
            )
        return normalised

    @staticmethod
    def _ensure_uri(value) -> URIRef:
        if isinstance(value, URIRef):
            return value
        return URIRef(str(value))

    @staticmethod
    def _sample_pairs_to_keep(
        aligned_entities: Set[Tuple[URIRef, URIRef]],
        keep_count: int,
        seed: Optional[int] = None,
    ) -> Set[Tuple[URIRef, URIRef]]:
        """Randomly select a subset of alignment pairs to keep."""
        if keep_count >= len(aligned_entities):
            return aligned_entities

        # Sort for deterministic order before sampling
        ordered = sorted(
            aligned_entities,
            key=lambda pair: (str(pair[0]), str(pair[1])),
        )
        rng = random.Random(seed)
        return set(rng.sample(ordered, keep_count))


__all__ = ["AlignmentOnlyReducer"]
