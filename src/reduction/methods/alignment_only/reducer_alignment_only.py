"""Alignment-only reduction strategy.

This reducer reduces ONLY the aligned entity pairs without modifying the underlying
knowledge graphs. This is useful for experiments where we want to test augmentation
at various supervision levels while keeping the graph topology intact.

Key difference from RandomEntitiesReducer:
- RandomEntitiesReducer: Removes alignment pairs AND prunes triples from graphs
- AlignmentOnlyReducer: Removes alignment pairs only, graphs stay complete

Supports two modes:
1. Simple mode: Just sample `ratio` fraction of all pairs
2. Supervision mode: First split into pool/test, then sample from pool

Use case: Testing data augmentation effectiveness at different supervision levels (r%)
where the model should still have access to the full graph structure but only
a subset of alignment labels for training.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

from rdflib import URIRef

from src.core.dataset import Dataset
from src.logger import get_logger
from src.reduction.registry import REDUCTION_REGISTRY

logger = get_logger(__name__)


@REDUCTION_REGISTRY.register("alignment_only")
class AlignmentOnlyReducer:
    """Sample aligned entity pairs without modifying the knowledge graphs.

    Simple mode configuration:
        reduction:
            method: alignment_only
            ratio: 0.5               # Keep 50% of all pairs
            random_seed: 42          # For reproducibility

    Supervision mode configuration:
        reduction:
            method: alignment_only
            ratio: 0.5               # Supervision level: use 50% of POOL
            pool_ratio: 0.2          # Pool is 20% of total, test is 80%
            random_seed: 42          # For reproducibility

    In supervision mode:
    - First splits into M_pool (pool_ratio) and M_test (1-pool_ratio)
    - Then samples ratio fraction from M_pool as training
    - Saves M_test to test_pairs.json for writer to use as fixed test set
    """

    def __init__(self, config: dict):
        self.config = config
        reduction_cfg = self.config.get("reduction", {})

        # Basic parameters
        self.ratio = reduction_cfg.get("ratio", 1.0)
        self.target_entities = reduction_cfg.get("target_entities")

        # Supervision mode parameters
        self.pool_ratio = reduction_cfg.get("pool_ratio")  # If set, enables supervision mode
        self.supervision_mode = self.pool_ratio is not None

        # Seed
        self.seed = reduction_cfg.get("random_seed")
        if self.seed is None:
            experiment_cfg = self.config.get("experiment", {})
            self.seed = experiment_cfg.get("seed")

        # Stage root for saving test set
        aug_cfg = self.config.get("augmentation", {})
        self.stage_root = aug_cfg.get("stage_root")

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

        # Record graph sizes (should remain unchanged)
        source_size = len(dataset.knowledge_graph_source)
        target_size = len(dataset.knowledge_graph_target)

        if self.supervision_mode:
            # Supervision mode: pool/test split first
            kept_pairs, test_pairs = self._reduce_supervision_mode(aligned_set, total_pairs)
            # Save test set for writer
            self._save_test_set(test_pairs)
        else:
            # Simple mode: just sample from all pairs
            kept_pairs = self._reduce_simple_mode(aligned_set, total_pairs)

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

    def _reduce_simple_mode(
        self,
        aligned_set: Set[Tuple[URIRef, URIRef]],
        total_pairs: int,
    ) -> Set[Tuple[URIRef, URIRef]]:
        """Simple reduction: sample ratio fraction of all pairs."""
        if self.target_entities is not None:
            target_pairs = min(max(1, int(self.target_entities)), total_pairs)
        else:
            target_pairs = max(1, int(total_pairs * self.ratio))

        logger.info(
            "Simple mode: Reducing %d pairs to %d (ratio=%.2f). Graphs unchanged.",
            total_pairs,
            target_pairs,
            self.ratio,
        )

        return self._sample_pairs(aligned_set, target_pairs, self.seed)

    def _reduce_supervision_mode(
        self,
        aligned_set: Set[Tuple[URIRef, URIRef]],
        total_pairs: int,
    ) -> Tuple[Set[Tuple[URIRef, URIRef]], Set[Tuple[URIRef, URIRef]]]:
        """Supervision mode: pool/test split, then sample from pool.

        Returns:
            Tuple of (training_pairs, test_pairs)
        """
        # Step 1: Split into pool and test
        pool_size = max(1, int(total_pairs * self.pool_ratio))
        test_size = total_pairs - pool_size

        logger.info(
            "Supervision mode: Splitting %d pairs into pool=%d (%.0f%%) and test=%d (%.0f%%)",
            total_pairs,
            pool_size,
            self.pool_ratio * 100,
            test_size,
            (1 - self.pool_ratio) * 100,
        )

        # Deterministic shuffle for split
        rng = random.Random(self.seed)
        ordered = sorted(aligned_set, key=lambda p: (str(p[0]), str(p[1])))
        shuffled = ordered.copy()
        rng.shuffle(shuffled)

        pool = set(shuffled[:pool_size])
        test = set(shuffled[pool_size:])

        # Step 2: Sample supervision_level from pool
        if self.target_entities is not None:
            train_size = min(max(1, int(self.target_entities)), pool_size)
        else:
            train_size = max(1, int(pool_size * self.ratio))

        logger.info(
            "Sampling %d pairs from pool (supervision level=%.0f%% of pool)",
            train_size,
            self.ratio * 100,
        )

        # Use different seed for training sample (based on level)
        train_seed = self.seed + int(self.ratio * 1000) if self.seed else None
        train = self._sample_pairs(pool, train_size, train_seed)

        logger.info(
            "Result: train=%d pairs, test=%d pairs (FIXED)",
            len(train),
            len(test),
        )

        return train, test

    def _save_test_set(self, test_pairs: Set[Tuple[URIRef, URIRef]]) -> None:
        """Save test set to file for writer to use."""
        if not self.stage_root:
            # Try to get from lineage
            lineage = self.config.get("lineage", {})
            self.stage_root = lineage.get("reduction_root")

        if not self.stage_root:
            logger.warning("No stage_root configured, cannot save test set file")
            return

        stage_path = Path(self.stage_root)
        stage_path.mkdir(parents=True, exist_ok=True)

        test_file = stage_path / "fixed_test_pairs.json"
        test_data = {
            "pairs": sorted([(str(e1), str(e2)) for e1, e2 in test_pairs]),
            "count": len(test_pairs),
            "pool_ratio": self.pool_ratio,
            "supervision_level": self.ratio,
            "seed": self.seed,
        }

        with test_file.open("w") as f:
            json.dump(test_data, f, indent=2)

        logger.info("Saved fixed test set (%d pairs) to %s", len(test_pairs), test_file)

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
    def _sample_pairs(
        pairs: Set[Tuple[URIRef, URIRef]],
        count: int,
        seed: Optional[int] = None,
    ) -> Set[Tuple[URIRef, URIRef]]:
        """Randomly sample a subset of pairs."""
        if count >= len(pairs):
            return pairs.copy()

        ordered = sorted(pairs, key=lambda p: (str(p[0]), str(p[1])))
        rng = random.Random(seed)
        return set(rng.sample(ordered, count))


__all__ = ["AlignmentOnlyReducer"]
