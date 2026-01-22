"""Supervision experiment splitter for M_pool/M_test division.

This module handles the critical data splitting for supervision level experiments:

1. Initial Split (once per dataset):
   - M_gold (all alignments) -> M_pool (20%) + M_test (80%)
   - Graphs remain unchanged (full topology preserved)

2. Supervision Level Sampling (per r%):
   - M_train^(r) = sample r% from M_pool
   - M_test remains FIXED for all r levels (apples-to-apples comparison)

Key Principles:
- Graphs stay intact: We only reduce supervision (alignment labels)
- Fixed test set: M_test is the same across all r levels
- Reproducible: All splits use configurable random seeds
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Iterator
import json

from rdflib import URIRef

from src.core.dataset import Dataset
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SupervisionSplit:
    """Result of splitting alignments into pool and test sets.

    Attributes:
        pool: Set of alignment pairs available for training (20% by default)
        test: Set of alignment pairs held out for evaluation (80% by default)
        seed: Random seed used for the split (for reproducibility)
    """
    pool: Set[Tuple[URIRef, URIRef]]
    test: Set[Tuple[URIRef, URIRef]]
    seed: int

    def __post_init__(self):
        # Verify no overlap
        overlap = self.pool & self.test
        if overlap:
            raise ValueError(f"Pool and test sets overlap! {len(overlap)} common pairs.")

    @property
    def pool_size(self) -> int:
        return len(self.pool)

    @property
    def test_size(self) -> int:
        return len(self.test)

    def to_dict(self) -> Dict:
        """Serialize to dictionary for saving."""
        return {
            "pool": sorted([
                (str(e1), str(e2)) for e1, e2 in self.pool
            ]),
            "test": sorted([
                (str(e1), str(e2)) for e1, e2 in self.test
            ]),
            "seed": self.seed,
            "pool_size": self.pool_size,
            "test_size": self.test_size,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SupervisionSplit":
        """Deserialize from dictionary."""
        pool = {
            (URIRef(e1), URIRef(e2)) for e1, e2 in data["pool"]
        }
        test = {
            (URIRef(e1), URIRef(e2)) for e1, e2 in data["test"]
        }
        return cls(pool=pool, test=test, seed=data["seed"])

    def save(self, path: Path) -> None:
        """Save split to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved supervision split to {path}")

    @classmethod
    def load(cls, path: Path) -> "SupervisionSplit":
        """Load split from JSON file."""
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded supervision split from {path}")
        return cls.from_dict(data)


@dataclass
class SupervisionLevelData:
    """Data for a specific supervision level r%.

    Attributes:
        level: The supervision level (0.1 = 10%, 0.5 = 50%, etc.)
        train: Set of alignment pairs for training (r% of M_pool)
        test: Reference to fixed test set (same for all levels)
        seed: Random seed used for sampling
    """
    level: float
    train: Set[Tuple[URIRef, URIRef]]
    test: Set[Tuple[URIRef, URIRef]]
    seed: int

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def test_size(self) -> int:
        return len(self.test)


class SupervisionExperimentSplitter:
    """Handles data splitting for supervision level experiments.

    Usage:
        splitter = SupervisionExperimentSplitter(seed=42)

        # Initial split (once)
        split = splitter.split_pool_test(dataset, pool_ratio=0.2)

        # Get training data for different r levels
        for r in [0.1, 0.2, 0.5, 1.0]:
            level_data = splitter.sample_supervision_level(split, r)
            # Train model with level_data.train, evaluate on level_data.test
    """

    def __init__(self, seed: int = 42):
        """Initialize splitter with random seed.

        Args:
            seed: Base random seed for reproducibility
        """
        self.seed = seed
        self._split_cache: Optional[SupervisionSplit] = None

    def split_pool_test(
        self,
        dataset: Dataset,
        pool_ratio: float = 0.2,
        cache_path: Optional[Path] = None,
    ) -> SupervisionSplit:
        """Split aligned entities into pool (for training) and test (for evaluation).

        This is the INITIAL split that divides M_gold into M_pool and M_test.
        This split should be done ONCE per dataset and reused for all r levels.

        Args:
            dataset: Input dataset with aligned entities
            pool_ratio: Fraction of pairs for pool (default 0.2 = 20%)
            cache_path: Optional path to save/load the split

        Returns:
            SupervisionSplit with pool and test sets
        """
        # Try to load from cache
        if cache_path and cache_path.exists():
            logger.info(f"Loading cached supervision split from {cache_path}")
            self._split_cache = SupervisionSplit.load(cache_path)
            return self._split_cache

        # Normalize aligned entities
        aligned = self._normalize_alignments(dataset.aligned_entities)
        total = len(aligned)

        if total == 0:
            raise ValueError("Dataset has no aligned entities!")

        # Calculate pool size
        pool_size = max(1, int(total * pool_ratio))
        test_size = total - pool_size

        logger.info(
            f"Splitting {total} aligned pairs: "
            f"pool={pool_size} ({pool_ratio:.0%}), test={test_size} ({1-pool_ratio:.0%})"
        )

        # Deterministic sampling
        rng = random.Random(self.seed)
        ordered = sorted(aligned, key=lambda p: (str(p[0]), str(p[1])))
        shuffled = ordered.copy()
        rng.shuffle(shuffled)

        pool = set(shuffled[:pool_size])
        test = set(shuffled[pool_size:])

        split = SupervisionSplit(pool=pool, test=test, seed=self.seed)

        # Save to cache if path provided
        if cache_path:
            split.save(cache_path)

        self._split_cache = split
        return split

    def sample_supervision_level(
        self,
        split: SupervisionSplit,
        level: float,
        level_seed_offset: int = 0,
    ) -> SupervisionLevelData:
        """Sample r% of pool for a specific supervision level.

        This creates M_train^(r) by sampling level*100% of M_pool.

        Args:
            split: The initial pool/test split
            level: Supervision level (0.1 = 10%, 0.5 = 50%, 1.0 = 100%)
            level_seed_offset: Optional offset for seed (use different value for
                              multiple runs at same level)

        Returns:
            SupervisionLevelData with train and test sets
        """
        if level <= 0 or level > 1.0:
            raise ValueError(f"Level must be in (0, 1.0], got {level}")

        pool = split.pool
        pool_size = len(pool)

        # Number of pairs to sample from pool
        train_size = max(1, int(pool_size * level))

        if level >= 1.0:
            # 100% level: use all pool pairs
            train = pool.copy()
        else:
            # Sample from pool with level-specific seed
            level_seed = self.seed + int(level * 1000) + level_seed_offset
            rng = random.Random(level_seed)
            ordered_pool = sorted(pool, key=lambda p: (str(p[0]), str(p[1])))
            train = set(rng.sample(ordered_pool, train_size))

        logger.info(
            f"Supervision level {level:.0%}: "
            f"train={len(train)} (from pool of {pool_size}), "
            f"test={len(split.test)} (fixed)"
        )

        return SupervisionLevelData(
            level=level,
            train=train,
            test=split.test,
            seed=self.seed,
        )

    def iter_supervision_levels(
        self,
        split: SupervisionSplit,
        levels: List[float],
    ) -> Iterator[SupervisionLevelData]:
        """Iterate over multiple supervision levels.

        Args:
            split: The initial pool/test split
            levels: List of supervision levels to iterate (e.g., [0.1, 0.2, ..., 1.0])

        Yields:
            SupervisionLevelData for each level
        """
        for level in sorted(levels):
            yield self.sample_supervision_level(split, level)

    def create_level_dataset(
        self,
        original_dataset: Dataset,
        level_data: SupervisionLevelData,
    ) -> Tuple[Dataset, Set[Tuple[URIRef, URIRef]]]:
        """Create a dataset for training at a specific supervision level.

        The returned dataset has:
        - Same knowledge graphs as original (full topology)
        - aligned_entities = train set (for training/augmentation)

        The test set is returned separately for evaluation.

        Args:
            original_dataset: Original dataset with full KGs
            level_data: Supervision level data with train/test split

        Returns:
            Tuple of (train_dataset, test_pairs)
        """
        train_dataset = original_dataset.clone()
        train_dataset.aligned_entities = level_data.train

        # Verify graphs unchanged
        assert len(train_dataset.knowledge_graph_source) == len(original_dataset.knowledge_graph_source)
        assert len(train_dataset.knowledge_graph_target) == len(original_dataset.knowledge_graph_target)

        return train_dataset, level_data.test

    @staticmethod
    def _normalize_alignments(aligned_entities) -> Set[Tuple[URIRef, URIRef]]:
        """Normalize aligned entity pairs to URIRef tuples."""
        normalized = set()
        for left, right in aligned_entities:
            if isinstance(left, URIRef):
                left_uri = left
            else:
                left_uri = URIRef(str(left))

            if isinstance(right, URIRef):
                right_uri = right
            else:
                right_uri = URIRef(str(right))

            normalized.add((left_uri, right_uri))

        return normalized


__all__ = ["SupervisionExperimentSplitter", "SupervisionSplit", "SupervisionLevelData"]
