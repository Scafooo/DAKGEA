"""Mini-batch generator for the basic BERT unit training phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

Example = Tuple[int, int, int, int]
Pair = Tuple[int, int]


@dataclass
class TrainingPairGenerator(Iterator[Tuple[List[int], List[int], List[int], List[int]]]):
    """Yield positive and negative training pairs for margin ranking loss."""

    train_ill: Sequence[Pair]
    ent_ids_left: Sequence[int]
    ent_ids_right: Sequence[int]
    batch_size: int
    negatives_per_positive: int
    seed: int | None = None

    def __post_init__(self) -> None:
        self._iter_index = 0
        self._batches: List[Example] | None = None
        self._batch_count = 0
        # Create a dedicated RNG for reproducible negative sampling
        self._rng = np.random.RandomState(self.seed) if self.seed is not None else np.random

    def build_indices(self, candidate_dict: Dict[int, Sequence[int]]) -> None:
        """Prepare shuffled training tuples based on the provided candidates."""
        candidates = {ent: np.array(vals) for ent, vals in candidate_dict.items()}
        candidate_size = min(len(vals) for vals in candidates.values() if len(vals) > 0)

        examples: List[Example] = []
        for pos_left, pos_right in self.train_ill:
            for _ in range(self.negatives_per_positive):
                if self._rng.rand() <= 0.5:
                    neg_left = int(self._rng.choice(candidates[pos_right][:candidate_size]))
                    neg_right = pos_right
                else:
                    neg_left = pos_left
                    neg_right = int(self._rng.choice(candidates[pos_left][:candidate_size]))

                if pos_left != neg_left or pos_right != neg_right:
                    examples.append((pos_left, pos_right, neg_left, neg_right))

        self._rng.shuffle(examples)
        self._batches = examples
        self._batch_count = int(np.ceil(len(examples) / self.batch_size))
        self._iter_index = 0

    def __iter__(self) -> "TrainingPairGenerator":
        return self

    def __next__(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        if self._batches is None or self._iter_index >= self._batch_count:
            self._iter_index = 0
            self._batches = None
            raise StopIteration

        start = self._iter_index * self.batch_size
        end = start + self.batch_size
        self._iter_index += 1

        batch = self._batches[start:end]
        pos_left = [item[0] for item in batch]
        pos_right = [item[1] for item in batch]
        neg_left = [item[2] for item in batch]
        neg_right = [item[3] for item in batch]
        return pos_left, pos_right, neg_left, neg_right
