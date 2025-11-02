"""Batch generator mirroring the original BERT-INT negative sampling logic."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np


class BatchTrainDataGenerator(Iterator[Tuple[List[int], List[int], List[int], List[int]]]):
    """Generate batches of positive/negative pairs for basic unit training."""

    def __init__(
        self,
        train_pairs: Sequence[Tuple[int, int]],
        ent_ids1: Sequence[int],
        ent_ids2: Sequence[int],
        index2entity: Dict[int, str],
        batch_size: int,
        neg_num: int,
    ) -> None:
        self.ent_ill = list(train_pairs)
        self.ent_ids1 = list(ent_ids1)
        self.ent_ids2 = list(ent_ids2)
        self.batch_size = batch_size
        self.neg_num = neg_num
        self.iter_count = 0
        self.index2entity = index2entity
        self.train_index: List[Tuple[int, int, int, int]] = []
        self.batch_num = 0

    def train_index_gene(self, candidate_dict: Dict[int, Sequence[int]]) -> None:
        """Generate negative sampling combinations following the original logic."""
        train_index: List[Tuple[int, int, int, int]] = []
        candid_min = min(len(values) for values in candidate_dict.values() if values) or 1
        for entity, values in candidate_dict.items():
            candidate_dict[entity] = np.array(values)

        for pe1, pe2 in self.ent_ill:
            for _ in range(self.neg_num):
                if np.random.rand() <= 0.5:
                    ne1 = int(candidate_dict[pe2][np.random.randint(candid_min)])
                    ne2 = pe2
                else:
                    ne1 = pe1
                    ne2 = int(candidate_dict[pe1][np.random.randint(candid_min)])
                if pe1 != ne1 or pe2 != ne2:
                    train_index.append((pe1, pe2, ne1, ne2))

        np.random.shuffle(train_index)
        self.train_index = train_index
        self.batch_num = int(np.ceil(len(train_index) * 1.0 / self.batch_size))
        self.iter_count = 0

    def __iter__(self) -> "BatchTrainDataGenerator":
        return self

    def __next__(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        if self.iter_count >= self.batch_num:
            self.iter_count = 0
            raise StopIteration()

        start = self.iter_count * self.batch_size
        end = start + self.batch_size
        batch_data = self.train_index[start:end]
        self.iter_count += 1

        pe1s = [pe1 for pe1, _, _, _ in batch_data]
        pe2s = [pe2 for _, pe2, _, _ in batch_data]
        ne1s = [ne1 for _, _, ne1, _ in batch_data]
        ne2s = [ne2 for _, _, _, ne2 in batch_data]
        return pe1s, pe2s, ne1s, ne2s
