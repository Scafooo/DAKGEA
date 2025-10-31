"""Batch data generator mirroring the original BERT-INT implementation."""

from __future__ import annotations

import numpy as np

from src.logger import get_structured_logger

logger = get_structured_logger(__name__)

class BatchTrainDataGenerator:
    """Generate positive/negative training tuples for margin ranking."""

    def __init__(
        self,
        train_ill,
        ent_ids1,
        ent_ids2,
        batch_size: int,
        neg_num: int,
    ) -> None:
        self.ent_ill = list(train_ill)
        self.ent_ids1 = list(ent_ids1)
        self.ent_ids2 = list(ent_ids2)
        self.batch_size = batch_size
        self.neg_num = neg_num
        self.iter_count = 0
        self.batch_num = 0
        self.train_index = []

    def build_candidate_dict(self, candidate_dict):
        lengths = [len(values) for values in candidate_dict.values() if len(values) > 0]
        logger.info(f"[BATCH_GEN] Candidate dict: {len(candidate_dict)} entities, {len(lengths)} with candidates")
        if not lengths:
            logger.info("[BATCH_GEN] No candidates found!")
            self.train_index = []
            self.batch_num = 0
            return
        minim = min(lengths)
        logger.info(f"[BATCH_GEN] Min candidates per entity: {minim}")
        for ent in candidate_dict:
            candidate_dict[ent] = np.array(candidate_dict[ent][:minim])
        self._generate_training_indices(candidate_dict)

    def _generate_training_indices(self, candidate_dict):
        train_index = []
        skipped = 0
        for pe1, pe2 in self.ent_ill:
            # Skip if either entity doesn't have candidates
            if pe1 not in candidate_dict or pe2 not in candidate_dict:
                skipped += 1
                continue
            for _ in range(self.neg_num):
                if np.random.rand() <= 0.5:
                    ne1 = np.random.choice(candidate_dict[pe2])
                    ne2 = pe2
                else:
                    ne1 = pe1
                    ne2 = np.random.choice(candidate_dict[pe1])
                if pe1 != ne1 or pe2 != ne2:
                    train_index.append([pe1, pe2, ne1, ne2])
        np.random.shuffle(train_index)
        self.train_index = train_index
        self.batch_num = int(np.ceil(len(self.train_index) / self.batch_size))
        logger.info(f"[BATCH_GEN] Training indices: {len(train_index)} pairs, {skipped} skipped, {self.batch_num} batches")

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_count < self.batch_num:
            batch_index = self.iter_count
            self.iter_count += 1
            batch_data = self.train_index[
                batch_index * self.batch_size : (batch_index + 1) * self.batch_size
            ]
            pe1s = [pe1 for pe1, _, _, _ in batch_data]
            pe2s = [pe2 for _, pe2, _, _ in batch_data]
            ne1s = [ne1 for _, _, ne1, _ in batch_data]
            ne2s = [ne2 for _, _, _, ne2 in batch_data]
            return pe1s, pe2s, ne1s, ne2s
        self.iter_count = 0
        raise StopIteration
