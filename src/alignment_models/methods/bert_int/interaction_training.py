"""Interaction model training mirroring the original BERT-INT implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.logger import get_logger

logger = get_logger(__name__)

Pair = Tuple[int, int]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.dense2 = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.xavier_normal_(self.dense1.weight)
        nn.init.xavier_normal_(self.dense2.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.dense1(features))
        x = torch.tanh(self.dense2(x))
        return torch.squeeze(x, dim=1)


class TrainIndexGenerator:
    def __init__(
        self,
        train_pairs: Sequence[Pair],
        train_candidate: Dict[int, Sequence[int]],
        entpair2f_idx: Dict[Pair, int],
        neg_num: int,
        batch_size: int,
    ) -> None:
        self.train_pairs = list(train_pairs)
        self.train_candidate = {k: np.array(v) for k, v in train_candidate.items()}
        self.entpair2f_idx = entpair2f_idx
        self.neg_num = neg_num
        self.batch_size = batch_size
        self.iter_count = 0
        self.train_pair_indexs, self.batch_num = self._generate_indices()

    def _generate_indices(self) -> Tuple[List[Tuple[Pair, Pair]], int]:
        train_pair_indexs: List[Tuple[Pair, Pair]] = []
        for pe1, pe2 in self.train_pairs:
            if pe1 not in self.train_candidate or pe2 not in self.train_candidate:
                continue
            neg_indexs = np.random.randint(len(self.train_candidate[pe1]), size=self.neg_num)
            for idx in neg_indexs:
                ne2 = int(self.train_candidate[pe1][idx])
                if ne2 == pe2:
                    continue
                train_pair_indexs.append(((pe1, pe2), (pe1, ne2)))
        np.random.shuffle(train_pair_indexs)
        batch_num = int(np.ceil(len(train_pair_indexs) * 1.0 / self.batch_size))
        return train_pair_indexs, batch_num

    def __iter__(self) -> "TrainIndexGenerator":
        return self

    def __next__(self) -> Tuple[List[int], List[int]]:
        if self.iter_count < self.batch_num:
            batch_index = self.iter_count
            self.iter_count += 1
            batch_ids = self.train_pair_indexs[
                batch_index * self.batch_size : (batch_index + 1) * self.batch_size
            ]
            pos_pairs = [pos for pos, _ in batch_ids]
            neg_pairs = [neg for _, neg in batch_ids]

            pos_f_ids = [self.entpair2f_idx[pair_id] for pair_id in pos_pairs]
            neg_f_ids = [self.entpair2f_idx[pair_id] for pair_id in neg_pairs]
            return pos_f_ids, neg_f_ids

        self.iter_count = 0
        self.train_pair_indexs, self.batch_num = self._generate_indices()
        raise StopIteration()


@dataclass
class InteractionArtifacts:
    scores: Dict[int, List[Tuple[int, float]]]


def _compute_metrics(
    model: nn.Module,
    test_candidate: Dict[int, Sequence[int]],
    test_pairs: Sequence[Pair],
    entpair2f_idx: Dict[Pair, int],
    feature_emb: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Dict[int, List[Tuple[int, float]]]:
    model.eval()
    test_pairs_all: List[Pair] = []
    for src, candidates in test_candidate.items():
        for tgt in candidates:
            test_pairs_all.append((src, tgt))

    scores: List[float] = []
    for start in range(0, len(test_pairs_all), batch_size):
        batch_pairs = test_pairs_all[start : start + batch_size]
        batch_f_ids = [entpair2f_idx[pair] for pair in batch_pairs]
        batch_features = feature_emb[torch.LongTensor(batch_f_ids)].to(device)
        batch_scores = model(batch_features)
        scores.extend(batch_scores.detach().cpu().tolist())

    scores_dict: Dict[int, List[Tuple[int, float]]] = {}
    for (src, tgt), score in zip(test_pairs_all, scores):
        scores_dict.setdefault(src, []).append((tgt, score))

    for src in scores_dict:
        scores_dict[src].sort(key=lambda item: item[1], reverse=True)
    return scores_dict


def train_interaction_model(
    features: Sequence[Sequence[float]],
    entity_pairs: Sequence[Pair],
    train_pairs: Sequence[Pair],
    test_pairs: Sequence[Pair],
    train_candidates: Dict[int, Sequence[int]],
    test_candidates: Dict[int, Sequence[int]],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    margin: float,
    neg_num: int,
    candidate_topk: int,
    device: torch.device,
) -> InteractionArtifacts:
    feature_emb = torch.FloatTensor(features)
    entpair2f_idx = {pair: idx for idx, pair in enumerate(entity_pairs)}

    generator = TrainIndexGenerator(
        train_pairs,
        train_candidates,
        entpair2f_idx,
        neg_num=neg_num,
        batch_size=batch_size,
    )

    model = MLP(len(features[0]), hidden_dim=11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MarginRankingLoss(margin=margin)

    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        for pos_f_ids, neg_f_ids in generator:
            optimizer.zero_grad()
            pos_feature = feature_emb[torch.LongTensor(pos_f_ids)].to(device)
            neg_feature = feature_emb[torch.LongTensor(neg_f_ids)].to(device)
            p_score = model(pos_feature).unsqueeze(-1)
            n_score = model(neg_feature).unsqueeze(-1)
            label_y = torch.ones(p_score.shape, device=device)
            batch_loss = criterion(p_score, n_score, label_y)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item() * p_score.size(0)
            batches += 1
        logger.debug(
            "[BERT-INT] Interaction epoch %d loss %.4f",
            epoch + 1,
            epoch_loss / max(1, batches),
        )

    scores = _compute_metrics(
        model,
        test_candidates,
        test_pairs,
        entpair2f_idx,
        feature_emb,
        batch_size=2048,
        device=device,
    )
    return InteractionArtifacts(scores=scores)
