"""Interaction model training adapted from the original BERT-INT implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.alignment_models.methods.bert_int.similarity import batched_topk
from src.logger import get_logger

logger = get_logger(__name__)


Pair = Tuple[int, int]


class EarlyStopping:
    """Early stopping to avoid overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            loss: Current loss value

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


class InteractionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.dense1.weight)
        nn.init.xavier_normal_(self.dense2.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.dense1(features))
        x = torch.tanh(self.dense2(x))
        return x.squeeze(-1)


class TrainIndexGenerator:
    def __init__(
        self,
        train_pairs: Sequence[Pair],
        train_candidate: Dict[int, Sequence[int]],
        pair_index: Dict[Pair, int],
        neg_num: int,
        batch_size: int,
    ) -> None:
        self.train_pairs = list(train_pairs)
        self.candidates = {k: np.array(v) for k, v in train_candidate.items() if v}
        self.pair_index = pair_index
        self.neg_num = neg_num
        self.batch_size = batch_size
        self.indices: List[Tuple[int, int]] = []
        self.iter = 0
        self._regenerate()

    def _regenerate(self):
        pairs = []
        for src, tgt in self.train_pairs:
            if src not in self.candidates:
                continue
            # Use randint to sample indices, then get candidates (matches original BERT-INT)
            neg_indices = np.random.randint(len(self.candidates[src]), size=self.neg_num)
            neg_samples = self.candidates[src][neg_indices].tolist()
            for neg in neg_samples:
                if neg == tgt:
                    continue
                pos_idx = self.pair_index[(src, tgt)]
                neg_idx = self.pair_index.get((src, neg))
                if neg_idx is None:
                    continue
                pairs.append((pos_idx, neg_idx))
        np.random.shuffle(pairs)
        self.indices = pairs
        self.iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter < len(self.indices):
            batch = self.indices[self.iter : self.iter + self.batch_size]
            self.iter += self.batch_size
            pos_idx = [p for p, _ in batch]
            neg_idx = [n for _, n in batch]
            return pos_idx, neg_idx
        self._regenerate()
        raise StopIteration


@dataclass
class InteractionArtifacts:
    scores: Dict[int, List[Tuple[int, float]]]


def train_interaction_model(
    features: List[List[float]],
    entity_pairs: List[Pair],
    train_pairs: Sequence[Pair],
    test_pairs: Sequence[Pair],
    train_candidates: Dict[int, List[int]],
    test_candidates: Dict[int, List[int]],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    margin: float,
    neg_num: int,
    candidate_topk: int,
    device: torch.device,
) -> InteractionArtifacts:
    logger.info(
        "[BERT-INT] Interaction training: features=%d dim=%d train_pairs=%d test_pairs=%d",
        len(features),
        len(features[0]) if features else 0,
        len(train_pairs),
        len(test_pairs),
    )
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    pair_index = {pair: idx for idx, pair in enumerate(entity_pairs)}

    model = InteractionMLP(feature_tensor.shape[1], hidden_dim=11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MarginRankingLoss(margin=margin)
    early_stopping = EarlyStopping(patience=10, min_delta=1e-5)

    generator = TrainIndexGenerator(train_pairs, train_candidates, pair_index, neg_num, batch_size)

    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        for pos_idx, neg_idx in generator:
            if not pos_idx:
                continue
            optimizer.zero_grad()
            pos_feat = feature_tensor[pos_idx].to(device)
            neg_feat = feature_tensor[neg_idx].to(device)
            pos_score = model(pos_feat).unsqueeze(-1)
            neg_score = model(neg_feat).unsqueeze(-1)
            # label=1 means pos_score should be ranked higher than neg_score
            label = torch.ones_like(pos_score)
            loss = criterion(pos_score, neg_score, label)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1

        avg_loss = epoch_loss / max(1, steps)
        logger.info(
            "[BERT-INT] Interaction epoch %d/%d loss=%.4f",
            epoch + 1,
            epochs,
            avg_loss,
        )

        # Early stopping check
        if early_stopping(avg_loss):
            logger.info("[BERT-INT] Early stopping triggered at epoch %d/%d", epoch + 1, epochs)
            break

    model.eval()
    with torch.no_grad():
        scores = model(feature_tensor.to(device)).cpu().numpy()

    logger.info("[BERT-INT] Model produced %d scores for %d entity pairs", len(scores), len(entity_pairs))

    score_map: Dict[int, List[Tuple[int, float]]] = {}
    for (src, tgt), score in zip(entity_pairs, scores):
        score_map.setdefault(src, []).append((tgt, float(score)))

    logger.info("[BERT-INT] Score map has %d source entities before filtering", len(score_map))

    # keep top-k for evaluation
    for src in score_map:
        score_map[src].sort(key=lambda x: x[1], reverse=True)
        score_map[src] = score_map[src][:candidate_topk]

    logger.info("[BERT-INT] Interaction training complete; scored %d source entities with top-%d candidates.",
                len(score_map), candidate_topk)

    return InteractionArtifacts(scores=score_map)
