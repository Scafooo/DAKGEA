"""Training loop for the BERT-INT basic unit."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from src.alignment_models.methods.bert_int.basic_unit.dataset import BasicUnitDataBundle
from src.alignment_models.methods.bert_int.basic_unit.generator import TrainingPairGenerator
from src.alignment_models.methods.bert_int.basic_unit.metrics import (
    batch_cosine_similarity,
    batch_topk,
    compute_hits,
)
from src.alignment_models.methods.bert_int.basic_unit.model import BasicBertUnit
from src.logger import get_logger

logger = get_logger(__name__)

Pair = Tuple[int, int]
DeviceSpec = Union[int, str, torch.device]


def _select_device(device_spec: Optional[DeviceSpec], fallback_cuda: int) -> torch.device:
    """
    Select PyTorch device based on specification with fallback logic.

    Args:
        device_spec: Device specification. Can be:
            - int: CUDA device index (e.g., 0 for cuda:0)
            - str: Device string (e.g., "cuda:0", "cpu")
            - torch.device: Direct device object
            - None: Use fallback_cuda if CUDA available, else CPU
        fallback_cuda: CUDA device index to use when device_spec is None

    Returns:
        torch.device: Selected PyTorch device

    Examples:
        >>> _select_device(0, fallback_cuda=0)
        device(type='cuda', index=0)
        >>> _select_device("cpu", fallback_cuda=0)
        device(type='cpu')
    """
    if device_spec is not None:
        if isinstance(device_spec, torch.device):
            return device_spec
        if isinstance(device_spec, int):
            if torch.cuda.is_available():
                return torch.device(f"cuda:{device_spec}")
            logger.warning("CUDA device %d requested but not available. Falling back to CPU.", device_spec)
            return torch.device("cpu")
        if isinstance(device_spec, str):
            if device_spec.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA requested via '%s' but not available. Falling back to CPU.", device_spec)
                return torch.device("cpu")
            return torch.device(device_spec)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{fallback_cuda}")
    return torch.device("cpu")


class BasicUnitTrainer:
    """Train and evaluate the BERT-INT basic unit."""

    def __init__(
        self,
        model: BasicBertUnit,
        config: Dict[str, Any],
        data: BasicUnitDataBundle,
        paths: Dict[str, Any],
        device_spec: Optional[int | str] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.data = data
        self.paths = paths
        self.device_spec = device_spec

        self.device = _select_device(self.device_spec, self.config.get("cuda_device", 0))
        self.model.to(self.device)

    def fit(self) -> List[Dict[str, float]]:
        """Train the model for the configured number of epochs, returning metrics per epoch."""
        self._set_seeds(int(self.config.get("seed", 11037)))

        learning_rate = float(self.config.get("learning_rate", 1.0e-5))
        weight_decay = float(self.config.get("weight_decay", 0.0))
        margin = float(self.config.get("margin", 1.0))

        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        criterion = torch.nn.MarginRankingLoss(margin=margin)

        ent_ids_left = [e1 for e1, _ in self.data.ent_ill]
        ent_ids_right = [e2 for _, e2 in self.data.ent_ill]
        generator = TrainingPairGenerator(
            train_ill=self.data.train_ill,
            ent_ids_left=ent_ids_left,
            ent_ids_right=ent_ids_right,
            batch_size=self.config.get("batch_size", 24),
            negatives_per_positive=self.config.get("negatives_per_positive", 2),
        )

        history: List[Dict[str, float]] = []
        epochs = self.config.get("epochs", 1)
        for epoch in range(epochs):
            logger.info("[BERT-INT] Epoch %d/%d", epoch + 1, epochs)

            candidate_dict = self._generate_candidate_dict(generator)
            generator.build_indices(candidate_dict)
            train_loss = self._train_epoch(generator, optimizer, criterion)

            metrics = self.evaluate(self.data.test_ill, batch_size=self.config.get("eval_batch_size", 128))
            metrics["loss"] = train_loss
            metrics["epoch"] = epoch + 1
            history.append(metrics)

            logger.debug(
                "[BERT-INT] Epoch %d metrics: loss=%.4f hits@1=%.4f hits@5=%.4f hits@10=%.4f mrr=%.4f",
                epoch + 1,
                train_loss,
                metrics.get("hits@1", 0.0),
                metrics.get("hits@5", 0.0),
                metrics.get("hits@10", 0.0),
                metrics.get("mrr", 0.0),
            )

            self._maybe_save(epoch + 1)
        return history

    def evaluate(self, pairs: Sequence[Pair], batch_size: Optional[int] = None) -> Dict[str, float]:
        """Evaluate the model on the provided alignment pairs."""
        if not pairs:
            return {"hits@1": 0.0, "hits@10": 0.0, "mrr": 0.0, "evaluated": 0}

        if batch_size is None:
            batch_size = self.config.get("eval_batch_size", 128)
        self.model.eval()
        with torch.no_grad():
            ent_left = [left for left, _ in pairs]
            ent_right = [right for _, right in pairs]
            left_emb = self._encode_entities(ent_left, batch_size)
            right_emb = self._encode_entities(ent_right, batch_size)

            sim_matrix = batch_cosine_similarity(
                left_emb,
                right_emb,
                batch_size=batch_size,
                device=self.device,
            )
            _, indices = batch_topk(
                sim_matrix,
                batch_size=batch_size,
                topk=min(self.config.get("eval_top_k", 1000), sim_matrix.size(1)),
                device=self.device,
            )
        return compute_hits(indices)

    def _train_epoch(self, generator: TrainingPairGenerator, optimizer, criterion) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for pos_left, pos_right, neg_left, neg_right in generator:
            optimizer.zero_grad()

            pos_emb_left = self._forward_entities(pos_left)
            pos_emb_right = self._forward_entities(pos_right)
            neg_emb_left = self._forward_entities(neg_left)
            neg_emb_right = self._forward_entities(neg_right)

            pos_score = F.pairwise_distance(pos_emb_left, pos_emb_right, p=1, keepdim=True)
            neg_score = F.pairwise_distance(neg_emb_left, neg_emb_right, p=1, keepdim=True)
            target = -torch.ones_like(pos_score, device=self.device)
            loss = criterion(pos_score, neg_score, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get("max_grad_norm", 1.0),
            )
            optimizer.step()

            batch_size = pos_score.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / max(total_samples, 1)

    def _generate_candidate_dict(self, generator: TrainingPairGenerator) -> Dict[int, List[int]]:
        train_left = [e1 for e1, _ in self.data.train_ill]
        train_right = [e2 for _, e2 in self.data.train_ill]

        candidates_left = generator.ent_ids_left
        candidates_right = generator.ent_ids_right

        candidate_batch = self.config.get("candidate_batch_size", 128)
        nearest = self.config.get("nearest_sample_num", 128)

        emb_left_train = self._encode_entities(train_left, candidate_batch)
        emb_right_candidates = self._encode_entities(candidates_right, candidate_batch)
        sim_left = batch_cosine_similarity(
            emb_left_train,
            emb_right_candidates,
            batch_size=candidate_batch,
            device=self.device,
        )
        _, idx_left = batch_topk(
            sim_left,
            batch_size=candidate_batch,
            topk=nearest,
            device=self.device,
        )

        emb_right_train = self._encode_entities(train_right, candidate_batch)
        emb_left_candidates = self._encode_entities(candidates_left, candidate_batch)
        sim_right = batch_cosine_similarity(
            emb_right_train,
            emb_left_candidates,
            batch_size=candidate_batch,
            device=self.device,
        )
        _, idx_right = batch_topk(
            sim_right,
            batch_size=candidate_batch,
            topk=nearest,
            device=self.device,
        )

        candidate_dict: Dict[int, List[int]] = {}
        for pos, entity in enumerate(train_left):
            candidate_dict[entity] = [int(candidates_right[i]) for i in idx_left[pos].tolist()]
        for pos, entity in enumerate(train_right):
            candidate_dict[entity] = [int(candidates_left[i]) for i in idx_right[pos].tolist()]
        return candidate_dict

    def _forward_entities(self, entity_ids: Sequence[int]) -> torch.Tensor:
        if not entity_ids:
            return torch.empty(0, self.model.projection.out_features, device=self.device)

        token_batch, mask_batch = self._batchify(entity_ids)
        return self.model(token_batch, mask_batch)

    def _encode_entities(self, entity_ids: Sequence[int], batch_size: int) -> np.ndarray:
        if not entity_ids:
            return np.empty((0, self.model.projection.out_features), dtype=np.float32)

        embeddings: List[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(entity_ids), batch_size):
                batch_ids = entity_ids[start : start + batch_size]
                token_batch, mask_batch = self._batchify(batch_ids)
                outputs = self.model(token_batch, mask_batch).detach().cpu().numpy()
                embeddings.append(outputs)
        return np.vstack(embeddings)

    def _batchify(self, entity_ids: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.tensor(
            [self.data.ent2data[e_id][0] for e_id in entity_ids],
            dtype=torch.long,
            device=self.device,
        )
        masks = torch.tensor(
            [self.data.ent2data[e_id][1] for e_id in entity_ids],
            dtype=torch.float32,
            device=self.device,
        )
        return tokens, masks

    def _maybe_save(self, epoch: int) -> None:
        model_save_dir = self.paths.get("model_save_dir")
        if not model_save_dir:
            return
        save_dir = Path(model_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.paths.get("model_save_prefix") or "model"
        model_path = save_dir / f"{prefix}_epoch_{epoch}.pt"
        torch.save(self.model.state_dict(), model_path)

    @staticmethod
    def _set_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
