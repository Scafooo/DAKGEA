"""Training utilities for the Basic BERT Unit."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup

from .basic_unit_model import BasicBertUnitModel
from .config import BertIntBasicUnitSettings
from .data import PreparedBertIntData
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BasicUnitArtifacts:
    epochs: int
    losses: List[float]
    checkpoint_path: Optional[Path]
    total_updates: int


def _set_random_seed(seed: int) -> random.Random:
    """Synchronise RNGs across Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return random.Random(seed)


def _build_training_entries(train_pairs: Sequence[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    """
    Create directional alignment pairs.

    Each aligned entity contributes two training samples: left→right and right→left.
    """
    entries: List[Tuple[str, str, str]] = []
    for left_ent, right_ent in train_pairs:
        entries.append(("left", left_ent, right_ent))
        entries.append(("right", right_ent, left_ent))
    return entries


def _sample_negative(
    pool: Sequence[str],
    positive: str,
    rng: random.Random,
) -> str:
    """Sample a negative entity distinct from ``positive`` when possible."""
    if len(pool) <= 1:
        return positive
    candidate = rng.choice(pool)
    if candidate == positive:
        candidate = rng.choice(pool)
    return candidate if candidate != positive else positive


def _stack_inputs(
    entity_ids: Sequence[str],
    entity_inputs: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return batched token ids and attention masks for the given entities."""
    input_tensors = []
    mask_tensors = []
    for entity_id in entity_ids:
        encoded = entity_inputs[entity_id]
        input_tensors.append(encoded["input_ids"])
        mask_tensors.append(encoded["attention_mask"])
    input_batch = torch.stack(input_tensors, dim=0).to(device=device, non_blocking=True).long()
    mask_batch = torch.stack(mask_tensors, dim=0).to(device=device, non_blocking=True).long()
    return input_batch, mask_batch


def _batch_iter(sequence: Sequence[Tuple[str, str, str]], batch_size: int):
    for idx in range(0, len(sequence), batch_size):
        yield sequence[idx : idx + batch_size]


def train_basic_unit_model(
    model: BasicBertUnitModel,
    data: PreparedBertIntData,
    settings: BertIntBasicUnitSettings,
    *,
    device: torch.device,
    cache_dir: Path,
    seed: int,
) -> BasicUnitArtifacts:
    """Fine-tune the Basic BERT Unit using margin-based contrastive learning."""
    if not data.train_pairs:
        logger.warning("[BERT-INT] No training pairs available; skipping Basic Unit fine-tuning.")
        return BasicUnitArtifacts(epochs=0, losses=[], checkpoint_path=None, total_updates=0)

    rng = _set_random_seed(seed)
    model.to(device)
    model.train()

    cache_dir.mkdir(parents=True, exist_ok=True)
    entries = _build_training_entries(data.train_pairs)
    total_batches = max(1, math.ceil(len(entries) / settings.batch_size))
    updates_per_epoch = max(1, math.ceil(total_batches / settings.gradient_accumulation))
    total_updates = updates_per_epoch * max(1, settings.epochs)
    warmup_steps = min(settings.warmup_steps, total_updates)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    losses: List[float] = []

    logger.info(
        "[BERT-INT] Training Basic Unit on %d aligned pairs (batches per epoch=%d, updates=%d, negatives/pos=%d)",
        len(data.train_pairs),
        total_batches,
        total_updates,
        settings.negatives_per_positive,
    )

    for epoch in range(settings.epochs):
        epoch_entries = entries[:]
        rng.shuffle(epoch_entries)
        epoch_loss = 0.0
        samples_observed = 0
        optimizer.zero_grad(set_to_none=True)

        for step_idx, batch_entries in enumerate(_batch_iter(epoch_entries, settings.batch_size), start=1):
            source_ids: List[str] = []
            pos_target_ids: List[str] = []
            neg_target_ids: List[str] = []

            for direction, source, target in batch_entries:
                pool = data.right_entities if direction == "left" else data.left_entities
                for _ in range(settings.negatives_per_positive):
                    negative = _sample_negative(pool, target, rng)
                    source_ids.append(source)
                    pos_target_ids.append(target)
                    neg_target_ids.append(negative)

            if not source_ids:
                continue

            src_inputs, src_masks = _stack_inputs(source_ids, data.entity_inputs, device)
            pos_inputs, pos_masks = _stack_inputs(pos_target_ids, data.entity_inputs, device)
            neg_inputs, neg_masks = _stack_inputs(neg_target_ids, data.entity_inputs, device)

            anchor_emb = model(src_inputs, src_masks)
            pos_emb = model(pos_inputs, pos_masks)
            neg_emb = model(neg_inputs, neg_masks)

            pos_dist = F.pairwise_distance(anchor_emb, pos_emb, p=2)
            neg_dist = F.pairwise_distance(anchor_emb, neg_emb, p=2)

            raw_loss = torch.relu(settings.margin + pos_dist - neg_dist).mean()
            loss = raw_loss / settings.gradient_accumulation
            loss.backward()

            batch_size_effective = len(source_ids)
            epoch_loss += raw_loss.item() * batch_size_effective
            samples_observed += batch_size_effective

            if step_idx % settings.gradient_accumulation == 0 or step_idx == total_batches:
                if settings.max_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), settings.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        mean_loss = epoch_loss / max(1, samples_observed)
        losses.append(mean_loss)
        logger.info(
            "[BERT-INT] Epoch %d/%d finished — loss=%.4f (samples=%d)",
            epoch + 1,
            settings.epochs,
            mean_loss,
            samples_observed,
        )

    checkpoint_path = cache_dir / "basic_unit.pt"
    torch.save(model.state_dict(), checkpoint_path)
    logger.info("[BERT-INT] Basic Unit checkpoint saved → %s", checkpoint_path)

    model.eval()

    return BasicUnitArtifacts(
        epochs=settings.epochs,
        losses=losses,
        checkpoint_path=checkpoint_path,
        total_updates=total_updates,
    )
