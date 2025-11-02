"""Utility helpers to prepare BERT-INT training data from DAKGEA datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoTokenizer

from src.core.dataset import Dataset
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreparedBertIntData:
    train_pairs: List[Tuple[str, str]]
    test_pairs: List[Tuple[str, str]]
    left_entities: List[str]
    right_entities: List[str]
    entity_inputs: Dict[str, Dict[str, torch.Tensor]]
    entity_to_side: Dict[str, str]
    tokenizer_name: str
    max_length: int
    seed: int


def _friendly_name(uri: str) -> str:
    if not uri:
        return ""
    if "resource/" in uri:
        name = uri.split("resource/")[-1]
    elif "property/" in uri:
        name = uri.split("property/")[-1]
    else:
        name = uri.rsplit("/", 1)[-1]
    return name.replace("_", " ")


def _build_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    except Exception as exc:
        logger.warning(
            "[BERT-INT] Falling back to local tokenizer files for '%s' (%s)",
            model_name,
            exc,
        )
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)


def _encode_entities(
    tokenizer,
    entities: Sequence[str],
    max_length: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    encoded: Dict[str, Dict[str, torch.Tensor]] = {}
    for entity in entities:
        text = _friendly_name(entity)
        if not text:
            text = entity
        encoded_example = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        encoded[entity] = {
            "input_ids": encoded_example["input_ids"].squeeze(0),
            "attention_mask": encoded_example["attention_mask"].squeeze(0),
        }
    return encoded


def prepare_basic_unit_data(
    dataset: Dataset,
    *,
    train_ratio: float,
    max_length: int,
    tokenizer_name: str,
    seed: int,
) -> PreparedBertIntData:
    pairs = list(dataset.aligned_entities)
    if not pairs:
        raise ValueError("Dataset has no aligned entities for BERT-INT training.")

    rng = random.Random(seed)
    rng.shuffle(pairs)

    train_ratio = max(0.0, min(train_ratio, 1.0))
    total_pairs = len(pairs)
    if total_pairs == 1:
        split_idx = 1
    else:
        split_idx = int(total_pairs * train_ratio)
        split_idx = max(1, split_idx)
        split_idx = min(split_idx, total_pairs - 1)

    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:] if split_idx < len(pairs) else []

    left_entities = sorted({src for src, _ in pairs})
    right_entities = sorted({tgt for _, tgt in pairs})
    all_entities = left_entities + [e for e in right_entities if e not in left_entities]
    entity_to_side = {entity: "left" for entity in left_entities}
    entity_to_side.update({entity: "right" for entity in right_entities})

    tokenizer = _build_tokenizer(tokenizer_name)
    entity_inputs = _encode_entities(tokenizer, all_entities, max_length)

    return PreparedBertIntData(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        left_entities=left_entities,
        right_entities=right_entities,
        entity_inputs=entity_inputs,
        entity_to_side=entity_to_side,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        seed=seed,
    )
