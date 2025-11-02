"""Tokenisation helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, Tuple

from transformers import AutoTokenizer


@lru_cache(maxsize=8)
def _load_tokenizer(name: str, cache_dir: str | None = None):
    return AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)


def encode_entities(
    encoder_name: str,
    entity_texts: Dict[str, str],
    entity_order: Iterable[str],
    *,
    max_length: int,
    cache_dir: str | None,
) -> Tuple:
    tokenizer = _load_tokenizer(encoder_name, cache_dir)
    texts = [entity_texts[entity] for entity in entity_order]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"], tokenizer


def normalise_uri(uri: str) -> str:
    parts = uri.split("/")
    token = parts[-1] if parts else uri
    token = token.replace("_", " ")
    return token
