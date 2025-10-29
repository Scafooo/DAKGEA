"""Text extraction and tokenisation utilities for BERT-INT."""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from rdflib import Graph, Literal, URIRef

from src.logger import get_logger

logger = get_logger(__name__)


def normalise_uri(uri: str) -> str:
    """Convert a URI into a readable label by stripping namespaces."""

    candidate = uri.split("/")[-1]
    candidate = candidate.split("#")[-1]
    candidate = candidate.replace("_", " ")
    if not candidate:
        return uri
    return candidate


def _select_best_literal(values: Sequence[str]) -> str:
    """Pick the most informative literal among the available ones."""

    if not values:
        return ""
    # Prefer shorter literals (names) over long descriptions
    sorted_vals = sorted(values, key=lambda v: (len(v), v))
    return sorted_vals[0]


def _collect_literals(graph: Graph) -> Dict[str, List[str]]:
    literals: Dict[str, List[str]] = {}
    for subj, _, obj in graph.triples((None, None, None)):
        if isinstance(subj, URIRef) and isinstance(obj, Literal):
            literals.setdefault(str(subj), []).append(str(obj))
    return literals


def extract_entity_texts(
    graph: Graph,
    focus_entities: Iterable[URIRef],
) -> Dict[str, str]:
    """Build human-readable text for the requested entities."""

    literals = _collect_literals(graph)
    texts: Dict[str, str] = {}
    for entity in focus_entities:
        uri = str(entity)
        options = literals.get(uri, [])
        label = _select_best_literal(options)
        if not label:
            label = normalise_uri(uri)
        texts[uri] = label
    return texts


def encode_entities(
    tokenizer_name: str,
    entity_texts: Dict[str, str],
    entity_order: Sequence[str],
    *,
    max_length: int = 128,
    cache_dir: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, object]:
    """Tokenise entity texts and return tensors ready for BERT."""

    from transformers import AutoTokenizer  # local import to keep module light

    class HashingTokenizer:
        """Fallback tokenizer that hashes tokens into a fixed vocabulary."""

        def __init__(self, vocab_size: int = 30522, pad_token_id: int = 0):
            self.vocab_size = vocab_size
            self.pad_token_id = pad_token_id
            self.unk_token_id = 1

        def _token_to_id(self, token: str) -> int:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            hashed = int(digest, 16)
            return 2 + (hashed % max(1, self.vocab_size - 2))

        def __call__(
            self,
            sentences: Sequence[str],
            padding: str | bool = "max_length",
            truncation: bool = True,
            max_length: int = 128,
            return_tensors: str | None = None,
        ) -> Dict[str, torch.Tensor]:
            if return_tensors not in (None, "pt"):
                raise ValueError("HashingTokenizer only supports return_tensors='pt'")

            encoded: List[Tuple[List[int], List[int]]] = []
            for sentence in sentences:
                text = (sentence or "").strip().lower()
                tokens = text.split() or ["[UNK]"]
                ids = [self._token_to_id(tok) for tok in tokens]
                if truncation and len(ids) > max_length:
                    ids = ids[:max_length]
                attention = [1] * len(ids)
                encoded.append((ids, attention))

            if padding == "max_length":
                target_len = max_length
            elif padding is True:
                target_len = max((len(ids) for ids, _ in encoded), default=max_length)
            else:
                target_len = None

            input_ids: List[List[int]] = []
            attention_masks: List[List[int]] = []
            for ids, attention in encoded:
                if target_len is not None:
                    pad_len = max(0, target_len - len(ids))
                    ids = ids + [self.pad_token_id] * pad_len
                    attention = attention + [0] * pad_len
                    if len(ids) > target_len:
                        ids = ids[:target_len]
                        attention = attention[:target_len]
                input_ids.append(ids)
                attention_masks.append(attention)

            input_tensor = torch.tensor(input_ids, dtype=torch.long)
            attention_tensor = torch.tensor(attention_masks, dtype=torch.long)
            return {"input_ids": input_tensor, "attention_mask": attention_tensor}

    resolved_cache_dir = cache_dir or _resolve_cache_dir()

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            local_files_only=True,
            cache_dir=resolved_cache_dir,
        )
        logger.debug("[BERT-INT] Loaded tokenizer '%s' from local cache.", tokenizer_name)
    except OSError as exc:
        logger.info("[BERT-INT] Tokenizer '%s' not cached locally (%s).", tokenizer_name, exc)

    if tokenizer is None:
        local_dir = _maybe_download_snapshot(tokenizer_name, resolved_cache_dir)
        if local_dir:
            try:
                tokenizer = AutoTokenizer.from_pretrained(local_dir, cache_dir=resolved_cache_dir)
                logger.debug("[BERT-INT] Loaded tokenizer '%s' from snapshot cache.", tokenizer_name)
            except OSError as exc:
                logger.warning(
                    "[BERT-INT] Snapshot download for tokenizer '%s' is unusable (%s).",
                    tokenizer_name,
                    exc,
                )

    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=resolved_cache_dir)
            logger.debug("[BERT-INT] Downloaded tokenizer '%s'.", tokenizer_name)
        except OSError as exc:
            logger.warning(
                "[BERT-INT] Could not load tokenizer '%s' (%s); using hashing fallback.",
                tokenizer_name,
                exc,
            )
            tokenizer = HashingTokenizer()
    sentences = [entity_texts[eid] for eid in entity_order]
    encoded = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"], tokenizer


def _maybe_download_snapshot(repo_id: str, cache_dir: Optional[str]) -> Optional[str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.debug("[BERT-INT] huggingface_hub not available; skipping snapshot download for '%s'.", repo_id)
        return None

    try:
        local_dir = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, local_files_only=False)
        logger.info("[BERT-INT] Snapshot downloaded for '%s' into '%s'.", repo_id, local_dir)
        return local_dir
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("[BERT-INT] Snapshot download for '%s' failed (%s).", repo_id, exc)
        return None


def _resolve_cache_dir() -> Optional[str]:
    for var in ("DAKGEA_HF_CACHE", "HF_HOME", "TRANSFORMERS_CACHE"):
        if var in os.environ:
            return os.environ[var]
    return None
