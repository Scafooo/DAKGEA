"""Implementation of the basic BERT unit encoder used by BERT-INT."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from src.logger import get_logger

logger = get_logger(__name__)


class BasicBertUnitModel(nn.Module):
    """Thin wrapper around HuggingFace BERT to obtain entity embeddings."""

    def __init__(
        self,
        encoder_name: str,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        load_strategy: str = "auto",
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.cache_dir = self._resolve_cache_dir(cache_dir)
        self.encoder = self._load_encoder(encoder_name, input_dim, load_strategy)
        self.output_layer = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=token_ids, attention_mask=attention_mask)
        sequence_output, _ = outputs[:2]
        cls_repr = sequence_output[:, 0]
        hidden = self.dropout(cls_repr)
        projected = self.output_layer(hidden)
        return projected

    def _load_encoder(self, encoder_name: str, input_dim: int, load_strategy: str) -> BertModel:
        strategy = (load_strategy or "auto").lower()
        if strategy not in {"auto", "local", "snapshot", "download", "random"}:
            logger.warning(
                "[BERT-INT] Unknown encoder load strategy '%s'; defaulting to 'auto'.",
                load_strategy,
            )
            strategy = "auto"

        if strategy in {"auto", "local", "snapshot"}:
            encoder = self._try_load_local(encoder_name)
            if encoder is not None:
                return encoder

        if strategy in {"auto", "snapshot"}:
            encoder = self._try_load_snapshot(encoder_name)
            if encoder is not None:
                return encoder

        if strategy in {"auto", "download"}:
            encoder = self._try_download(encoder_name)
            if encoder is not None:
                return encoder

        logger.warning(
            "[BERT-INT] Falling back to randomly initialised encoder for '%s' (strategy=%s).",
            encoder_name,
            strategy,
        )
        return self._build_random_encoder(input_dim)

    def _try_load_local(self, encoder_name: str) -> Optional[BertModel]:
        try:
            encoder = BertModel.from_pretrained(
                encoder_name,
                local_files_only=True,
                cache_dir=self.cache_dir,
            )
            logger.debug("[BERT-INT] Loaded pretrained encoder '%s' from local cache.", encoder_name)
            return encoder
        except OSError as exc:
            logger.info("[BERT-INT] Pretrained encoder '%s' not cached locally (%s).", encoder_name, exc)
            return None

    def _try_load_snapshot(self, encoder_name: str) -> Optional[BertModel]:
        local_dir = self._maybe_download_snapshot(encoder_name)
        if not local_dir:
            return None
        try:
            encoder = BertModel.from_pretrained(local_dir, cache_dir=self.cache_dir)
            logger.debug("[BERT-INT] Loaded pretrained encoder '%s' from snapshot cache.", encoder_name)
            return encoder
        except OSError as exc:
            logger.warning(
                "[BERT-INT] Snapshot download for encoder '%s' is unusable (%s).",
                encoder_name,
                exc,
            )
            return None

    def _try_download(self, encoder_name: str) -> Optional[BertModel]:
        try:
            encoder = BertModel.from_pretrained(encoder_name, cache_dir=self.cache_dir)
            logger.debug("[BERT-INT] Downloaded pretrained encoder '%s'.", encoder_name)
            return encoder
        except OSError as exc:
            logger.warning(
                "[BERT-INT] Direct download for encoder '%s' failed (%s).",
                encoder_name,
                exc,
            )
            return None

    def _resolve_cache_dir(self, explicit: Optional[str]) -> Optional[str]:
        if explicit:
            return explicit
        for var in ("DAKGEA_HF_CACHE", "HF_HOME", "TRANSFORMERS_CACHE"):
            if var in os.environ:
                return os.environ[var]
        return None

    def _maybe_download_snapshot(self, model_name: str) -> Optional[str]:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.debug("[BERT-INT] huggingface_hub not available; skipping snapshot download for '%s'.", model_name)
            return None

        try:
            local_dir = snapshot_download(repo_id=model_name, cache_dir=self.cache_dir, local_files_only=False)
            logger.info("[BERT-INT] Snapshot downloaded for '%s' into '%s'.", model_name, local_dir)
            return local_dir
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("[BERT-INT] Snapshot download for '%s' failed (%s).", model_name, exc)
            return None

    @staticmethod
    def _build_random_encoder(input_dim: int) -> BertModel:
        heads = max(1, input_dim // 64)
        while input_dim % heads != 0 and heads > 1:
            heads -= 1
        if input_dim % heads != 0:
            heads = 1
        intermediate = max(input_dim * 4, 256)
        config = BertConfig(
            vocab_size=30522,
            hidden_size=input_dim,
            num_hidden_layers=2,
            num_attention_heads=heads,
            intermediate_size=intermediate,
        )
        return BertModel(config)
