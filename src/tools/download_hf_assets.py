"""Download HuggingFace models/tokenizers declared in the BERT-INT configuration."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, BertModel

from src.alignment_models.methods.bert_int.config import BertIntConfig
from src.config.loader import PROJECT_ROOT, load_yaml
from src.logger import get_logger, set_global_level

logger = get_logger(__name__)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "config/models/bert_int.yaml",
        help="Path to the BERT-INT model YAML configuration.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Override the encoder load strategy ('auto', 'snapshot', 'download', 'local', 'random').",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Explicit cache directory for downloaded assets.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ...).",
    )
    return parser.parse_args(argv)


def resolve_cache_dir(explicit: Path | None, cfg_cache: Optional[str]) -> Path | None:
    if explicit:
        return explicit
    if cfg_cache:
        cache_path = Path(cfg_cache)
        if not cache_path.is_absolute():
            cache_path = (PROJECT_ROOT / cache_path).resolve()
        return cache_path
    for var in ("DAKGEA_HF_CACHE", "HF_HOME", "TRANSFORMERS_CACHE"):
        env_path = os.environ.get(var)
        if env_path:
            return Path(env_path)
    return None


def ensure_parent(path: Path | None) -> None:
    if path is None:
        return
    path.mkdir(parents=True, exist_ok=True)


def snapshot_download(repo_id: str, cache_dir: Path | None) -> Path | None:
    try:
        from huggingface_hub import snapshot_download as hub_snapshot_download
    except ImportError:
        logger.warning(
            "huggingface_hub is not installed; skipping snapshot download for '%s'.",
            repo_id,
        )
        return None

    ensure_parent(cache_dir)
    try:
        local_dir = hub_snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=False,
        )
        logger.info("Snapshot downloaded for '%s' into '%s'.", repo_id, local_dir)
        return Path(local_dir)
    except Exception as exc:  # pragma: no cover - network / auth dependent
        logger.error("Snapshot download for '%s' failed: %s", repo_id, exc)
        return None


def warm_transformers_cache(repo_id: str, cache_dir: Path | None) -> None:
    ensure_parent(cache_dir)
    cache_path = str(cache_dir) if cache_dir else None
    logger.debug("Loading BertModel.from_pretrained('%s') (cache_dir=%s).", repo_id, cache_path)
    BertModel.from_pretrained(repo_id, cache_dir=cache_path)
    logger.debug("Loading AutoTokenizer.from_pretrained('%s') (cache_dir=%s).", repo_id, cache_path)
    AutoTokenizer.from_pretrained(repo_id, cache_dir=cache_path)
    logger.info("Transformers cache warmed for '%s'.", repo_id)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    set_global_level(args.log_level)

    model_cfg = load_yaml(args.model_config).get("model", {})
    config = BertIntConfig.from_dict(model_cfg)

    strategy = args.strategy or config.basic_unit.encoder_strategy
    strategy = (strategy or "auto").lower()
    if strategy == "random":
        logger.info("Encoder strategy set to 'random'; nothing to download.")
        return 0

    cache_dir = resolve_cache_dir(args.cache_dir, config.paths.cache_dir)
    repo_id = config.basic_unit.encoder_name

    if strategy == "local":
        logger.info("Strategy 'local' selected; verifying local cache for '%s'.", repo_id)
        try:
            BertModel.from_pretrained(repo_id, cache_dir=str(cache_dir) if cache_dir else None, local_files_only=True)
            AutoTokenizer.from_pretrained(repo_id, cache_dir=str(cache_dir) if cache_dir else None, local_files_only=True)
            logger.info("Local cache already contains '%s'.", repo_id)
            return 0
        except OSError:
            logger.error("Local cache missing files for '%s'. Consider using strategy 'snapshot' or 'download'.", repo_id)
            return 1

    if strategy in {"auto", "snapshot"}:
        local_dir = snapshot_download(repo_id, cache_dir)
        if local_dir is None and strategy == "snapshot":
            logger.error("Snapshot strategy requested but download failed.")
            return 1

    if strategy in {"auto", "snapshot", "download"}:
        try:
            warm_transformers_cache(repo_id, cache_dir)
        except OSError as exc:
            logger.error("Unable to cache transformers resources for '%s': %s", repo_id, exc)
            return 1

    logger.info("HuggingFace assets ready for repo '%s'.", repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
