"""Logging utilities with colored console output and module-level helpers."""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

from src.config.loader import Config

LOG_NAMESPACE = "KG_EA"

_LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


class ColorFormatter(logging.Formatter):
    """Render log records with ANSI colors and concise metadata."""

    COLORS = {
        "DEBUG": "\033[37m",
        "INFO": "\033[36m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        levelname = f"{color}{record.levelname:<8}{self.RESET}"
        filename = Path(record.pathname).name
        message = record.getMessage()
        return f"{levelname} | {filename}:{record.lineno} | {message}"


def _parse_level(level: Union[str, int, None]) -> int:
    """Normalize a log level specification to its numeric value."""
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    try:
        return _LOG_LEVELS[level.upper()]
    except (KeyError, AttributeError):
        raise ValueError(f"Unsupported log level: {level}") from None


@lru_cache(maxsize=1)
def _default_settings() -> tuple[int, Optional[Path]]:
    """Resolve the default logging level and log file configured via YAML."""
    cfg = Config().get()
    level = cfg.get("logging", {}).get("level", "INFO")
    log_file = cfg.get("paths", {}).get("log_file")
    log_path = Path(log_file) if log_file else None
    return _parse_level(level), log_path


def _configure_root_logger() -> logging.Logger:
    """Ensure the root logger for the project is configured exactly once."""
    level, log_path = _default_settings()
    root = logging.getLogger(LOG_NAMESPACE)

    if not root.handlers:
        root.setLevel(level)

        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColorFormatter())
        root.addHandler(console_handler)

        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            root.addHandler(file_handler)

        root.propagate = False
    else:
        root.setLevel(level)
        for handler in root.handlers:
            handler.setLevel(level)

    return root


def get_logger(name: str, level: Union[str, int, None] = None) -> logging.Logger:
    """Return a child logger scoped under the global project namespace."""
    root = _configure_root_logger()
    qualified_name = f"{root.name}.{name}" if name else root.name
    logger = logging.getLogger(qualified_name)
    if level is not None:
        logger.setLevel(_parse_level(level))
    return logger


def set_global_level(level: Union[str, int]) -> None:
    """Override the global logging level at runtime."""
    numeric_level = _parse_level(level)
    root = _configure_root_logger()
    root.setLevel(numeric_level)
    for handler in root.handlers:
        handler.setLevel(numeric_level)
