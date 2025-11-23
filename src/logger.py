"""Logging utilities with colored console output and module-level helpers."""

from __future__ import annotations

import logging
import os
import sys
from types import MethodType
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

from src.config.loader import Config

LOG_NAMESPACE = "KG_EA"
VERBOSE_LEVEL = logging.DEBUG - 5
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")

_LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "VERBOSE": VERBOSE_LEVEL,
    "NOTSET": logging.NOTSET,
}


class ColorFormatter(logging.Formatter):
    """Render log records with ANSI colors and concise metadata with improved formatting."""

    # ANSI color codes for different log levels
    COLORS = {
        "DEBUG": "\033[37m",      # White
        "INFO": "\033[36m",       # Cyan
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[41m",   # Red background
    }

    # ANSI color codes for file locations
    LOCATION_COLOR = "\033[35m"  # Magenta

    # ANSI color codes for separators
    SEPARATOR_COLOR = "\033[90m" # Dark gray

    # Highlight tags
    HIGHLIGHTS = {
        "[STEP]": "\033[95m",      # Magenta
        "[SUCCESS]": "\033[32m",   # Green
        "[IMPORTANT]": "\033[96m", # Bright cyan
    }

    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors, level, and detailed source location."""
        message = record.getMessage().rstrip()
        stripped_msg = message.strip()

        if not stripped_msg or all(c in '=- ' for c in stripped_msg):
            return message

        color = self.COLORS.get(record.levelname, self.RESET)
        levelname_text = f"[{record.levelname:<7}]"
        levelname = f"{color}{levelname_text}{self.RESET}"

        # Resolve relative path within the project (fallback to basename)
        try:
            project_root = Path(__file__).resolve().parents[2]
            rel_path = Path(record.pathname).resolve().relative_to(project_root)
        except Exception:
            rel_path = Path(record.pathname).name
        location = self._shorten_location(f"{rel_path}:{record.lineno}")
        location_colored = f"{self.LOCATION_COLOR}{location:<40}{self.RESET}"

        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        timestamp_text = f"{self.SEPARATOR_COLOR}{timestamp}{self.RESET}"

        formatted_message = message
        for tag, tag_color in self.HIGHLIGHTS.items():
            if formatted_message.startswith(tag):
                formatted_message = formatted_message.replace(
                    tag, f"{tag_color}{tag}{self.RESET}", 1
                )
                break

        separator = f"{self.SEPARATOR_COLOR}│{self.RESET}"
        return f"{timestamp_text} {separator} {levelname} {separator} {location_colored} {separator} {formatted_message}"

    @staticmethod
    def _shorten_location(text: str, width: int = 40) -> str:
        if len(text) <= width:
            return text
        return "…" + text[-(width - 1):]


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
            # Ensure all handlers use ColorFormatter for console output
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                if not isinstance(handler.formatter, ColorFormatter):
                    handler.setFormatter(ColorFormatter())

    return root


def get_logger(name: str, level: Union[str, int, None] = None) -> logging.Logger:
    """Return a child logger scoped under the global project namespace."""
    root = _configure_root_logger()
    qualified_name = f"{root.name}.{name}" if name else root.name
    logger = logging.getLogger(qualified_name)
    # Add a convenience verbose method using the custom VERBOSE level
    def verbose(self, message, *args, **kwargs):
        if self.isEnabledFor(VERBOSE_LEVEL):
            self._log(VERBOSE_LEVEL, message, args, **kwargs)

    if not hasattr(logger, "verbose"):
        logger.verbose = MethodType(verbose, logger)  # type: ignore[attr-defined]
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


class StructuredLogger:
    """Wrapper for structured logging with formatted output."""

    def __init__(self, logger: logging.Logger):
        """Initialize with a logger instance."""
        self.logger = logger
        # Get terminal width, default to 80 if not available
        try:
            self.terminal_width = os.get_terminal_size().columns
        except (AttributeError, ValueError, OSError):
            self.terminal_width = 80

    def section(self, title: str) -> None:
        """Log a section header."""
        self.logger.info("")
        self.logger.info(f"[STEP] {title.upper()}")

    def subsection(self, title: str) -> None:
        """Log a subsection header."""
        self.logger.info(f"[STEP] -> {title}")

    def table(self, title: str, data: dict, indent: int = 2) -> None:
        """Log a formatted table of key-value pairs."""
        indent_str = " " * indent
        self.logger.info(f"{indent_str}{title}:")

        if not data:
            self.logger.info(f"{indent_str}  (empty)")
            return

        # Calculate column widths
        max_key_len = max(len(str(k)) for k in data.keys()) if data else 0

        for key, value in data.items():
            formatted_key = str(key).ljust(max_key_len)
            self.logger.info(f"{indent_str}  {formatted_key} : {value}")

    def list_items(self, title: str, items: list, indent: int = 2) -> None:
        """Log a formatted list of items."""
        indent_str = " " * indent
        self.logger.info(f"{indent_str}{title}:")

        if not items:
            self.logger.info(f"{indent_str}  (empty)")
            return

        for i, item in enumerate(items, 1):
            self.logger.info(f"{indent_str}  {i}. {item}")

    def progress(self, message: str, current: int, total: int, indent: int = 2) -> None:
        """Log progress with percentage."""
        indent_str = " " * indent
        percentage = (current / total * 100) if total > 0 else 0
        bar_length = 30
        filled = int(bar_length * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_length - filled)

        # Format current and total with padding for alignment
        total_width = len(str(total))
        current_str = str(current).rjust(total_width)
        percentage_str = f"{percentage:5.1f}%"  # Right-align percentage to 5 chars (e.g., " 8.0%")

        self.logger.info(
            f"{indent_str}{message}:  [{bar}] {percentage_str} ({current_str}/{total})"
        )

    def success(self, message: str) -> None:
        """Log a success message."""
        self.logger.info(f"[SUCCESS] {message}")

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(f"[WARNING] {message}")

    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(f"[ERROR] {message}")

    def debug_dict(self, title: str, data: dict, indent: int = 2) -> None:
        """Log a dictionary in debug mode with nice formatting."""
        if self.logger.level > logging.DEBUG:
            return

        indent_str = " " * indent
        self.logger.debug(f"\n{indent_str}{title}:")

        for key, value in data.items():
            if isinstance(value, dict):
                self.logger.debug(f"{indent_str}  {key}:")
                for k, v in value.items():
                    self.logger.debug(f"{indent_str}    {k}: {v}")
            elif isinstance(value, (list, tuple)):
                self.logger.debug(f"{indent_str}  {key}: [{len(value)} items]")
            else:
                self.logger.debug(f"{indent_str}  {key}: {value}")

        self.logger.debug("")


def get_structured_logger(name: str, level: Union[str, int, None] = None) -> StructuredLogger:
    """Return a structured logger scoped under the global project namespace."""
    logger = get_logger(name, level)
    return StructuredLogger(logger)
