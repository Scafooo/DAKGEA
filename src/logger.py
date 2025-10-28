import logging
import sys
from pathlib import Path
from src.config.loader import Config

class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[37m",
        "INFO": "\033[36m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        levelname = f"{color}{record.levelname:<8}{self.RESET}"
        filename = Path(record.pathname).name
        lineno = record.lineno
        message = record.getMessage()
        return f"{levelname} | {filename}:{lineno} | {message}"

def get_logger(name=__name__, level=logging.INFO, log_file=None):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)

        # Console handler (colored)
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        ch.setFormatter(ColorFormatter())
        logger.addHandler(ch)

        # File handler (ensure dir exists)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)  # ✅ crea la cartella
            fh = logging.FileHandler(log_path)
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            logger.addHandler(fh)

    return logger

# Global default logger
cfg = Config().get()
logger = get_logger(
    "KG_EA",
    level=cfg["logging"]["level"],
    log_file=cfg["paths"]["log_file"]
)
