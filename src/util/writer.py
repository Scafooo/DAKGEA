import os
from pathlib import Path
from typing import Iterable, Sequence, Union

from src.logger import get_logger

logger = get_logger(__name__, level="DEBUG")

def build_path(root: Union[str, os.PathLike], filename: str) -> str:
    """Join a root directory and filename into a filesystem path."""
    return os.path.join(root, filename)


def write_tsv(file_path: Union[str, os.PathLike], rows: Iterable[Union[Sequence, str]]) -> None:
    """Write rows to a TSV file, creating parent directories if required."""
    destination = Path(file_path)
    logger.debug("Writing TSV file to %s", destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as f:
        for row in rows:
            if isinstance(row, (list, tuple)):
                f.write("\t".join(map(str, row)) + "\n")
            else:
                f.write(str(row) + "\n")
