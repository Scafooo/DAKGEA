import csv
import os
from pathlib import Path
from typing import Iterable, List, Union

from src.logger import get_logger

logger = get_logger(__name__, level="DEBUG")

__all__ = [
    "read_tsv",
    "read_valid_pairs",
    "read_sup_pairs",
    "read_ref_pairs",
]

def read_tsv(file_path: Union[str, os.PathLike]) -> List[List[str]]:
    """Load a TSV file as a list of rows, logging the operation for traceability."""
    destination = Path(file_path)
    logger.debug("Reading TSV file from %s", destination)
    with destination.open(newline="", encoding="utf-8") as f:
        reader: Iterable[List[str]] = csv.reader(
            f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
        )
        return [row for row in reader]

read_valid_pairs = read_tsv
# Backwards compatibility for historical typo.
read_vaild_pairs = read_valid_pairs
read_sup_pairs = read_tsv
read_ref_pairs = read_tsv
