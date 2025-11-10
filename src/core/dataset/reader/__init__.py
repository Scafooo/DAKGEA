"""Dataset reader implementations and registries."""

from src.core.dataset.reader.dataset_reader_base import DatasetReader
from src.core.dataset.reader.dataset_reader_factory import DatasetReaderFactory

# Import readers to trigger registration
from src.core.dataset.reader.bert_int_reader import BertIntReader  # noqa: F401
from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader  # noqa: F401

__all__ = [
    "DatasetReader",
    "DatasetReaderFactory",
]
