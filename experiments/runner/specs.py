"""Dataclasses representing dataset and writer configuration for experiment runs."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DatasetSpec:
    """Configuration describing a dataset entry in the experiment suite.

    Two modes:
    1. Standard mode (name): Reader inferred from data/raw/{reader}/{name}
    2. Direct mode (path): Read dataset directly from specified path, skip reduction/writer

    The reader type is inferred from the dataset path structure:
    - data/raw/hybea/{name} -> reader=hybea
    - /path/to/hybea/dataset -> reader=hybea
    """

    name: str  # Dataset name or identifier
    reader: str  # Inferred from path if not specified
    subtype: Optional[str] = None  # Deprecated: inferred from path
    writer_conf: Optional[Any] = None
    direct_path: Optional[str] = None  # If set, read directly from this path (skip reduction/writer)


@dataclass
class WriterPlan:
    """Settings controlling how intermediate artefacts and results are persisted."""

    name: str
    writer: Any
    write_reduced: bool = False
    write_augmented: bool = False
    write_results: bool = False
