"""Dataclasses representing dataset and writer configuration for experiment runs."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DatasetSpec:
    """Configuration describing a dataset entry in the experiment suite."""

    name: str
    reader: str
    subtype: str
    writer_conf: Optional[Any]


@dataclass
class WriterPlan:
    """Settings controlling how intermediate artefacts and results are persisted."""

    name: str
    writer: Any
    write_reduced: bool = False
    write_augmented: bool = False
    write_results: bool = False
