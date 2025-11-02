"""Unit tests for the refactored pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest

from experiments.runner.specs import WriterPlan
from experiments.runner.stages import AugmentationStage, EvaluationStage, ReductionStage
from src.alignment_models.registry import load_builtin_models
from src.augmentation.registry import load_builtin_augmentations
from src.reduction.registry import load_builtin_reducers
from src.core.dataset import Dataset
from src.core.knowledge_graph.knowledge_graph import KnowledgeGraph


class DummyWriter:
    file_type = "dummy"

    def __init__(self) -> None:
        self.writes: Dict[str, int] = {}

    def write(self, dataset, directory: str, **_) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        marker = path / "marker.txt"
        marker.write_text("ok", encoding="utf-8")
        self.writes[str(path)] = self.writes.get(str(path), 0) + 1


class DummyReader:
    file_type = "dummy"

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    def read(self, *_args, **_kwargs) -> Dataset:
        return self._dataset.clone()


@pytest.fixture()
def sample_dataset() -> Dataset:
    kg_source = KnowledgeGraph()
    kg_source.add_relation_triples(("s1", "r", "o1"))
    kg_target = KnowledgeGraph()
    kg_target.add_relation_triples(("s2", "r", "o2"))
    aligned = [("s1", "s2")]
    return Dataset(kg_source, kg_target, aligned)


@pytest.fixture()
def writer_plan(tmp_path) -> WriterPlan:
    writer = DummyWriter()
    return WriterPlan(name="dummy", writer=writer, write_reduced=True, write_augmented=True, write_results=True)


@pytest.fixture()
def stage_cfg(tmp_path) -> Dict[str, Dict]:
    load_builtin_models()
    load_builtin_augmentations()
    load_builtin_reducers()
    return {
        "reduction": {"target_entities": 1},
        "experiment": {
            "name": "test_exp",
            "dataset": "sample",
            "ratio": 0.5,
            "ratio_tag": "50",
            "reduction_method": "stub",
        },
        "lineage": {
            "reduction_root": str(tmp_path / "reduction"),
            "augmentation_root": str(tmp_path / "augmentation"),
            "evaluation_root": str(tmp_path / "evaluation"),
        },
    }


def test_reduction_stage_writes_outputs(tmp_path, sample_dataset, writer_plan, stage_cfg):
    reader = DummyReader(sample_dataset)
    ratio_meta: Dict[str, Dict] = {}
    stage = ReductionStage("stub", [writer_plan], resume=False)
    dataset_reduced = stage.execute(
        stage_cfg,
        sample_dataset,
        reader,
        ratio=0.5,
        ratio_tag="50",
        lineage=stage_cfg["lineage"],
        ratio_root=tmp_path,
        ratio_meta=ratio_meta,
        subtype=None,
    )

    assert isinstance(dataset_reduced, Dataset)
    summary_path = tmp_path / "reduction" / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["method"] == "stub"


def test_augmentation_stage_handles_stub(tmp_path, sample_dataset, writer_plan, stage_cfg):
    reader = DummyReader(sample_dataset)
    ratio_meta: Dict[str, Dict] = {}
    stage = AugmentationStage([writer_plan], resume=False)
    dataset_augmented = stage.execute(
        stage_cfg,
        "stub",
        sample_dataset,
        reader,
        stage_cfg["lineage"],
        ratio=0.5,
        ratio_tag="50",
        ratio_root=tmp_path,
        ratio_meta=ratio_meta,
        subtype=None,
    )

    assert isinstance(dataset_augmented, Dataset)
    summary_path = tmp_path / "augmentation" / "stub" / "summary.json"
    assert summary_path.exists()


def test_evaluation_stage_emits_results(tmp_path, sample_dataset, writer_plan, stage_cfg):
    ratio_meta: Dict[str, Dict] = {}
    stage = EvaluationStage([writer_plan], models=["stub"], resume=False)
    stage.execute(
        "baseline",
        sample_dataset,
        sample_dataset,
        stage_cfg,
        stage_cfg["lineage"],
        ratio_root=tmp_path,
        ratio_tag="50",
        ratio_meta=ratio_meta,
    )

    evaluation_dir = tmp_path / "evaluation" / "baseline"
    result_file = evaluation_dir / "stub.json"
    assert result_file.exists()
    data = json.loads(result_file.read_text(encoding="utf-8"))
    assert "hits@1" in data
