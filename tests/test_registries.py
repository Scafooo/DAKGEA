"""Smoke tests covering registry registration behaviour."""

from src.alignment_models.registry import (
    MODEL_REGISTRY,
    get_alignment_model,
    load_builtin_models,
)
from src.augmentation.registry import AUGMENTATION_REGISTRY, load_builtin_augmentations
from src.reduction.registry import REDUCTION_REGISTRY, load_builtin_reducers


def setup_module(_module):
    load_builtin_models()
    load_builtin_augmentations()
    load_builtin_reducers()


def test_alignment_models_stub_registered():
    assert get_alignment_model("stub")


def test_alignment_models_bert_int_registered():
    assert get_alignment_model("bert_int")


def test_augmentation_stub_registered():
    assert AUGMENTATION_REGISTRY.get("stub"), "Stub augmentation should be registered"


def test_reduction_random_entities_registered():
    assert REDUCTION_REGISTRY.get("random_entities"), "Random entities reducer should be registered"
