"""Tests for the PLM-based augmentation scaffolding."""

from __future__ import annotations

from rdflib import URIRef

from src.augmentation.methods.plm.plm_augmenter import PLMAugmenter
from src.core.dataset import Dataset
from src.core.knowledge_graph.knowledge_graph import KnowledgeGraph


def _build_sample_dataset() -> Dataset:
    kg_source = KnowledgeGraph()
    kg_source.add_relation_triples(
        ("http://src.example.org/entity/A", "http://relation.example.org/linkedTo", "http://src.example.org/entity/B")
    )
    kg_source.add_attribute_triples(
        ("http://src.example.org/entity/A", "http://attribute.example.org/name", "Alpha")
    )

    kg_target = KnowledgeGraph()
    kg_target.add_relation_triples(
        ("http://tgt.example.org/entity/A", "http://relation.example.org/linkedTo", "http://tgt.example.org/entity/C")
    )
    kg_target.add_attribute_triples(
        ("http://tgt.example.org/entity/A", "http://attribute.example.org/name", "Alfa")
    )

    aligned = [
        ("http://src.example.org/entity/A", "http://tgt.example.org/entity/A"),
    ]
    return Dataset(kg_source, kg_target, aligned)


def _build_dataset_with_two_pairs() -> Dataset:
    kg_source = KnowledgeGraph()
    kg_source.add_relation_triples(
        ("http://src.example.org/entity/A", "http://relation.example.org/linkedTo", "http://src.example.org/entity/B")
    )
    kg_source.add_relation_triples(
        ("http://src.example.org/entity/D", "http://relation.example.org/linkedTo", "http://src.example.org/entity/E")
    )
    kg_source.add_attribute_triples(
        ("http://src.example.org/entity/A", "http://attribute.example.org/name", "Alpha")
    )
    kg_source.add_attribute_triples(
        ("http://src.example.org/entity/D", "http://attribute.example.org/name", "Delta")
    )

    kg_target = KnowledgeGraph()
    kg_target.add_relation_triples(
        ("http://tgt.example.org/entity/A", "http://relation.example.org/linkedTo", "http://tgt.example.org/entity/C")
    )
    kg_target.add_relation_triples(
        ("http://tgt.example.org/entity/D", "http://relation.example.org/linkedTo", "http://tgt.example.org/entity/F")
    )
    kg_target.add_attribute_triples(
        ("http://tgt.example.org/entity/A", "http://attribute.example.org/name", "Alfa")
    )
    kg_target.add_attribute_triples(
        ("http://tgt.example.org/entity/D", "http://attribute.example.org/name", "Delta")
    )

    aligned = [
        ("http://src.example.org/entity/A", "http://tgt.example.org/entity/A"),
        ("http://src.example.org/entity/D", "http://tgt.example.org/entity/D"),
    ]
    return Dataset(kg_source, kg_target, aligned)


def test_plm_augmenter_creates_augmented_alignment():
    dataset = _build_sample_dataset()
    augmenter = PLMAugmenter({"augmentation": {"max_depth": 1, "ratio": 1.0}, "experiment": {"seed": 7}})

    augmented = augmenter.augment(dataset)

    assert augmented is not dataset
    assert len(dataset.aligned_entities) == 1
    assert len(augmented.aligned_entities) == 2

    synthetic_pairs = [
        pair for pair in augmented.aligned_entities if "_aug" in pair[0] and pair[0].startswith("http://src.example.org/entity/A")
    ]
    assert synthetic_pairs, "Expected at least one synthetic aligned pair"

    new_src_uri = URIRef(synthetic_pairs[0][0])
    derived_predicate = URIRef("http://dakgea.org/augmentation/derivedFrom")
    linked_predicate = URIRef("http://relation.example.org/linkedTo")
    literal_predicate = URIRef("http://attribute.example.org/name")

    assert any(
        predicate == derived_predicate for _, predicate, _ in augmented.knowledge_graph_source.triples((new_src_uri, None, None))
    ), "Augmented node should reference its origin"

    clone_targets = [
        obj for _, _, obj in augmented.knowledge_graph_source.triples((new_src_uri, linked_predicate, None))
    ]
    assert clone_targets, "Expected augmented source node to connect to synthetic neighbor"
    clone_uri = clone_targets[0]
    assert clone_uri != URIRef("http://src.example.org/entity/B")
    assert (
        clone_uri,
        derived_predicate,
        URIRef("http://src.example.org/entity/B"),
    ) in augmented.knowledge_graph_source, "Synthetic neighbor should retain provenance"

    target_clone_targets = [
        obj for _, _, obj in augmented.knowledge_graph_target.triples((URIRef(synthetic_pairs[0][1]), linked_predicate, None))
    ]
    assert target_clone_targets, "Expected augmented target node to connect to synthetic neighbor"
    target_clone_uri = target_clone_targets[0]
    assert target_clone_uri != URIRef("http://tgt.example.org/entity/C")
    assert (
        target_clone_uri,
        derived_predicate,
        URIRef("http://tgt.example.org/entity/C"),
    ) in augmented.knowledge_graph_target

    assert any(
        literal.toPython() == "Alpha"
        for _, _, literal in augmented.knowledge_graph_source.triples((new_src_uri, literal_predicate, None))
    ), "Literal attributes should be copied to augmented node"

    assert all(
        "_aug" not in str(subject)
        for subject, _, _ in dataset.knowledge_graph_source.triples((None, None, None))
    ), "Original dataset should remain untouched"


def test_ratio_controls_number_of_augmented_pairs():
    dataset = _build_dataset_with_two_pairs()
    augmenter = PLMAugmenter({"augmentation": {"ratio": 0.5, "max_depth": 0}, "experiment": {"seed": 11}})

    augmented = augmenter.augment(dataset)

    assert len(dataset.aligned_entities) == 2
    assert len(augmented.aligned_entities) == 3, "Ratio=0.5 over 2 pairs should add exactly one alignment"


def test_plm_augmenter_is_deterministic_with_same_seed():
    cfg = {"augmentation": {"ratio": 1.0, "max_depth": 1}, "experiment": {"seed": 1234}}
    augmenter = PLMAugmenter(cfg)

    dataset_a = _build_dataset_with_two_pairs()
    dataset_b = _build_dataset_with_two_pairs()

    augmented_a = augmenter.augment(dataset_a)
    augmented_b = augmenter.augment(dataset_b)

    assert augmented_a.aligned_entities == augmented_b.aligned_entities
    assert set(augmented_a.knowledge_graph_source) == set(augmented_b.knowledge_graph_source)
    assert set(augmented_a.knowledge_graph_target) == set(augmented_b.knowledge_graph_target)
