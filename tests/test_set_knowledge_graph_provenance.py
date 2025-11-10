"""Tests for SetKnowledgeGraph relation provenance tracking."""

from __future__ import annotations

from rdflib import Literal, URIRef

from src.augmentation.methods.plm.set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph
from src.core.dataset import Dataset
from src.core.knowledge_graph.knowledge_graph import KnowledgeGraph


def _build_dataset() -> Dataset:
    kg_source = KnowledgeGraph()
    s1 = URIRef("http://src/entity/A")
    s2 = URIRef("http://src/entity/B")
    kg_source.add_relation_triples((str(s1), "http://src/predicate/rel", str(s2)))
    kg_source.add_attribute_triples((str(s1), "http://src/predicate/name", "Alpha"))

    kg_target = KnowledgeGraph()
    t1 = URIRef("http://tgt/entity/A")
    t2 = URIRef("http://tgt/entity/C")
    t3 = URIRef("http://tgt/entity/D")
    kg_target.add_relation_triples((str(t1), "http://tgt/predicate/rel", str(t2)))
    kg_target.add_relation_triples((str(t3), "http://tgt/predicate/backlink", str(t1)))
    kg_target.add_attribute_triples((str(t1), "http://tgt/predicate/name", "Alfa"))

    aligned = [(str(s1), str(t1))]
    return Dataset(kg_source, kg_target, aligned)


def test_relation_provenance_records_source_and_target():
    dataset = _build_dataset()
    skg = SetKnowledgeGraph.from_dataset(dataset)

    fused_node = next(skg.iter_set_nodes())
    origins = skg.get_relation_origins(fused_node)

    source_rel = (
        URIRef("http://src/predicate/rel"),
        URIRef("http://src/entity/B"),
        "out",
    )
    target_rel = (
        URIRef("http://tgt/predicate/rel"),
        URIRef("http://tgt/entity/C"),
        "out",
    )
    incoming_rel = (
        URIRef("http://tgt/predicate/backlink"),
        URIRef("http://tgt/entity/D"),
        "in",
    )

    assert source_rel in origins["source"]
    assert target_rel in origins["target"]
    assert incoming_rel in origins["target"]


def test_non_set_node_has_empty_origin():
    dataset = _build_dataset()
    skg = SetKnowledgeGraph.from_dataset(dataset)
    some_non_set = URIRef("http://src/entity/B")
    origins = skg.get_relation_origins(some_non_set)
    assert origins == {"source": set(), "target": set()}
