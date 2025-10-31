"""Thin wrapper around rdflib's Graph with convenience helpers for triples."""

from __future__ import annotations

from typing import Tuple

from rdflib import Graph, Literal, URIRef


class KnowledgeGraph(Graph):
    """Knowledge graph with helpers for registering relation and attribute triples."""

    def __init__(self) -> None:
        super().__init__()
        self.attr_to_name = dict()

    def add_attribute_triples(self, triple: Tuple[str, str, str]) -> None:
        """Insert attribute triples (entity, attribute, literal value)."""
        self.add((URIRef(triple[0]), URIRef(triple[1]), Literal(triple[2])))

    def add_relation_triples(self, triple: Tuple[str, str, str]) -> None:
        """Insert relation triples (subject, predicate, object)."""
        self.add((URIRef(triple[0]), URIRef(triple[1]), URIRef(triple[2])))

    def clone(self) -> "KnowledgeGraph":
        """Create a deep copy of the graph, including namespaces and attribute names."""
        duplicate = KnowledgeGraph()
        duplicate.attr_to_name = dict(self.attr_to_name)

        for prefix, namespace in self.namespace_manager.namespaces():
            duplicate.namespace_manager.bind(prefix, namespace, override=True)

        for triple in self.triples((None, None, None)):
            duplicate.add(triple)

        return duplicate
