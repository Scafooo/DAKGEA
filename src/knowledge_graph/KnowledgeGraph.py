"""Thin wrapper around rdflib's Graph with convenience helpers for triples."""

from typing import Tuple

from rdflib import Graph, Literal, URIRef


class KnowledgeGraph(Graph):
    """Knowledge graph with helpers for registering relation and attribute triples."""

    def __init__(self):
        super().__init__()
        self.attr_to_name = dict()

    def add_attribute_triples(self, triple: Tuple[str, str, str]) -> None:
        """Insert attribute triples (entity, attribute, literal value)."""
        self.add((URIRef(triple[0]), URIRef(triple[1]), Literal(triple[2])))

    def add_relation_triples(self, triple: Tuple[str, str, str]) -> None:
        """Insert relation triples (subject, predicate, object)."""
        self.add((URIRef(triple[0]), URIRef(triple[1]), URIRef(triple[2])))
