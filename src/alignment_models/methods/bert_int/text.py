"""Helpers for generating textual descriptions of entities."""

from __future__ import annotations

from typing import Dict

from rdflib import URIRef


def friendly_name(uri: str, dataset_name: str | None = None) -> str:
    token = str(uri)
    if "/" in token:
        token = token.split("/")[-1]
    token = token.replace("_", " ")
    return token


def build_graph_entity_texts(graph, dataset_name: str, *, kg_index: int) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    for subj in set(graph.subjects()):
        if isinstance(subj, URIRef):
            texts[str(subj)] = friendly_name(str(subj), dataset_name)
    return texts
