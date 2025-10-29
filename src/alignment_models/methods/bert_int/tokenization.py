"""Text extraction and tokenisation utilities for BERT-INT."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from rdflib import Graph, Literal, URIRef


def normalise_uri(uri: str) -> str:
    """Convert a URI into a readable label by stripping namespaces."""

    candidate = uri.split("/")[-1]
    candidate = candidate.split("#")[-1]
    candidate = candidate.replace("_", " ")
    if not candidate:
        return uri
    return candidate


def _select_best_literal(values: Sequence[str]) -> str:
    """Pick the most informative literal among the available ones."""

    if not values:
        return ""
    # Prefer shorter literals (names) over long descriptions
    sorted_vals = sorted(values, key=lambda v: (len(v), v))
    return sorted_vals[0]


def _collect_literals(graph: Graph) -> Dict[str, List[str]]:
    literals: Dict[str, List[str]] = {}
    for subj, _, obj in graph.triples((None, None, None)):
        if isinstance(subj, URIRef) and isinstance(obj, Literal):
            literals.setdefault(str(subj), []).append(str(obj))
    return literals


def extract_entity_texts(
    graph: Graph,
    focus_entities: Iterable[URIRef],
) -> Dict[str, str]:
    """Build human-readable text for the requested entities."""

    literals = _collect_literals(graph)
    texts: Dict[str, str] = {}
    for entity in focus_entities:
        uri = str(entity)
        options = literals.get(uri, [])
        label = _select_best_literal(options)
        if not label:
            label = normalise_uri(uri)
        texts[uri] = label
    return texts


def encode_entities(
    tokenizer_name: str,
    entity_texts: Dict[str, str],
    entity_order: Sequence[str],
    *,
    max_length: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, object]:
    """Tokenise entity texts and return tensors ready for BERT."""

    from transformers import AutoTokenizer  # local import to keep module light

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    sentences = [entity_texts[eid] for eid in entity_order]
    encoded = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"], tokenizer
