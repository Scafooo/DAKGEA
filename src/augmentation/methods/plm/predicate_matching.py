"""Utilities to align predicates across aligned knowledge graphs."""

import re
from itertools import product
from typing import Iterable, Tuple

import numpy as np
import torch
from rdflib import Literal, URIRef
from sentence_transformers import SentenceTransformer, util

from src.logger import get_logger

logger = get_logger(__name__)


def get_predicate_label(graph, predicate) -> str:
    """Return a human-readable label for a predicate."""
    if hasattr(graph, "attr_to_name") and predicate in graph.attr_to_name:
        label = graph.attr_to_name[predicate].lower()
    else:
        label = re.split(r"[/#]", str(predicate))[-1].lower()

    return label.replace("_", " ").strip()


def string_similarity(left: str, right: str) -> float:
    tokens_left = set(left.split())
    tokens_right = set(right.split())
    union = tokens_left | tokens_right
    if not union:
        return 0.0
    return len(tokens_left & tokens_right) / len(union)


def _compute_predicate_embeddings(graph_left, graph_right, model_name: str) -> tuple[dict, SentenceTransformer]:
    predicates = set(graph_left.predicates()) | set(graph_right.predicates())
    names = [
        get_predicate_label(graph_left if predicate in graph_left.predicates() else graph_right, predicate)
        for predicate in predicates
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading SentenceTransformer '%s' on %s", model_name, device)

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(names, convert_to_tensor=True)
    return dict(zip(predicates, embeddings)), model


def _predicate_similarity(
    pred_left,
    pred_right,
    emb_map,
    graph_left,
    graph_right,
    alpha: float,
) -> float:
    name_left = get_predicate_label(graph_left, pred_left)
    name_right = get_predicate_label(graph_right, pred_right)
    semantic = util.cos_sim(emb_map[pred_left], emb_map[pred_right]).item()
    lexical = string_similarity(name_left, name_right)
    return alpha * semantic + (1 - alpha) * lexical


def _collect_triples(graph, entity):
    outgoing = [(predicate, obj, "out") for _, predicate, obj in graph.triples((entity, None, None))]
    incoming = [(predicate, subj, "in") for subj, predicate, _ in graph.triples((None, None, entity))]
    return outgoing + incoming


def match_relations(
    dataset,
    alpha: float = 0.6,
    bonus_entity_match: float = 0.2,
    min_score: float = 0.3,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
):
    """Return predicate correspondences scored by semantic and structural evidence."""

    graph_left = dataset.knowledge_graph_source
    graph_right = dataset.knowledge_graph_target

    emb_map, _ = _compute_predicate_embeddings(graph_left, graph_right, model_name)

    candidate_matches = []
    entity_pairs_set = set(dataset.aligned_entities)

    for entity_left, entity_right in entity_pairs_set:
        triples_left = _collect_triples(graph_left, entity_left)
        triples_right = _collect_triples(graph_right, entity_right)

        for (pred_left, obj_left, dir_left), (pred_right, obj_right, dir_right) in product(
            triples_left, triples_right
        ):
            if dir_left != dir_right:
                continue

            score = _predicate_similarity(pred_left, pred_right, emb_map, graph_left, graph_right, alpha)

            if (
                isinstance(obj_left, URIRef)
                and isinstance(obj_right, URIRef)
                and (obj_left, obj_right) in entity_pairs_set
            ):
                score += bonus_entity_match

            if score >= min_score:
                candidate_matches.append(((pred_left, pred_right), score, dir_left))

    scores = {}
    for (pred_left, pred_right), value, _ in candidate_matches:
        scores.setdefault((pred_left, pred_right), []).append(value)

    averaged = {key: float(np.mean(vals)) for key, vals in scores.items()}
    return sorted(averaged.items(), key=lambda item: item[1], reverse=True)


def reduce_to_best_matches(matches):
    best = {}
    for (pred_left, pred_right), score in matches:
        current = best.get(pred_left)
        if current is None or score > current[1]:
            best[pred_left] = (pred_right, score)
    return [((pred_left, pred_right), score) for pred_left, (pred_right, score) in best.items()]
