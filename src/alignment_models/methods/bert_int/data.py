"""Dataset preparation helpers for the BERT-INT pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from rdflib import Literal, URIRef

from src.dataset.Dataset import Dataset


@dataclass
class KnowledgeGraphSnapshot:
    entities: List[str]
    relations: List[str]
    relation_triples: List[Tuple[int, str, int]]
    attribute_triples: List[Tuple[int, str, str, str]]
    local_relation_index: Dict[str, int]


@dataclass
class BertIntDataset:
    """All structures required by the original BERT-INT pipeline."""

    kg1: KnowledgeGraphSnapshot
    kg2: KnowledgeGraphSnapshot
    index2entity: Dict[int, str]
    entity2index: Dict[str, int]
    index2rel: Dict[int, str]
    rel2index: Dict[str, int]
    ent_ids_1: List[int]
    ent_ids_2: List[int]
    train_pairs: List[Tuple[int, int]]
    test_pairs: List[Tuple[int, int]]
    source_global_to_local: Dict[int, int]
    target_global_to_local: Dict[int, int]


def _split_alignment(
    aligned_entities: Sequence[Tuple[URIRef, URIRef]],
    train_ratio: float,
) -> Tuple[List[Tuple[URIRef, URIRef]], List[Tuple[URIRef, URIRef]]]:
    train_ratio = max(0.05, min(train_ratio, 0.95))
    total = len(aligned_entities)
    cutoff = max(1, int(total * train_ratio))
    train = list(aligned_entities[:cutoff])
    test = list(aligned_entities[cutoff:]) or list(aligned_entities[-max(1, total // 5) :])
    return train, test


def _build_entity_index(entities: Iterable[str], offset: int = 0) -> Tuple[List[str], Dict[str, int]]:
    ordered = sorted(set(entities))
    mapping = {entity: idx + offset for idx, entity in enumerate(ordered)}
    return ordered, mapping


def _collect_entities(graph, include_literals: bool = False) -> Set[str]:
    entities: Set[str] = set()
    for subj, pred, obj in graph.triples((None, None, None)):
        if isinstance(subj, URIRef):
            entities.add(str(subj))
        if isinstance(obj, URIRef):
            entities.add(str(obj))
    return entities


def _collect_relations(graph) -> Set[str]:
    relations: Set[str] = set()
    for _, pred, obj in graph.triples((None, None, None)):
        if isinstance(obj, URIRef):
            relations.add(str(pred))
    return relations


def _extract_relation_triples(
    graph,
    entity_index: Dict[str, int],
) -> List[Tuple[int, str, int]]:
    triples: List[Tuple[int, str, int]] = []
    for subj, pred, obj in graph.triples((None, None, None)):
        if not isinstance(obj, URIRef):
            continue
        s = entity_index[str(subj)]
        o = entity_index[str(obj)]
        triples.append((s, str(pred), o))
    return triples


def _attribute_type(value: str) -> str:
    try:
        int(value)
        return "integer"
    except ValueError:
        try:
            float(value)
            return "float"
        except ValueError:
            return "string"


def _extract_attribute_triples(
    graph,
    entity_index: Dict[str, int],
) -> List[Tuple[int, str, str, str]]:
    triples: List[Tuple[int, str, str, str]] = []
    for subj, pred, obj in graph.triples((None, None, None)):
        if not isinstance(obj, Literal):
            continue
        s = entity_index[str(subj)]
        attr = str(pred)
        value = str(obj)
        value_type = _attribute_type(value)
        triples.append((s, attr, value, value_type))
    return triples


def _build_snapshot(graph, base_offset: int = 0) -> KnowledgeGraphSnapshot:
    entities_set = _collect_entities(graph)
    entities, entity_index_part = _build_entity_index(entities_set, offset=base_offset)

    relations_set = _collect_relations(graph)
    relations = sorted(relations_set)
    relation_index = {rel: idx for idx, rel in enumerate(relations)}

    relation_triples = _extract_relation_triples(graph, entity_index_part)
    attribute_triples = _extract_attribute_triples(graph, entity_index_part)

    return KnowledgeGraphSnapshot(
        entities=entities,
        relations=relations,
        relation_triples=relation_triples,
        attribute_triples=attribute_triples,
        local_relation_index=relation_index,
    )


def build_dataset(
    dataset: Dataset,
    train_ratio: float,
) -> BertIntDataset:
    """Convert a Dataset object into the structures expected by BERT-INT."""

    kg1_snapshot = _build_snapshot(dataset.knowledge_graph_source, base_offset=0)
    kg2_snapshot = _build_snapshot(
        dataset.knowledge_graph_target,
        base_offset=len(kg1_snapshot.entities),
    )

    index2entity: Dict[int, str] = {}
    entity2index: Dict[str, int] = {}
    for idx, entity in enumerate(kg1_snapshot.entities + kg2_snapshot.entities):
        index2entity[idx] = entity
        entity2index[entity] = idx

    index2rel: Dict[int, str] = {}
    rel2index: Dict[str, int] = {}
    rel_offset = 0
    for relation in kg1_snapshot.relations + kg2_snapshot.relations:
        if relation in rel2index:
            continue
        rel2index[relation] = rel_offset
        index2rel[rel_offset] = relation
        rel_offset += 1

    aligned_pairs = sorted(
        (str(src), str(tgt)) for src, tgt in dataset.aligned_entities
    )
    def _remap_rel_triples(snapshot: KnowledgeGraphSnapshot) -> List[Tuple[int, int, int]]:
        remapped: List[Tuple[int, int, int]] = []
        for head, rel_str, tail in snapshot.relation_triples:
            rel_id = rel2index[rel_str]
            remapped.append((head, rel_id, tail))
        return remapped

    kg1_snapshot.relation_triples = _remap_rel_triples(kg1_snapshot)
    kg2_snapshot.relation_triples = _remap_rel_triples(kg2_snapshot)

    train_pairs_raw, test_pairs_raw = _split_alignment(aligned_pairs, train_ratio)

    ent_ids_1 = [entity2index[e] for e in kg1_snapshot.entities]
    ent_ids_2 = [entity2index[e] for e in kg2_snapshot.entities]

    train_pairs = [(entity2index[src], entity2index[tgt]) for src, tgt in train_pairs_raw]
    test_pairs = [(entity2index[src], entity2index[tgt]) for src, tgt in test_pairs_raw]

    source_global_to_local = {entity2index[entity]: idx for idx, entity in enumerate(kg1_snapshot.entities)}
    target_global_to_local = {entity2index[entity]: idx for idx, entity in enumerate(kg2_snapshot.entities)}

    return BertIntDataset(
        kg1=kg1_snapshot,
        kg2=kg2_snapshot,
        index2entity=index2entity,
        entity2index=entity2index,
        index2rel=index2rel,
        rel2index=rel2index,
        ent_ids_1=ent_ids_1,
        ent_ids_2=ent_ids_2,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        source_global_to_local=source_global_to_local,
        target_global_to_local=target_global_to_local,
    )
