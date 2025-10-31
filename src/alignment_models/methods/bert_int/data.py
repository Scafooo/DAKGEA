"""Dataset preparation helpers for the BERT-INT pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from rdflib import Literal, URIRef

from src.core.dataset import Dataset
from src.logger import get_logger
from src.util.reader import read_tsv

logger = get_logger(__name__)


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


def _resolve_hybea_data_dir(base: Path) -> Optional[Path]:
    candidates = [
        base,
        base / "attribute_data",
        base / "ATTRIBUTE_DATA",
    ]
    for candidate in candidates:
        if (candidate / "ent_ids_1").exists() and (candidate / "ent_ids_2").exists():
            return candidate
    return None


def _read_id_map_int(path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if not path.exists():
        return mapping
    for row in read_tsv(path):
        if len(row) < 2:
            continue
        try:
            idx = int(str(row[0]).strip())
        except ValueError:
            continue
        uri = str(row[1]).strip()
        mapping[idx] = uri
    return mapping


def _read_pairs(path: Path) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    if not path.exists():
        return pairs
    for row in read_tsv(path):
        if len(row) < 2:
            continue
        try:
            left = int(str(row[0]).strip())
            right = int(str(row[1]).strip())
        except ValueError:
            continue
        pairs.append((left, right))
    return pairs


def _read_numeric_triples(path: Path) -> List[Tuple[int, int, int]]:
    triples: List[Tuple[int, int, int]] = []
    if not path.exists():
        return triples
    for row in read_tsv(path):
        if len(row) < 3:
            continue
        try:
            head = int(str(row[0]).strip())
            rel = int(str(row[1]).strip())
            tail = int(str(row[2]).strip())
        except ValueError:
            continue
        triples.append((head, rel, tail))
    return triples


def _parse_literal_text(raw: str) -> Tuple[str, str]:
    text = raw.strip()
    if not text:
        return "", "string"
    try:
        literal = Literal.from_n3(text)
    except Exception:
        cleaned = text.strip('"').replace("\t", " ")
        cleaned = " ".join(cleaned.split())
        return cleaned, "string"
    if literal.language:
        value_type = literal.language
    elif literal.datatype:
        value_type = str(literal.datatype)
    else:
        value_type = "string"
    cleaned_text = str(literal).replace("\t", " ")
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text, value_type


def _read_attribute_triples(path: Path, entity_lookup: Dict[str, int]) -> List[Tuple[int, str, str, str]]:
    triples: List[Tuple[int, str, str, str]] = []
    if not path.exists():
        return triples
    for row in read_tsv(path):
        if len(row) < 3:
            continue
        subj_raw = str(row[0]).strip()
        attr = str(row[1]).strip()
        value_raw = "\t".join(str(part) for part in row[2:]).strip()
        entity_id = entity_lookup.get(subj_raw)
        if entity_id is None:
            continue
        value_text, value_type = _parse_literal_text(value_raw)
        triples.append((entity_id, attr, value_text, value_type))
    return triples


def _build_dataset_from_hybea(
    lineage: Optional[Dict[str, object]],
    dataset_name: str,
) -> Optional[BertIntDataset]:
    if not lineage:
        return None
    path_str = lineage.get("hybea_dataset_path")
    if not path_str:
        return None
    base = Path(path_str)
    if not base.exists():
        logger.warning("[BERT-INT] HybEA dataset path does not exist: %s", base)
        return None

    data_dir = _resolve_hybea_data_dir(base)
    if data_dir is None:
        logger.warning("[BERT-INT] HybEA dataset missing attribute data directory at %s (dataset=%s)", base, dataset_name)
        return None

    ent_map1 = _read_id_map_int(data_dir / "ent_ids_1")
    ent_map2 = _read_id_map_int(data_dir / "ent_ids_2")
    if not ent_map1 or not ent_map2:
        logger.warning("[BERT-INT] HybEA entity files missing or empty at %s (dataset=%s)", data_dir, dataset_name)
        return None

    index2entity: Dict[int, str] = {**ent_map1, **ent_map2}
    entity2index: Dict[str, int] = {uri: idx for idx, uri in index2entity.items()}

    rel_map1 = _read_id_map_int(data_dir / "rel_ids_1")
    rel_map2 = _read_id_map_int(data_dir / "rel_ids_2")
    index2rel: Dict[int, str] = {**rel_map1, **rel_map2}
    rel2index: Dict[str, int] = {uri: idx for idx, uri in index2rel.items()}

    triples1_raw = _read_numeric_triples(data_dir / "triples_1")
    triples2_raw = _read_numeric_triples(data_dir / "triples_2")
    kg1_rel_triples: List[Tuple[int, str, int]] = []
    for head, rel_idx, tail in triples1_raw:
        rel_uri = index2rel.get(rel_idx)
        if rel_uri is None:
            continue
        kg1_rel_triples.append((head, rel_uri, tail))
    kg2_rel_triples: List[Tuple[int, str, int]] = []
    for head, rel_idx, tail in triples2_raw:
        rel_uri = index2rel.get(rel_idx)
        if rel_uri is None:
            continue
        kg2_rel_triples.append((head, rel_uri, tail))

    attr_triples1 = _read_attribute_triples(data_dir / "attr_triples1", entity2index)
    attr_triples2 = _read_attribute_triples(data_dir / "attr_triples2", entity2index)

    train_pairs = _read_pairs(data_dir / "sup_pairs")
    test_pairs = _read_pairs(data_dir / "ref_pairs")
    if not train_pairs or not test_pairs:
        logger.warning(
            "[BERT-INT] HybEA split files missing/empty at %s (train=%d, test=%d); falling back to graph-derived splits.",
            data_dir,
            len(train_pairs),
            len(test_pairs),
        )
        return None

    kg1_entities = [ent_map1[idx] for idx in sorted(ent_map1.keys())]
    kg2_entities = [ent_map2[idx] for idx in sorted(ent_map2.keys())]
    ent_ids_1 = [entity2index[uri] for uri in kg1_entities]
    ent_ids_2 = [entity2index[uri] for uri in kg2_entities]

    kg1_relations = [rel_map1[idx] for idx in sorted(rel_map1.keys())]
    kg2_relations = [rel_map2[idx] for idx in sorted(rel_map2.keys())]
    kg1_relation_index = {uri: pos for pos, uri in enumerate(kg1_relations)}
    kg2_relation_index = {uri: pos for pos, uri in enumerate(kg2_relations)}

    kg1_snapshot = KnowledgeGraphSnapshot(
        entities=kg1_entities,
        relations=kg1_relations,
        relation_triples=kg1_rel_triples,
        attribute_triples=attr_triples1,
        local_relation_index=kg1_relation_index,
    )
    kg2_snapshot = KnowledgeGraphSnapshot(
        entities=kg2_entities,
        relations=kg2_relations,
        relation_triples=kg2_rel_triples,
        attribute_triples=attr_triples2,
        local_relation_index=kg2_relation_index,
    )

    source_global_to_local = {ent_id: idx for idx, ent_id in enumerate(ent_ids_1)}
    target_global_to_local = {ent_id: idx for idx, ent_id in enumerate(ent_ids_2)}

    logger.info(
        "[BERT-INT] Loaded HybEA artefacts for dataset '%s': |KG1|=%d, |KG2|=%d, train=%d, test=%d",
        dataset_name,
        len(ent_ids_1),
        len(ent_ids_2),
        len(train_pairs),
        len(test_pairs),
    )

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


def _normalise_literal_value(lit: Literal) -> Tuple[str, str]:
    """Return cleaned literal text and a lightweight type label."""

    # Determine type / language label
    if lit.language:
        value_type = lit.language
    elif lit.datatype:
        value_type = str(lit.datatype)
    else:
        value_type = "string"

    # Obtain textual representation without rdflib decorations
    try:
        raw = lit.value  # may be Python primitive
        text = str(raw)
    except Exception:
        text = str(lit)

    text = text.replace("\t", " ")
    text = text.strip().strip('"')

    # Normalise whitespace
    text = " ".join(text.split())

    return text, value_type


def _extract_attribute_triples(
    graph,
    entity_index: Dict[str, int],
) -> List[Tuple[int, str, str, str]]:
    triples: List[Tuple[int, str, str, str]] = []
    for subj, pred, obj in graph.triples((None, None, None)):
        if not isinstance(obj, Literal):
            continue
        subject_uri = str(subj)
        if subject_uri not in entity_index:
            continue
        s = entity_index[subject_uri]
        attr = str(pred)
        value, value_type = _normalise_literal_value(obj)
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
    *,
    lineage: Optional[Dict[str, object]] = None,
    dataset_name: str = "",
) -> BertIntDataset:
    """Convert a Dataset object into the structures expected by BERT-INT."""

    hybea_dataset = _build_dataset_from_hybea(lineage, dataset_name)
    if hybea_dataset is not None:

        def _remap_from_strings(snapshot: KnowledgeGraphSnapshot) -> List[Tuple[int, int, int]]:
            remapped: List[Tuple[int, int, int]] = []
            for head, rel_str, tail in snapshot.relation_triples:
                rel_id = hybea_dataset.rel2index.get(rel_str)
                if rel_id is None:
                    continue
                remapped.append((head, rel_id, tail))
            return remapped

        hybea_dataset.kg1.relation_triples = _remap_from_strings(hybea_dataset.kg1)
        hybea_dataset.kg2.relation_triples = _remap_from_strings(hybea_dataset.kg2)
        return hybea_dataset

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

    hybea_pairs = _load_pairs_from_hybea(lineage, dataset_name, entity2index)

    def _remap_rel_triples(snapshot: KnowledgeGraphSnapshot) -> List[Tuple[int, int, int]]:
        remapped: List[Tuple[int, int, int]] = []
        for head, rel_str, tail in snapshot.relation_triples:
            rel_id = rel2index[rel_str]
            remapped.append((head, rel_id, tail))
        return remapped

    kg1_snapshot.relation_triples = _remap_rel_triples(kg1_snapshot)
    kg2_snapshot.relation_triples = _remap_rel_triples(kg2_snapshot)

    ent_ids_1 = [entity2index[e] for e in kg1_snapshot.entities]
    ent_ids_2 = [entity2index[e] for e in kg2_snapshot.entities]

    if hybea_pairs is not None:
        train_pairs, test_pairs = hybea_pairs
        logger.debug(
            "[BERT-INT] Loaded train/test pairs from HybEA artefacts (%d train, %d test)",
            len(train_pairs),
            len(test_pairs),
        )
    else:
        train_pairs_raw, test_pairs_raw = _split_alignment(aligned_pairs, train_ratio)
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


def _load_pairs_from_hybea(
    lineage: Optional[Dict[str, object]],
    dataset_name: str,
    entity2index: Dict[str, int],
) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    if not lineage:
        return None
    path_str = lineage.get("hybea_dataset_path")
    if not path_str:
        return None

    base = Path(path_str)
    if not base.exists():
        logger.debug("[BERT-INT] HybEA path %s does not exist", base)
        return None

    data_dir = _resolve_hybea_data_dir(base)
    if data_dir is None:
        logger.debug("[BERT-INT] HybEA attribute data missing at %s", base)
        return None

    attr_sup = data_dir / "sup_pairs"
    attr_ref = data_dir / "ref_pairs"
    knowformer_sup = data_dir / "sup_ents.txt"
    knowformer_ref = data_dir / "ref_ents.txt"

    if attr_sup.exists() and attr_ref.exists():
        ent_map1 = _load_ent_ids(data_dir / "ent_ids_1")
        ent_map2 = _load_ent_ids(data_dir / "ent_ids_2")
        train_raw = _load_pair_file(attr_sup, ent_map1, ent_map2)
        test_raw = _load_pair_file(attr_ref, ent_map1, ent_map2)
    elif knowformer_sup.exists() and knowformer_ref.exists():
        ent_map1 = _load_ent_ids(data_dir / "ent_ids_1")
        ent_map2 = _load_ent_ids(data_dir / "ent_ids_2")
        train_raw = _load_pair_file(knowformer_sup, ent_map1, ent_map2)
        test_raw = _load_pair_file(knowformer_ref, ent_map1, ent_map2)
    else:
        logger.debug(
            "[BERT-INT] No HybEA split files found at %s (dataset=%s)",
            data_dir,
            dataset_name,
        )
        return None

    def to_indices(pairs: List[Tuple[str, str]], label: str) -> List[Tuple[int, int]]:
        results: List[Tuple[int, int]] = []
        skipped = 0
        for left_uri, right_uri in pairs:
            if left_uri not in entity2index or right_uri not in entity2index:
                skipped += 1
                continue
            results.append((entity2index[left_uri], entity2index[right_uri]))
        if skipped:
            logger.debug(
                "[BERT-INT] Skipped %d %s pairs missing from entity index (dataset=%s)",
                skipped,
                label,
                dataset_name,
            )
        return results

    train_idx = to_indices(train_raw, "train")
    test_idx = to_indices(test_raw, "test")

    if not train_idx or not test_idx:
        logger.warning(
            "[BERT-INT] HybEA-derived splits were empty (train=%d, test=%d); falling back to ratio split",
            len(train_idx),
            len(test_idx),
        )
        return None

    return train_idx, test_idx


def _load_ent_ids(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping
    for row in read_tsv(path):
        if len(row) < 2:
            continue
        key = str(row[0]).strip()
        uri = str(row[1]).strip()
        mapping[key] = uri
        try:
            mapping[str(int(key))] = uri
        except (ValueError, TypeError):
            pass
    return mapping


def _load_pair_file(
    path: Path,
    map_left: Dict[str, str],
    map_right: Dict[str, str],
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not path.exists():
        return pairs
    for row in read_tsv(path):
        if len(row) < 2:
            continue
        left_raw = str(row[0]).strip()
        right_raw = str(row[1]).strip()
        left = map_left.get(left_raw, left_raw)
        right = map_right.get(right_raw, right_raw)
        pairs.append((left, right))
    return pairs
