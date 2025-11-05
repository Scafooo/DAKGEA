"""Dataset loading utilities for the BERT-INT basic unit."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from rdflib import Literal
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)

Pair = Tuple[int, int]
Triple = Tuple[int, int, int]


@dataclass
class BasicUnitDataBundle:
    """Container holding all artefacts required by the basic unit."""

    ent_ill: List[Pair]
    train_ill: List[Pair]
    test_ill: List[Pair]
    index2rel: Dict[int, str]
    index2entity: Dict[int, str]
    rel2index: Dict[str, int]
    entity2index: Dict[str, int]
    ent2data: Dict[int, Tuple[List[int], List[float]]]
    rel_triples_1: List[Triple]
    rel_triples_2: List[Triple]


def load_basic_unit_data_from_dataset(
    config: Mapping[str, Any],
    paths: Mapping[str, Any],
    dataset,
    dataset_workspace_path: Optional[Path] = None,
) -> BasicUnitDataBundle:
    """Load tensors and metadata from in-memory dataset object.

    Args:
        config: Configuration for basic unit
        paths: Resolved paths (currently unused)
        dataset: Dataset object with knowledge graphs
        dataset_workspace_path: Path to dataset files (sup_ents.txt, ref_ents.txt, etc.)
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.get("encoder_name", "bert-base-multilingual-cased"))

    # Convert dataset to BERT-INT format
    ent_ill, train_ill, test_ill, index2rel, index2entity, rel2index, entity2index, ent2data, rel_triples_1, rel_triples_2 = _convert_dataset_to_bert_format(
        dataset, tokenizer, config.get("max_seq_length", 128), dataset_workspace_path
    )

    return BasicUnitDataBundle(
        ent_ill=ent_ill,
        train_ill=train_ill,
        test_ill=test_ill,
        index2rel=index2rel,
        index2entity=index2entity,
        rel2index=rel2index,
        entity2index=entity2index,
        ent2data=ent2data,
        rel_triples_1=rel_triples_1,
        rel_triples_2=rel_triples_2,
    )


def _convert_dataset_to_bert_format(dataset, tokenizer, max_length, dataset_workspace_path=None):
    """Convert in-memory dataset to BERT-INT format.

    IMPORTANT: For proper evaluation, we need to maintain the original
    train/test split from the dataset files (sup_ents.txt/ref_ents.txt).
    We cannot do a random split as it would cause data leakage.

    Args:
        dataset: Dataset object with knowledge graphs
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        dataset_workspace_path: Path to dataset files (sup_ents.txt, ref_ents.txt, etc.)
    """
    # Build entity mappings first from all entities in the dataset
    all_entities = set()
    for src, tgt in dataset.aligned_entities:
        all_entities.add(str(src))
        all_entities.add(str(tgt))

    entity2index = {ent: idx for idx, ent in enumerate(sorted(all_entities))}
    index2entity = {idx: ent for ent, idx in entity2index.items()}

    # Read train/test split from files if available
    train_ill = []
    test_ill = []

    logger.info(f"[BERT-INT] dataset_workspace_path = {dataset_workspace_path}")
    if dataset_workspace_path:
        logger.info(f"[BERT-INT] Path exists: {Path(dataset_workspace_path).exists()}")

    if dataset_workspace_path and Path(dataset_workspace_path).exists():
        workspace = Path(dataset_workspace_path)
        sup_ents_file = workspace / "sup_ents.txt"
        ref_ents_file = workspace / "ref_ents.txt"

        if sup_ents_file.exists() and ref_ents_file.exists():
            logger.info(f"[BERT-INT] Reading train/test split from {workspace}")

            # Read sup_ents.txt (training pairs)
            with open(sup_ents_file, 'r', encoding='utf-8') as f:
                for line in f:
                    src, tgt = line.strip().split('\t')
                    if src in entity2index and tgt in entity2index:
                        train_ill.append((entity2index[src], entity2index[tgt]))

            # Read ref_ents.txt (test pairs)
            with open(ref_ents_file, 'r', encoding='utf-8') as f:
                for line in f:
                    src, tgt = line.strip().split('\t')
                    if src in entity2index and tgt in entity2index:
                        test_ill.append((entity2index[src], entity2index[tgt]))

            logger.info(f"[BERT-INT] Loaded {len(train_ill)} train pairs and {len(test_ill)} test pairs from files")
        else:
            logger.warning(f"[BERT-INT] Files {sup_ents_file} or {ref_ents_file} not found")

    # Fallback: use 70/30 split if files not available
    if not train_ill and not test_ill:
        logger.warning("[BERT-INT] No train/test files found, falling back to 70/30 split")
        aligned_indices = [
            (entity2index[str(src)], entity2index[str(tgt)])
            for src, tgt in dataset.aligned_entities
        ]
        split_idx = int(len(aligned_indices) * 0.7)
        train_ill = aligned_indices[:split_idx]
        test_ill = aligned_indices[split_idx:]

    ent_ill = train_ill + test_ill  # All pairs for candidate generation
    
    # Build relation mappings from triples
    all_relations = set()
    for subj, pred, obj in dataset.knowledge_graph_source:
        all_relations.add(str(pred))
    for subj, pred, obj in dataset.knowledge_graph_target:
        all_relations.add(str(pred))
    
    rel2index = {rel: idx for idx, rel in enumerate(sorted(all_relations))}
    index2rel = {idx: rel for rel, idx in rel2index.items()}
    
    # Convert triples (only relation triples, not attribute triples)
    rel_triples_1 = []
    for subj, pred, obj in dataset.knowledge_graph_source:
        subj_str = str(subj)
        pred_str = str(pred)
        obj_str = str(obj)
        # Skip if object is a literal (attribute triple)
        if subj_str in entity2index and obj_str in entity2index and pred_str in rel2index:
            rel_triples_1.append((entity2index[subj_str], rel2index[pred_str], entity2index[obj_str]))

    rel_triples_2 = []
    for subj, pred, obj in dataset.knowledge_graph_target:
        subj_str = str(subj)
        pred_str = str(pred)
        obj_str = str(obj)
        # Skip if object is a literal (attribute triple)
        if subj_str in entity2index and obj_str in entity2index and pred_str in rel2index:
            rel_triples_2.append((entity2index[subj_str], rel2index[pred_str], entity2index[obj_str]))
    
    # Create entity data (usando attributi se disponibili)
    # Raccogliamo attributi dai knowledge graphs (triples con literal values)
    entity_attributes = {}
    for subj, pred, obj in dataset.knowledge_graph_source:
        subj_str = str(subj)
        if subj_str in all_entities and isinstance(obj, Literal):
            if subj_str not in entity_attributes:
                entity_attributes[subj_str] = []
            entity_attributes[subj_str].append(str(obj))

    for subj, pred, obj in dataset.knowledge_graph_target:
        subj_str = str(subj)
        if subj_str in all_entities and isinstance(obj, Literal):
            if subj_str not in entity_attributes:
                entity_attributes[subj_str] = []
            entity_attributes[subj_str].append(str(obj))

    ent2data = {}
    for ent in all_entities:
        # Usa gli attributi se disponibili, altrimenti usa il nome dell'entità
        if ent in entity_attributes:
            desc = " ".join(entity_attributes[ent])
        else:
            # Extract last part of URI as name
            desc = ent.split('/')[-1].split('#')[-1].replace('_', ' ')

        tokens = tokenizer(desc, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
        # BasicUnit expects (input_ids, attention_mask) tuple
        ent2data[entity2index[ent]] = (
            tokens['input_ids'].squeeze().tolist(),
            tokens['attention_mask'].squeeze().tolist()
        )
    
    return ent_ill, train_ill, test_ill, index2rel, index2entity, rel2index, entity2index, ent2data, rel_triples_1, rel_triples_2


def load_basic_unit_data(
    config: Mapping[str, Any],
    paths: Mapping[str, Any]
) -> BasicUnitDataBundle:
    """Load tensors and metadata required by the basic unit stage."""
    dataset_root = paths.get("dataset_root")
    if dataset_root is None:
        raise ValueError("BERT-INT basic unit requires paths.dataset_root in the configuration.")

    dataset_root_path = Path(dataset_root)
    logger.info("[BERT-INT] Loading basic unit data from %s", dataset_root_path)
    dataset_name = config.get("dataset", {}).get("name") or dataset_root_path.name

    tokenizer = AutoTokenizer.from_pretrained(config.get("encoder_name", "bert-base-multilingual-cased"))
    description_path = paths.get("description_dict")
    description_path = Path(description_path) if description_path else None
    ent_ill, train_ill, test_ill, index2rel, index2entity, rel2index, entity2index, ent2data, rel_triples_1, rel_triples_2 = _read_data(
        dataset_root_path,
        description_path,
        tokenizer,
        dataset_name,
        max_length=config.get("max_seq_length", 128),
    )

    return BasicUnitDataBundle(
        ent_ill=ent_ill,
        train_ill=train_ill,
        test_ill=test_ill,
        index2rel=index2rel,
        index2entity=index2entity,
        rel2index=rel2index,
        entity2index=entity2index,
        ent2data=ent2data,
        rel_triples_1=rel_triples_1,
        rel_triples_2=rel_triples_2,
    )


def _read_data(
    data_path: Path,
    description_path: Optional[Path],
    tokenizer,
    dataset_label: str,
    max_length: int,
):
    logger.info("Loading BERT-INT basic unit data from %s", data_path)

    index2entity = _read_id2object([data_path / "ent_ids_1", data_path / "ent_ids_2"])
    index2rel = _read_id2object([data_path / "rel_ids_1", data_path / "rel_ids_2"])
    entity2index = {entity: idx for idx, entity in index2entity.items()}
    rel2index = {rel: idx for idx, rel in index2rel.items()}

    # DEBUG: Log entity counts
    logger.info("[BERT-INT DEBUG] Total entities loaded: %d", len(index2entity))

    rel_triples_1 = _read_idtuple_file(data_path / "triples_1")
    rel_triples_2 = _read_idtuple_file(data_path / "triples_2")

    ent_ids_1 = _read_idobj_tuple_file(data_path / "ent_ids_1")
    ent_ids_2 = _read_idobj_tuple_file(data_path / "ent_ids_2")

    # DEBUG: Log entity counts per KG
    logger.info("[BERT-INT DEBUG] KG1 entities: %d, KG2 entities: %d", len(ent_ids_1), len(ent_ids_2))

    train_ill = _read_idtuple_file(data_path / "sup_pairs")
    test_ill = _read_idtuple_file(data_path / "ref_pairs")
    ent_ill = list(train_ill) + list(test_ill)

    entid_1 = [ent_id for ent_id, _ in ent_ids_1]
    entid_2 = [ent_id for ent_id, _ in ent_ids_2]
    all_ent_ids = list(range(len(index2entity)))

    attrs = {}
    attrs.update(_get_preferred_attributes(data_path / "attr_triples1", dataset_label))
    attrs.update(_get_preferred_attributes(data_path / "attr_triples2", dataset_label))

    if description_path and description_path.exists():
        ent2des_tokens = _ent2description_tokens(
            tokenizer,
            description_path,
            [index2entity[idx] for idx in entid_1],
            [index2entity[idx] for idx in entid_2],
            max_length=max_length - 2,
        )
    else:
        ent2des_tokens = None

    ent2token_ids = _entities_to_token_ids(
        tokenizer,
        ent2des_tokens,
        all_ent_ids,
        index2entity,
        attrs,
        dataset_label,
        max_length=max_length - 2,
    )
    ent2data = _entities_to_model_inputs(tokenizer, all_ent_ids, ent2token_ids, max_length=max_length)

    return (
        ent_ill,
        train_ill,
        test_ill,
        index2rel,
        index2entity,
        rel2index,
        entity2index,
        ent2data,
        rel_triples_1,
        rel_triples_2,
    )


def _read_idtuple_file(path: Path) -> List[Tuple[int, ...]]:
    values: List[Tuple[int, ...]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            tokens = [int(part) for part in line.rstrip("\n").split("\t")]
            values.append(tuple(tokens))
    return values


def _read_id2object(paths: Sequence[Path]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                idx_str, value = line.rstrip("\n").split("\t")
                mapping[int(idx_str)] = value
    return mapping


def _read_idobj_tuple_file(path: Path) -> List[Tuple[int, str]]:
    values: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            idx_str, value = line.rstrip("\n").split("\t")
            values.append((int(idx_str), value))
    return values


def _ent2description_tokens(
    tokenizer,
    description_path: Path,
    ent_list_1: Iterable[str],
    ent_list_2: Iterable[str],
    max_length: int,
) -> Dict[str, List[int]]:
    logger.info("Loading entity descriptions from %s", description_path)
    with description_path.open("rb") as handle:
        descriptions = pickle.load(handle)
    ent_set_1 = set(ent_list_1)
    ent_set_2 = set(ent_list_2)
    ent2tokens: Dict[str, List[int]] = {}
    for entity, text in descriptions.items():
        if entity not in ent_set_1 and entity not in ent_set_2:
            continue
        # Replicate original behavior: encode WITH special tokens (will be added again later)
        token_ids = tokenizer.encode(text)[:max_length]
        ent2tokens[entity] = token_ids
    logger.info("Loaded descriptions for %d entities", len(ent2tokens))
    return ent2tokens


def _entities_to_token_ids(
    tokenizer,
    ent2des_tokens: Optional[Mapping[str, List[int]]],
    ent_ids: Iterable[int],
    index2entity: Mapping[int, str],
    attributes: Mapping[str, str],
    dataset_label: str,
    max_length: int,
) -> Dict[int, List[int]]:
    ent2token_ids: Dict[int, List[int]] = {}
    for ent_id in ent_ids:
        entity_iri = index2entity[ent_id]
        if ent2des_tokens and entity_iri in ent2des_tokens:
            tokens = ent2des_tokens[entity_iri]
        elif entity_iri in attributes:
            # Replicate original behavior: encode WITH special tokens
            tokens = tokenizer.encode(attributes[entity_iri])[:max_length]
        else:
            surface = _friendly_name(entity_iri, dataset_label)
            # Replicate original behavior: encode WITH special tokens
            tokens = tokenizer.encode(surface)[:max_length]
        ent2token_ids[ent_id] = tokens
    return ent2token_ids


def _entities_to_model_inputs(
    tokenizer,
    ent_ids: Iterable[int],
    ent2token_ids: Mapping[int, List[int]],
    max_length: int,
) -> Dict[int, Tuple[List[int], List[float]]]:
    pad_id = tokenizer.pad_token_id
    ent2data: Dict[int, Tuple[List[int], List[float]]] = {}
    for ent_id in ent_ids:
        token_ids = tokenizer.build_inputs_with_special_tokens(ent2token_ids[ent_id])
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        padding = [pad_id] * (max_length - len(token_ids))
        token_ids = token_ids + padding

        attention_mask = np.ones(len(token_ids), dtype=float)
        attention_mask[np.array(token_ids) == pad_id] = 0.0

        ent2data[ent_id] = (list(token_ids), attention_mask.tolist())
    return ent2data


def _friendly_name(entity: str, dataset_label: str) -> str:
    name = entity
    if "EN_JA" in dataset_label:
        if "http://dbpedia.org/resource" in entity:
            name = entity.split("http://dbpedia.org/resource/")[-1]
        else:
            name = entity.split("http://ja.dbpedia.org/resource/")[-1]
    elif "EN_DE" in dataset_label:
        if "http://dbpedia.org/resource" in entity:
            name = entity.split("http://dbpedia.org/resource/")[-1]
        else:
            name = entity.split("http://de.dbpedia.org/resource/")[-1]
    elif "EN_FR" in dataset_label:
        if "http://dbpedia.org/resource" in entity:
            name = entity.split("http://dbpedia.org/resource/")[-1]
        else:
            name = entity.split("http://fr.dbpedia.org/resource/")[-1]
    elif "DBP_en_YG_en" in dataset_label or "D_Y" in dataset_label:
        if "http://dbpedia.org/resource" in entity:
            name = entity.split("http://dbpedia.org/resource/")[-1]
        else:
            name = entity
    elif "DBP_en_WD_en" in dataset_label or "D_W" in dataset_label:
        if "http://dbpedia.org/resource" in entity:
            name = entity.split("http://dbpedia.org/resource/")[-1]
        else:
            name = entity.split("http://www.wikidata.org/entity/")[-1]
    elif "bbc" in dataset_label:
        if "dbp:" in entity:
            name = entity.split("dbp:")[-1]
        else:
            name = entity.split("/")[-1]
    else:
        name = entity.split("/")[-1]

    return name.replace("_", " ")


def _get_preferred_attributes(path: Path, dataset_label: str) -> Dict[str, str]:
    priority: Dict[str, int] = {}
    path_str = str(path)

    if "EN_JA" in path_str:
        if "attr_triples1" in path_str:
            priority = {
                "http://xmlns.com/foaf/0.1/name": 0,
                "http://dbpedia.org/ontology/birthName": 1,
                "http://xmlns.com/foaf/0.1/nick": 2,
                "http://dbpedia.org/ontology/synonym": 3,
                "http://dbpedia.org/ontology/alias": 4,
                "http://dbpedia.org/ontology/office": 5,
                "http://dbpedia.org/ontology/background": 5,
                "http://dbpedia.org/ontology/leaderTitle": 5,
                "http://dbpedia.org/ontology/orderInOffice": 5,
            }
        else:
            priority = {
                "http://xmlns.com/foaf/0.1/name": 0,
                "http://dbpedia.org/ontology/title": 1,
                "http://dbpedia.org/ontology/commonName": 2,
                "http://xmlns.com/foaf/0.1/nick": 3,
                "http://dbpedia.org/ontology/givenName": 4,
                "http://dbpedia.org/ontology/alias": 5,
                "http://dbpedia.org/ontology/background": 6,
                "http://dbpedia.org/ontology/purpose": 6,
            }
    elif "EN_DE" in path_str:
        if "attr_triples1" in path_str:
            priority = {
                "http://xmlns.com/foaf/0.1/name": 0,
                "http://dbpedia.org/ontology/title": 1,
                "http://dbpedia.org/ontology/birthName": 2,
                "http://xmlns.com/foaf/0.1/nick": 3,
                "http://dbpedia.org/ontology/office": 4,
                "http://dbpedia.org/ontology/leaderTitle": 5,
                "http://dbpedia.org/ontology/orderInOffice": 5,
            }
        else:
            priority = {
                "http://xmlns.com/foaf/0.1/name": 0,
                "http://dbpedia.org/ontology/originalTitle": 1,
                "http://xmlns.com/foaf/0.1/nick": 2,
                "http://dbpedia.org/ontology/motto": 3,
                "http://dbpedia.org/ontology/leaderTitle": 4,
            }
    elif "EN_FR" in path_str:
        if "attr_triples1" in path_str:
            priority = {
                "http://xmlns.com/foaf/0.1/name": 0,
                "http://dbpedia.org/ontology/title": 1,
                "http://dbpedia.org/ontology/birthName": 2,
                "http://xmlns.com/foaf/0.1/nick": 3,
                "http://dbpedia.org/ontology/office": 4,
                "http://dbpedia.org/ontology/leaderTitle": 5,
                "http://dbpedia.org/ontology/motto": 5,
                "http://dbpedia.org/ontology/combatant": 5,
            }
        else:
            priority = {
                "http://xmlns.com/foaf/0.1/name": 0,
                "http://dbpedia.org/ontology/birthName": 1,
                "http://xmlns.com/foaf/0.1/nick": 2,
                "http://dbpedia.org/ontology/peopleName": 3,
                "http://dbpedia.org/ontology/thumbnailCaption": 4,
                "http://dbpedia.org/ontology/flag": 4,
                "http://dbpedia.org/ontology/motto": 5,
                "http://dbpedia.org/ontology/title": 5,
            }
    elif "DBP_en_YG_en" in path_str:
        if "attr_triples1" in path_str:
            priority = {
                "http://xmlns.com/foaf/0.1/name": 0,
                "http://dbpedia.org/ontology/birthName": 1,
                "http://xmlns.com/foaf/0.1/nick": 2,
                "http://dbpedia.org/ontology/alias": 3,
                "http://dbpedia.org/ontology/office": 4,
                "http://dbpedia.org/ontology/leaderTitle": 4,
                "http://dbpedia.org/ontology/motto": 5,
                "http://dbpedia.org/ontology/combatant": 5,
            }
        else:
            priority = {
                "skos:prefLabel": 0,
                "rdfs:label": 1,
                "redirectedFrom": 2,
                "hasFamilyName": 3,
                "hasGivenName": 4,
                "hasMotto": 5,
            }
    elif "DBP_en_WD_en" in path_str or "D_W" in path_str:
        if "attr_triples1" in path_str:
            # Priority from reference implementation (Read_data_func.py line 176-185)
            priority = {
                "http://xmlns.com/foaf/0.1/name": 0,
                "http://dbpedia.org/ontology/birthName": 1,
                "http://purl.org/dc/elements/1.1/description": 2,
                "http://xmlns.com/foaf/0.1/nick": 3,
                "http://xmlns.com/foaf/0.1/givenName": 4,
                "http://dbpedia.org/ontology/leaderTitle": 5,
                "http://dbpedia.org/ontology/alias": 6,
                "http://dbpedia.org/ontology/motto": 7,
                "http://dbpedia.org/ontology/office": 7,
            }
        else:
            # Priority from reference implementation (Read_data_func.py line 169-173)
            priority = {
                "http://www.wikidata.org/entity/P373": 0,
                "http://schema.org/description": 1,
                "http://www.wikidata.org/entity/P1476": 2,
                "http://www.wikidata.org/entity/P935": 3,
                "http://www.w3.org/2004/02/skos/core#altLabel": 4,
            }
    elif "D_Y" in path_str:
        if "attr_triples1" in path_str:
            priority = {
                "http://xmlns.com/foaf/0.1/name": 0,
                "http://dbpedia.org/ontology/birthName": 1,
                "http://purl.org/dc/elements/1.1/description": 2,
                "http://xmlns.com/foaf/0.1/nick": 3,
                "http://xmlns.com/foaf/0.1/givenName": 4,
                "http://dbpedia.org/ontology/leaderTitle": 5,
                "http://dbpedia.org/ontology/alias": 6,
                "http://dbpedia.org/ontology/motto": 7,
                "http://dbpedia.org/ontology/office": 7,
            }
        else:
            priority = {
                "skos:prefLabel": 0,
                "redirectedFrom": 1,
                "hasFamilyName": 2,
                "hasGivenName": 3,
                "hasMotto": 4,
            }
    elif "bbc" in path_str:
        if "attr_triples1" in path_str:
            priority = {
                "http://purl.org/dc/elements/1.1/title": 0,
                "http://xmlns.com/foaf/0.1/name": 1,
                "http://open.vocab.org/terms/sortlabel": 2,
            }
        else:
            priority = {"prop:title": 0}

    attributes: Dict[str, Tuple[str, str]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                ent, predicate, value = line.rstrip("\n").split("\t")
                if predicate in priority:
                    if ent in attributes:
                        if priority[predicate] < priority[attributes[ent][0]]:
                            attributes[ent] = (predicate, value)
                    else:
                        attributes[ent] = (predicate, value)

    return {entity: value for entity, (_, value) in attributes.items()}
