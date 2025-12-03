"""Data loading utilities for RREA from pre-processed OpenEA format."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, Tuple, NamedTuple

from src.logger import get_logger
from src.utils.reader import read_tsv

logger = get_logger(__name__)


class RREADataBundle(NamedTuple):
    """Bundle of pre-processed data for RREA training."""
    entity2id: Dict[str, int]  # URI → ID mapping
    id2entity: Dict[int, str]  # ID → URI mapping
    relation2id: Dict[str, int]  # URI → ID mapping
    triples: np.ndarray  # All triples [num_triples, 3] (h, r, t)
    aligned_pairs: np.ndarray  # Aligned entity pairs [num_pairs, 2]
    kg2_offset: int  # Offset for KG2 entity IDs


def load_rrea_data(dataset_root: str | Path) -> RREADataBundle:
    """Load pre-processed OpenEA-format data for RREA.

    Args:
        dataset_root: Path to dataset directory containing OpenEA files

    Returns:
        RREADataBundle with all necessary data structures

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If data format is invalid
    """
    dataset_root = Path(dataset_root)
    logger.info(f"[RREA] Loading pre-processed data from {dataset_root}")

    # Determine if we're using attribute_data or knowformer_data layout
    if (dataset_root / "attribute_data").exists():
        data_dir = dataset_root / "attribute_data"
        logger.info("[RREA] Using attribute_data layout")
    elif (dataset_root / "knowformer_data").exists():
        data_dir = dataset_root / "knowformer_data"
        logger.info("[RREA] Using knowformer_data layout")
    else:
        # Assume we're already in the data directory
        data_dir = dataset_root
        logger.info("[RREA] Using direct data directory")

    # Load entity IDs
    entity2id = {}
    id2entity = {}

    ent_ids_1_path = data_dir / "ent_ids_1"
    ent_ids_2_path = data_dir / "ent_ids_2"

    if not ent_ids_1_path.exists():
        raise FileNotFoundError(f"Missing {ent_ids_1_path}")
    if not ent_ids_2_path.exists():
        raise FileNotFoundError(f"Missing {ent_ids_2_path}")

    logger.info("[RREA] Loading entity mappings...")
    ent_ids_1 = read_tsv(str(ent_ids_1_path))
    for ent_id, ent_uri in ent_ids_1:
        ent_id_int = int(ent_id)
        entity2id[ent_uri] = ent_id_int
        id2entity[ent_id_int] = ent_uri

    kg2_offset = len(entity2id)
    logger.info(f"[RREA] KG1 entities: {kg2_offset}")

    ent_ids_2 = read_tsv(str(ent_ids_2_path))
    for ent_id, ent_uri in ent_ids_2:
        ent_id_int = int(ent_id)
        entity2id[ent_uri] = ent_id_int
        id2entity[ent_id_int] = ent_uri

    logger.info(f"[RREA] KG2 entities: {len(entity2id) - kg2_offset}")
    logger.info(f"[RREA] Total entities: {len(entity2id)}")

    # Load relation IDs
    relation2id = {}

    rel_ids_1_path = data_dir / "rel_ids_1"
    rel_ids_2_path = data_dir / "rel_ids_2"

    if not rel_ids_1_path.exists():
        raise FileNotFoundError(f"Missing {rel_ids_1_path}")
    if not rel_ids_2_path.exists():
        raise FileNotFoundError(f"Missing {rel_ids_2_path}")

    logger.info("[RREA] Loading relation mappings...")
    rel_ids_1 = read_tsv(str(rel_ids_1_path))
    for rel_id, rel_uri in rel_ids_1:
        relation2id[rel_uri] = int(rel_id)

    rel_ids_2 = read_tsv(str(rel_ids_2_path))
    for rel_id, rel_uri in rel_ids_2:
        rel_id_int = int(rel_id)
        if rel_uri not in relation2id:
            relation2id[rel_uri] = rel_id_int

    logger.info(f"[RREA] Total relations: {len(relation2id)}")

    # Load triples
    triples_1_path = data_dir / "triples_1"
    triples_2_path = data_dir / "triples_2"

    if not triples_1_path.exists():
        raise FileNotFoundError(f"Missing {triples_1_path}")
    if not triples_2_path.exists():
        raise FileNotFoundError(f"Missing {triples_2_path}")

    logger.info("[RREA] Loading triples...")
    triples_list = []

    triples_1 = read_tsv(str(triples_1_path))
    for h, r, t in triples_1:
        triples_list.append([int(h), int(r), int(t)])

    triples_2 = read_tsv(str(triples_2_path))
    for h, r, t in triples_2:
        triples_list.append([int(h), int(r), int(t)])

    triples = np.array(triples_list, dtype=np.int32)
    logger.info(f"[RREA] Total triples: {len(triples)}")

    # Load aligned entity pairs
    # Try different possible filenames (in priority order)
    ent_links_paths = [
        data_dir / "ent_links",       # Original OpenEA format
        data_dir / "ref_ent_ids",     # Alternative format
        data_dir / "sup_ent_ids",     # Supervision format
        data_dir / "ref_pairs",       # DAKGEA OpenEA writer format (training)
        data_dir / "sup_pairs",       # DAKGEA OpenEA writer format (supervision)
        data_dir / "valid_pairs",     # DAKGEA OpenEA writer format (validation)
    ]

    # Collect all aligned pairs from available files
    aligned_pairs_list = []
    files_found = []

    for path in ent_links_paths:
        if path.exists():
            files_found.append(path.name)
            logger.info(f"[RREA] Loading aligned pairs from {path.name}...")
            ent_links = read_tsv(str(path))

            for row in ent_links:
                # Handle different formats: (src_id, tgt_id) or just (id1, id2)
                if len(row) >= 2:
                    src_id, tgt_id = int(row[0]), int(row[1])
                    aligned_pairs_list.append([src_id, tgt_id])

    if len(aligned_pairs_list) == 0:
        raise FileNotFoundError(
            f"No aligned entities found. Tried: {[str(p) for p in ent_links_paths]}"
        )

    aligned_pairs = np.array(aligned_pairs_list, dtype=np.int32)
    logger.info(f"[RREA] Aligned pairs loaded from: {', '.join(files_found)}")
    logger.info(f"[RREA] Total aligned pairs: {len(aligned_pairs)}")

    logger.info("[RREA] Data loading completed successfully")

    return RREADataBundle(
        entity2id=entity2id,
        id2entity=id2entity,
        relation2id=relation2id,
        triples=triples,
        aligned_pairs=aligned_pairs,
        kg2_offset=kg2_offset,
    )
