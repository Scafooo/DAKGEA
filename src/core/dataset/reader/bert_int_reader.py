"""BERT-INT format dataset reader.

Reads datasets in the format written by BERT-INT writer.
"""

import os
import unicodedata
from pathlib import Path
from typing import Dict, List

from rdflib import URIRef

from src.core.dataset import Dataset
from src.core.dataset.reader.dataset_reader_base import DatasetReader
from src.core.knowledge_graph.reader import KnowledgeGraphReaderFactory
from src.logger import get_logger
from src.util.reader import read_tsv

logger = get_logger(__name__)


def _load_attribute_matches(data_dir: Path) -> Dict[str, List[str]]:
    """Load attribute matches from match_attr file.

    Args:
        data_dir: Path to the dataset directory

    Returns:
        Dict mapping source_uri -> [list of target_uris]
    """
    match_file = data_dir / "match_attr"

    if not match_file.exists():
        return {}

    matches: Dict[str, List[str]] = {}

    try:
        with open(match_file, 'r') as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Parse: attr1[,attr1.2] <TAB> attr2[,attr2.2]
                parts = line.split('\t')
                if len(parts) != 2:
                    logger.warning(f"Malformed match_attr line: {line}")
                    continue

                src_attrs = [a.strip() for a in parts[0].split(',')]
                tgt_attrs = [a.strip() for a in parts[1].split(',')]

                # Create matches for all combinations
                for src_attr in src_attrs:
                    if src_attr not in matches:
                        matches[src_attr] = []
                    matches[src_attr].extend(tgt_attrs)

        if matches:
            logger.info(f"Loaded {len(matches)} attribute matches from {match_file}")

    except Exception as e:
        logger.warning(f"Failed to load match_attr file from {match_file}: {e}")
        return {}

    return matches


class BertIntReader(DatasetReader):
    """Read datasets in BERT-INT native format."""

    file_type = "bert_int"

    def read(self, dataset_root: str, **kwargs) -> Dataset:
        """Read dataset from BERT-INT format files.

        Args:
            dataset_root: Directory containing BERT-INT files

        Returns:
            Dataset object
        """
        root = Path(dataset_root)
        logger.info(f"[BERT-INT Reader] Loading dataset from {root}")

        # Use KG reader to load knowledge graphs
        kg_reader = KnowledgeGraphReaderFactory.create_reader(self.file_type)

        logger.debug(f"Reading KG1 (Source) from {root}")
        kg1 = kg_reader.read(str(root), kg_number=1)

        logger.debug(f"Reading KG2 (Target) from {root}")
        kg2 = kg_reader.read(str(root), kg_number=2)

        # Read aligned entities
        logger.debug("Reading aligned entities")
        aligned_entities = self._read_aligned_entities(root)

        # Load attribute matches if available
        attribute_matches = _load_attribute_matches(root)

        dataset = Dataset(
            knowledge_graph_source=kg1,
            knowledge_graph_target=kg2,
            aligned_entities=aligned_entities,
            attribute_matches=attribute_matches,
        )

        logger.info(
            f"[BERT-INT Reader] Loaded dataset: {len(kg1)} source triples, "
            f"{len(kg2)} target triples, {len(aligned_entities)} aligned pairs"
        )

        return dataset

    def _read_aligned_entities(self, root: Path):
        """Read aligned entity pairs from sup_pairs, ref_pairs, valid_pairs files.

        Args:
            root: Directory containing BERT-INT files

        Returns:
            Set of aligned entity pairs (as URIRef tuples)
        """
        # Read entity ID mappings
        ent_ids_1 = read_tsv(str(root / "ent_ids_1"))
        ent_ids_2 = read_tsv(str(root / "ent_ids_2"))

        # Build index -> entity URI mapping
        # Note: BERT-INT format already includes offset in ent_ids_2 indices
        # ent_ids_1: 0, 1, 2, ... , N-1
        # ent_ids_2: N, N+1, N+2, ... , N+M-1
        # So we just read indices as-is from both files
        index2entity_combined = {}

        for row in ent_ids_1:
            idx = int(row[0])
            uri = unicodedata.normalize("NFC", row[1])
            index2entity_combined[idx] = uri

        for row in ent_ids_2:
            idx = int(row[0])
            uri = unicodedata.normalize("NFC", row[1])
            index2entity_combined[idx] = uri

        # Read alignment files
        aligned_entities = set()

        for filename in ["sup_pairs", "ref_pairs", "valid_pairs"]:
            filepath = root / filename
            if not filepath.exists():
                logger.warning(f"Alignment file not found: {filepath}")
                continue

            pairs = read_tsv(str(filepath))
            for row in pairs:
                idx1 = int(row[0])
                idx2 = int(row[1])

                if idx1 not in index2entity_combined:
                    logger.warning(f"Entity index {idx1} not found in mapping")
                    continue
                if idx2 not in index2entity_combined:
                    logger.warning(f"Entity index {idx2} not found in mapping")
                    continue

                entity1 = URIRef(index2entity_combined[idx1])
                entity2 = URIRef(index2entity_combined[idx2])
                aligned_entities.add((entity1, entity2))

        logger.debug(f"Read {len(aligned_entities)} aligned entity pairs")

        return aligned_entities
