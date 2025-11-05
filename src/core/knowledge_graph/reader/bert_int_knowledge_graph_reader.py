"""BERT-INT knowledge graph reader."""
from pathlib import Path

from rdflib import URIRef

from src.core.knowledge_graph import KnowledgeGraph
from src.core.knowledge_graph.reader.knowledge_graph_reader_base import (
    KnowledgeGraphReader,
)
from src.logger import get_logger
from src.util.reader import read_tsv

logger = get_logger(__name__)


class BertIntKnowledgeGraphReader(KnowledgeGraphReader):
    """Read knowledge graphs in BERT-INT native format."""

    file_type = "bert_int"

    def read(self, dir_path: str, kg_number: int, **_) -> KnowledgeGraph:
        """
        Read a knowledge graph from BERT-INT format.

        Args:
            dir_path: Directory containing BERT-INT files
            kg_number: Knowledge graph number (1 or 2)

        Returns:
            KnowledgeGraph object
        """
        dir_path = Path(dir_path)
        logger.debug(f"Reading BERT-INT KG {kg_number} from {dir_path}")

        # File paths for BERT-INT format
        ent_ids_file = dir_path / f"ent_ids_{kg_number}"
        rel_ids_file = dir_path / f"rel_ids_{kg_number}"
        triples_file = dir_path / f"triples_{kg_number}"

        # Read entity ID mapping (index -> URI)
        ent_data = read_tsv(str(ent_ids_file))
        index2entity = {int(row[0]): row[1] for row in ent_data}

        # Read relation ID mapping (index -> URI)
        rel_data = read_tsv(str(rel_ids_file))
        index2relation = {int(row[0]): row[1] for row in rel_data}

        # Read triples (as indices)
        triple_data = read_tsv(str(triples_file))

        # Create knowledge graph
        kg = KnowledgeGraph()

        # Convert indexed triples to URIs and add to graph
        for row in triple_data:
            subj_idx = int(row[0])
            pred_idx = int(row[1])
            obj_idx = int(row[2])

            subj_uri = URIRef(index2entity[subj_idx])
            pred_uri = URIRef(index2relation[pred_idx])
            obj_uri = URIRef(index2entity[obj_idx])

            kg.add((subj_uri, pred_uri, obj_uri))

        logger.info(
            f"BERT-INT KG {kg_number}: loaded {len(index2entity)} entities, "
            f"{len(index2relation)} relations, {len(triple_data)} triples"
        )

        return kg
