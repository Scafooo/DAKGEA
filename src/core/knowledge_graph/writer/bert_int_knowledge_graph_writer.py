"""BERT-INT knowledge graph writer."""
import os
from pathlib import Path

from rdflib import Literal

from src.core.knowledge_graph import KnowledgeGraph
from src.core.knowledge_graph.writer.knowledge_graph_writer_base import (
    KnowledgeGraphWriter,
)
from src.logger import get_logger
from src.utils.writer import write_tsv

logger = get_logger(__name__)


class BertIntKnowledgeGraphWriter(KnowledgeGraphWriter):
    """Write knowledge graphs in BERT-INT native format."""

    file_type = "bert_int"

    def write(self, kg: KnowledgeGraph, dir_path: str, kg_number: int) -> bool:
        """
        Write a knowledge graph in BERT-INT format.

        Args:
            kg: Knowledge graph to write
            dir_path: Output directory path
            kg_number: Knowledge graph number (1 or 2)

        Returns:
            True if successful
        """
        logger.info(f"Writing BERT-INT KG {kg_number} to {dir_path}")

        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # File paths for BERT-INT format
        ent_ids_file = dir_path / f"ent_ids_{kg_number}"
        rel_ids_file = dir_path / f"rel_ids_{kg_number}"
        triples_file = dir_path / f"triples_{kg_number}"
        attr_triples_file = dir_path / f"attr_triples{kg_number}"

        # Build mappings
        entity2index = {}
        relation2index = {}
        triples = []
        attr_triples = []

        # Sort triples for consistency
        ordered_triples = sorted([[s, p, o] for s, p, o in kg])

        # For KG2, read KG1 entity count to apply offset
        ent_idx = 0
        rel_idx = 0
        if kg_number == 2:
            kg1_ent_file = dir_path / "ent_ids_1"
            if kg1_ent_file.exists():
                from src.utils.reader import read_tsv
                kg1_ents = read_tsv(str(kg1_ent_file))
                ent_idx = len(kg1_ents)
                logger.debug(f"KG2: starting entity index at {ent_idx} (offset from KG1)")

            kg1_rel_file = dir_path / "rel_ids_1"
            if kg1_rel_file.exists():
                from src.utils.reader import read_tsv
                kg1_rels = read_tsv(str(kg1_rel_file))
                rel_idx = len(kg1_rels)
                logger.debug(f"KG2: starting relation index at {rel_idx} (offset from KG1)")

        # Process triples: separate relation triples from attribute triples
        for s, p, o in ordered_triples:
            # Index subject entity (always present)
            if str(s) not in entity2index:
                entity2index[str(s)] = ent_idx
                ent_idx += 1

            if isinstance(o, Literal):
                # Attribute triple: subject_uri predicate_uri literal_value
                # Note: predicates in attr_triples are NOT indexed in rel_ids
                attr_triples.append((str(s), str(p), str(o)))
            else:
                # Relation triple: index predicate and object entity
                # Only relation predicates are indexed in rel_ids
                if str(p) not in relation2index:
                    relation2index[str(p)] = rel_idx
                    rel_idx += 1

                if str(o) not in entity2index:
                    entity2index[str(o)] = ent_idx
                    ent_idx += 1

                # Store triple as indices
                triples.append(
                    (
                        str(entity2index[str(s)]),
                        str(relation2index[str(p)]),
                        str(entity2index[str(o)]),
                    )
                )

        # Write entity IDs (index, URI)
        ent_data = [
            [str(idx), uri]
            for idx, uri in sorted(
                [(idx, uri) for uri, idx in entity2index.items()], key=lambda x: x[0]
            )
        ]
        write_tsv(str(ent_ids_file), ent_data)

        # Write relation IDs (index, URI)
        rel_data = [
            [str(idx), uri]
            for idx, uri in sorted(
                [(idx, uri) for uri, idx in relation2index.items()], key=lambda x: x[0]
            )
        ]
        write_tsv(str(rel_ids_file), rel_data)

        # Write triples (as indices)
        write_tsv(str(triples_file), triples)

        # Write attribute triples (subject_uri, predicate_uri, literal_value)
        if attr_triples:
            write_tsv(str(attr_triples_file), attr_triples)
            logger.info(f"BERT-INT KG {kg_number}: wrote {len(attr_triples)} attribute triples")

        logger.info(
            f"BERT-INT KG {kg_number}: {len(entity2index)} entities, "
            f"{len(relation2index)} relations, {len(triples)} relation triples, "
            f"{len(attr_triples)} attribute triples"
        )

        return True
