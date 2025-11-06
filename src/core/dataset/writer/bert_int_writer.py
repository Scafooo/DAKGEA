"""BERT-INT format dataset writer.

Writes datasets in the format expected by the original BERT-INT implementation.
"""

import os
import unicodedata
from pathlib import Path

from src.core.dataset.dataset import Dataset
from src.core.dataset.writer.dataset_writer_base import DatasetWriter
from src.core.knowledge_graph.writer import KnowledgeGraphWriterFactory
from src.logger import get_logger
from src.util.reader import read_tsv
from src.util.writer import write_tsv

logger = get_logger(__name__)


class BertIntWriter(DatasetWriter):
    """Write datasets in BERT-INT native format."""

    file_type = "bert_int"

    def write(self, dataset: Dataset, output_dir: str) -> None:
        """Write dataset to BERT-INT format files.

        Args:
            dataset: Dataset to write
            output_dir: Directory where files will be written
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"[BERT-INT Writer] Writing dataset to {output_path}")

        # Use KG writer to write knowledge graphs
        logger.info(f"Writing KG1 (Source) to {output_path}")
        kg_writer = KnowledgeGraphWriterFactory.create_writer(self.file_type)
        kg_writer.write(dataset.knowledge_graph_source, str(output_path), kg_number=1)

        logger.info(f"Writing KG2 (Target) to {output_path}")
        kg_writer = KnowledgeGraphWriterFactory.create_writer(self.file_type)
        kg_writer.write(dataset.knowledge_graph_target, str(output_path), kg_number=2)

        # Write alignment pairs (sup_pairs, ref_pairs, valid_pairs)
        logger.info("Writing aligned entities")
        self._write_aligned_entities(dataset, str(output_path))

        logger.info("[BERT-INT Writer] Dataset writing completed")

    def _write_aligned_entities(self, dataset: Dataset, dir_path: str) -> None:
        """Write aligned entity pairs to sup_pairs, ref_pairs, valid_pairs files.

        Args:
            dataset: Dataset containing aligned entities
            dir_path: Directory path where files will be written
        """
        # Read entity ID mappings created by KG writer
        ent_ids_1 = read_tsv(os.path.join(dir_path, "ent_ids_1"))
        ent_ids_2 = read_tsv(os.path.join(dir_path, "ent_ids_2"))

        # Build entity URI -> index mapping
        ent_ids = {}
        for elem in ent_ids_1:
            key = unicodedata.normalize("NFC", str(elem[1]))
            ent_ids[key] = str(elem[0])

        # KG writer already applied offset to KG2 indices, use them as-is
        for elem in ent_ids_2:
            key = unicodedata.normalize("NFC", str(elem[1]))
            # Don't add offset - indices in file already have it from KG writer
            ent_ids[key] = str(elem[0])

        # Normalize and sort aligned pairs
        def norm_entity(entity):
            return unicodedata.normalize("NFC", str(entity))

        aligned_list = list(dataset.aligned_entities)
        normalised_pairs = sorted(
            [(norm_entity(e1), norm_entity(e2)) for e1, e2 in aligned_list]
        )

        # Filter out pairs with entities missing from KG
        missing = []
        list_aligned_entities = []
        for e1, e2 in normalised_pairs:
            if e1 not in ent_ids or e2 not in ent_ids:
                missing.append((e1, e2))
                continue
            list_aligned_entities.append((e1, e2))

        if missing:
            sample = ", ".join([f"({e1[:50]}, {e2[:50]})" for e1, e2 in missing[:3]])
            logger.warning(
                f"Skipping {len(missing)} aligned pairs missing from entity mappings. "
                f"Sample: {sample}"
            )

        # Split: 20% sup (train), 70% ref (test), 10% valid
        n = len(list_aligned_entities)
        n1 = int(n * 0.2)  # 20% for training
        n2 = int(n * 0.9)  # 20% + 70% = 90% (cumulative: train + test)
                           # remaining 10% for validation

        logger.info(
            f"Aligned entities: {n} total, {n1} sup (train), "
            f"{n2-n1} ref (test), {n-n2} valid"
        )

        # Convert to indices (using combined index space with offset for target)
        sup_pairs = [[ent_ids[e1], ent_ids[e2]] for e1, e2 in list_aligned_entities[:n1]]
        ref_pairs = [[ent_ids[e1], ent_ids[e2]] for e1, e2 in list_aligned_entities[n1:n2]]
        valid_pairs = [[ent_ids[e1], ent_ids[e2]] for e1, e2 in list_aligned_entities[n2:]]

        # Write files
        write_tsv(os.path.join(dir_path, "sup_pairs"), sup_pairs)
        write_tsv(os.path.join(dir_path, "ref_pairs"), ref_pairs)
        write_tsv(os.path.join(dir_path, "valid_pairs"), valid_pairs)

        logger.info(
            f"Wrote alignment files: sup_pairs ({len(sup_pairs)}), "
            f"ref_pairs ({len(ref_pairs)}), valid_pairs ({len(valid_pairs)})"
        )
