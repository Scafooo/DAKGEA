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
from src.utils.reader import read_tsv
from src.utils.writer import write_tsv

logger = get_logger(__name__)


class BertIntWriter(DatasetWriter):
    """Write datasets in BERT-INT native format."""

    file_type = "bert_int"

    def __init__(self, augmented_only_train: bool = False):
        """Initialize BERT-INT writer.

        Args:
            augmented_only_train: If True, put all augmented entities in training set,
                                  and split only original entities across train/test/valid.
                                  Default False maintains backward compatibility.
        """
        self.augmented_only_train = augmented_only_train
        if augmented_only_train:
            logger.info("[BERT-INT Writer] Augmented-only-train mode enabled: "
                       "augmented entities will be added to training set only")

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

        if self.augmented_only_train:
            original_pairs = []
            augmented_pairs = []

            for e1, e2 in list_aligned_entities:
                if "_aug" in e1 or "_aug" in e2:
                    augmented_pairs.append((e1, e2))
                else:
                    original_pairs.append((e1, e2))

            logger.debug(
                f"[augmented_only_train] Separated {len(original_pairs)} original + "
                f"{len(augmented_pairs)} augmented from {len(list_aligned_entities)} total pairs"
            )

            logger.info(
                f"Aligned entities: {len(list_aligned_entities)} total "
                f"({len(original_pairs)} original, {len(augmented_pairs)} augmented)"
            )

            # Split only original entities: 20% train, 70% test, 10% valid
            n = len(original_pairs)
            n1 = int(n * 0.2)  # 20% for training
            n2 = int(n * 0.9)  # 20% + 70% = 90% (cumulative: train + test)

            logger.info(
                f"Original entities split: {n1} train, {n2-n1} test, {n-n2} valid"
            )
            logger.info(
                f"Augmented entities: {len(augmented_pairs)} (all added to train)"
            )

            # Create splits: original split + all augmented in train
            sup_pairs_original = [[ent_ids[e1], ent_ids[e2]] for e1, e2 in original_pairs[:n1]]
            sup_pairs_augmented = [[ent_ids[e1], ent_ids[e2]] for e1, e2 in augmented_pairs]
            sup_pairs = sup_pairs_original + sup_pairs_augmented

            ref_pairs = [[ent_ids[e1], ent_ids[e2]] for e1, e2 in original_pairs[n1:n2]]
            valid_pairs = [[ent_ids[e1], ent_ids[e2]] for e1, e2 in original_pairs[n2:]]

        else:
            # Default mode: split all entities uniformly
            # Split: 20% sup (train), 70% ref (test), 10% valid
            n = len(list_aligned_entities)
            n1 = int(n * 0.2)  # 20% for training
            n2 = int(n * 0.9)  # 20% + 70% = 90% (cumulative: train + test)

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
