"""Dataset writer for supervision level experiments with fixed test sets.

This writer handles the special requirements of supervision experiments where:
1. The test set (M_test) is fixed across all supervision levels
2. The training set comes from the dataset's aligned_entities (M_train + M_aug)
3. Validation is taken as a small fraction of training
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path
from typing import Optional, Set, Tuple

from rdflib import URIRef

from src.core.dataset.dataset import Dataset
from src.core.knowledge_graph.writer import KnowledgeGraphWriterFactory
from src.logger import get_logger
from src.utils.reader import read_tsv
from src.utils.writer import write_tsv

logger = get_logger(__name__)


class SupervisionExperimentWriter:
    """Write datasets for supervision level experiments with a fixed test set.

    Unlike BertIntWriter which splits aligned_entities into train/test/valid,
    this writer:
    - Takes the test set (M_test) as an explicit parameter
    - Uses dataset.aligned_entities as the full training pool (M_train + M_aug)
    - Splits training pool into sup_pairs (training) and valid_pairs (validation)
    - Writes the fixed test set to ref_pairs

    This ensures apples-to-apples comparison across supervision levels.
    """

    file_type = "bert_int"

    def __init__(
        self,
        fixed_test_pairs: Set[Tuple[URIRef, URIRef]],
        validation_ratio: float = 0.1,
        augmented_in_train_only: bool = True,
    ):
        """Initialize writer with fixed test set.

        Args:
            fixed_test_pairs: The fixed test set (M_test) for evaluation
            validation_ratio: Fraction of training pairs to use for validation
                             (default 0.1 = 10% of training for validation)
            augmented_in_train_only: If True, augmented entities only go to training,
                                    not validation. Original entities are split.
        """
        self.fixed_test_pairs = fixed_test_pairs
        self.validation_ratio = validation_ratio
        self.augmented_in_train_only = augmented_in_train_only

        logger.info(
            f"[SupervisionExperimentWriter] Fixed test set: {len(fixed_test_pairs)} pairs, "
            f"validation_ratio={validation_ratio:.0%}"
        )

    def write(self, dataset: Dataset, output_dir: str) -> None:
        """Write dataset with fixed test set.

        The dataset's aligned_entities are treated as the training pool.
        Test pairs are fixed (provided at init).

        Args:
            dataset: Dataset to write (aligned_entities = training pool)
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"[SupervisionExperimentWriter] Writing to {output_path}")

        # Write knowledge graphs using standard KG writer
        kg_writer = KnowledgeGraphWriterFactory.create_writer(self.file_type)
        kg_writer.write(dataset.knowledge_graph_source, str(output_path), kg_number=1)
        kg_writer.write(dataset.knowledge_graph_target, str(output_path), kg_number=2)

        # Write alignment pairs with fixed test set
        self._write_aligned_entities(dataset, str(output_path))

        logger.info("[SupervisionExperimentWriter] Dataset writing completed")

    def _write_aligned_entities(self, dataset: Dataset, dir_path: str) -> None:
        """Write alignment files with fixed test set."""
        # Read entity ID mappings
        ent_ids_1 = read_tsv(os.path.join(dir_path, "ent_ids_1"))
        ent_ids_2 = read_tsv(os.path.join(dir_path, "ent_ids_2"))

        # Build URI -> index mapping
        ent_ids = {}
        for elem in ent_ids_1:
            key = unicodedata.normalize("NFC", str(elem[1]))
            ent_ids[key] = str(elem[0])
        for elem in ent_ids_2:
            key = unicodedata.normalize("NFC", str(elem[1]))
            ent_ids[key] = str(elem[0])

        def norm_entity(entity):
            return unicodedata.normalize("NFC", str(entity))

        # Get training pool from dataset (M_train + M_aug)
        training_pool = [
            (norm_entity(e1), norm_entity(e2))
            for e1, e2 in dataset.aligned_entities
        ]

        # Separate original and augmented if needed
        if self.augmented_in_train_only:
            original_train = []
            augmented_train = []

            for e1, e2 in training_pool:
                if "_aug" in e1 or "_aug" in e2:
                    augmented_train.append((e1, e2))
                else:
                    original_train.append((e1, e2))

            # Split only original entities for validation
            n_original = len(original_train)
            n_valid = max(1, int(n_original * self.validation_ratio))

            valid_pairs_list = sorted(original_train)[:n_valid]
            train_pairs_list = sorted(original_train)[n_valid:] + sorted(augmented_train)

            logger.info(
                f"Training split: {len(train_pairs_list)} train "
                f"({len(original_train)-n_valid} original + {len(augmented_train)} augmented), "
                f"{len(valid_pairs_list)} valid"
            )
        else:
            # Simple split: validation from all training pairs
            training_sorted = sorted(training_pool)
            n_train = len(training_sorted)
            n_valid = max(1, int(n_train * self.validation_ratio))

            valid_pairs_list = training_sorted[:n_valid]
            train_pairs_list = training_sorted[n_valid:]

            logger.info(
                f"Training split: {len(train_pairs_list)} train, {len(valid_pairs_list)} valid"
            )

        # Process fixed test set
        test_pairs_list = sorted([
            (norm_entity(e1), norm_entity(e2))
            for e1, e2 in self.fixed_test_pairs
        ])

        # Filter out pairs with missing entity mappings
        def filter_pairs(pairs, name):
            filtered = []
            missing = 0
            for e1, e2 in pairs:
                if e1 in ent_ids and e2 in ent_ids:
                    filtered.append((e1, e2))
                else:
                    missing += 1
            if missing > 0:
                logger.warning(f"{name}: {missing} pairs skipped (missing entity mappings)")
            return filtered

        train_pairs_list = filter_pairs(train_pairs_list, "train")
        valid_pairs_list = filter_pairs(valid_pairs_list, "valid")
        test_pairs_list = filter_pairs(test_pairs_list, "test")

        # Convert to indices
        sup_pairs = [[ent_ids[e1], ent_ids[e2]] for e1, e2 in train_pairs_list]
        valid_pairs = [[ent_ids[e1], ent_ids[e2]] for e1, e2 in valid_pairs_list]
        ref_pairs = [[ent_ids[e1], ent_ids[e2]] for e1, e2 in test_pairs_list]

        # Write files
        write_tsv(os.path.join(dir_path, "sup_pairs"), sup_pairs)
        write_tsv(os.path.join(dir_path, "valid_pairs"), valid_pairs)
        write_tsv(os.path.join(dir_path, "ref_pairs"), ref_pairs)

        logger.info(
            f"Wrote alignment files: "
            f"sup_pairs (train): {len(sup_pairs)}, "
            f"ref_pairs (FIXED test): {len(ref_pairs)}, "
            f"valid_pairs: {len(valid_pairs)}"
        )


__all__ = ["SupervisionExperimentWriter"]
