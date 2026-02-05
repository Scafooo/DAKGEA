"""Forget-labels reduction strategy - reduces training pairs while preserving test split."""

import logging
import random
from pathlib import Path
from typing import Set, Tuple

from rdflib import URIRef

from src.core.dataset import Dataset
from src.reduction.methods.random_entities.reducer_random_entities import RandomEntitiesReducer
from src.reduction.registry import REDUCTION_REGISTRY
from src.utils.reader import read_tsv

logger = logging.getLogger(__name__)


@REDUCTION_REGISTRY.register("forget_labels")
class ForgetLabelsReducer(RandomEntitiesReducer):
    """Reduce training pairs while preserving original train/test splits."""

    def __init__(self, config):
        super().__init__(config)

        exp_cfg = config.get("experiment", config)
        reduction_cfg = exp_cfg.get("reduction", config.get("reduction", {}))

        self.ratio = reduction_cfg.get("ratio")

        # Extract dataset name
        ds_info = exp_cfg.get("dataset", config.get("dataset", {}))
        if isinstance(ds_info, dict):
            self.dataset_name = ds_info.get("name", "")
        else:
            self.dataset_name = str(ds_info)

        logger.info(f"[ForgetLabels] Initialized: dataset='{self.dataset_name}', ratio={self.ratio}")

        self.raw_data_root = Path("data/raw")
        if not self.raw_data_root.exists():
            self.raw_data_root = Path(".")

    def _load_split_pairs(self, dataset_name: str) -> Tuple[Set, Set]:
        """Load original train/test splits from dataset files."""
        # Clean dataset name (remove openea/ prefix if present for search)
        clean_name = dataset_name.replace("openea/", "")

        candidates = [
            self.raw_data_root / "openea" / clean_name,
            self.raw_data_root / clean_name,
            Path(dataset_name)
        ]

        found_dir = None
        for p in candidates:
            # Check for attribute_data subdir which usually contains the splits
            if p.exists():
                if (p / "attribute_data" / "ent_ids_1").exists():
                    found_dir = p / "attribute_data"
                    break
                if (p / "ent_ids_1").exists():
                    found_dir = p
                    break

        if not found_dir:
            logger.warning(
                f"Could not locate raw files for '{dataset_name}' (cleaned: '{clean_name}'). "
                f"Checked: {[str(c) for c in candidates]}. Performing random 20/80 split."
            )
            return None, None

        logger.info(f"Loading original splits from {found_dir}")

        # Load ID mapping
        ent_ids = {}
        for fname in ["ent_ids_1", "ent_ids_2"]:
            fpath = found_dir / fname
            if fpath.exists():
                for src, dst in read_tsv(fpath):
                    ent_ids[src] = dst

        def load_pairs(fname) -> Set[Tuple[URIRef, URIRef]]:
            pairs = set()
            fpath = found_dir / fname
            if fpath.exists():
                for left, right in read_tsv(fpath):
                    if left in ent_ids and right in ent_ids:
                        pairs.add((URIRef(ent_ids[left]), URIRef(ent_ids[right])))
            return pairs

        train_pairs = load_pairs("sup_pairs")
        test_pairs = load_pairs("ref_pairs")
        valid_pairs = load_pairs("valid_pairs")

        final_test = test_pairs.union(valid_pairs)
        return train_pairs, final_test

    def reduce(self, dataset: Dataset) -> Dataset:
        """Apply forget-labels reduction preserving test split."""
        logger.info(f"[STEP] ForgetLabels (Split-Aware) reduction started for {self.dataset_name}")

        all_aligned = self._normalise_alignment(dataset.aligned_entities)
        total_original = len(all_aligned)

        # 1. Identify Splits
        train_pool, test_pool = self._load_split_pairs(self.dataset_name)

        if train_pool is None:
            # Fallback: Random 20/80 split
            all_list = list(all_aligned)
            random.seed(self.seed)
            random.shuffle(all_list)
            split_idx = int(len(all_list) * 0.2)
            train_pool = set(all_list[:split_idx])
            test_pool = set(all_list[split_idx:])
        else:
            # Filter loaded pools
            train_pool = train_pool.intersection(all_aligned)
            test_pool = test_pool.intersection(all_aligned)

            remaining = all_aligned - train_pool - test_pool
            if remaining:
                logger.info(f"Found {len(remaining)} pairs not in standard splits. Adding to Test pool.")
                test_pool.update(remaining)

        logger.info(f"Split Layout: Training Pool: {len(train_pool)}, Test Pool: {len(test_pool)}")

        # 2. Apply Reduction ONLY to Training Pool
        if self.ratio is not None:
            keep_count = max(1, int(len(train_pool) * float(self.ratio)))
        else:
            keep_count = min(self.target_entities, len(train_pool))

        logger.info(f"Applying Retention Ratio {self.ratio}: Keeping {keep_count} of {len(train_pool)} training pairs.")

        train_list = sorted(list(train_pool), key=lambda p: (str(p[0]), str(p[1])))
        rng = random.Random(self.seed)
        train_retained = set(rng.sample(train_list, keep_count))

        # 3. Store test pool in dataset for writer to use
        dataset.fixed_test_pairs = test_pool
        logger.info(f"Stored {len(test_pool)} test pairs in dataset.fixed_test_pairs for Writer.")

        # 4. Update Dataset - only training pairs in aligned_entities
        dataset.aligned_entities = train_retained

        logger.info(
            "Reduction complete.\n"
            f"  - Original Total Aligned: {total_original}\n"
            f"  - Train Pool: {len(train_pool)} -> Reduced to: {len(train_retained)}\n"
            f"  - Test Pool (Fixed): {len(test_pool)}"
        )
        logger.info("[SUCCESS] ForgetLabels reduction finished")

        return dataset
