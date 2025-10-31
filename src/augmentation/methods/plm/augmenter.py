"""PLM-based augmentation strategy leveraging latent interpolation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import torch
from rdflib import Literal, URIRef
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.augmentation.base import AugmentationMethod
from src.augmentation.methods.plm.interpolator import BartInterpolatorPLM, _clean_pred
from src.augmentation.methods.plm.predicate_matching import match_relations
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.core.dataset import Dataset
from src.core.knowledge_graph import KnowledgeGraph


@AUGMENTATION_REGISTRY.register("plm_augmentation")
class PLMAugmenter(AugmentationMethod):
    """Augment datasets via BART-based latent interpolation of literal attributes."""

    registry_name = "plm_augmentation"

    def __init__(self, config):
        super().__init__(config, name=self.registry_name)

        augmentation_cfg = dict(self.config.get("augmentation", {}))
        if self.registry_name in augmentation_cfg:
            augmentation_cfg = dict(augmentation_cfg[self.registry_name])
        self.augmentation_cfg = augmentation_cfg

        experiment_cfg = self.config.get("experiment", {})
        seed = augmentation_cfg.get("seed", experiment_cfg.get("seed", 42))
        device_cfg = augmentation_cfg.get("device") or augmentation_cfg.get("accelerator")
        if device_cfg is None:
            device_cfg = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device_cfg)
        self.random = random.Random(seed)

        self.aug_percentage = float(augmentation_cfg.get("aug_percentage", 0.2))
        self.max_depth = int(augmentation_cfg.get("max_depth", 1))
        self.pending_link_policy = augmentation_cfg.get("pending_link_policy", "connect_parent")

        interpolator_cfg = dict(augmentation_cfg.get("interpolator", {}))
        reuse_flag = interpolator_cfg.get("reuse_if_available", augmentation_cfg.get("reuse_if_available", True))
        self.interpolator_settings = {
            "model_name": interpolator_cfg.get("model_name", "facebook/bart-base"),
            "base_alpha": interpolator_cfg.get("base_alpha", 0.35),
            "alpha_spread": interpolator_cfg.get("alpha_spread", 0.25),
            "max_len_in": int(interpolator_cfg.get("max_len_in", 96)),
            "max_len_out": int(interpolator_cfg.get("max_len_out", 48)),
            "seed": seed,
            "reuse_if_available": reuse_flag,
        }
        self.output_dir_override = interpolator_cfg.get("output_dir") or augmentation_cfg.get("output_dir")

        fine_tune_cfg = dict(augmentation_cfg.get("fine_tune", {}))
        self.ft_epochs = int(fine_tune_cfg.get("epochs", 10))
        self.ft_batch_size = int(fine_tune_cfg.get("batch_size", 16))
        self.ft_lr = float(fine_tune_cfg.get("lr", 5e-5))
        self.ft_force = bool(fine_tune_cfg.get("force_retrain", False))
        self.ft_max_samples = fine_tune_cfg.get("max_train_samples", 4000)
        self.ft_val_split = float(fine_tune_cfg.get("val_split", 0.1))
        self.ft_workers = int(fine_tune_cfg.get("num_proc", 2))
        self.ft_patience = int(fine_tune_cfg.get("patience", 3))

        self.logger.debug(
            "PLMAugmenter configured (ratio=%.3f, depth=%d, device=%s)",
            self.aug_percentage,
            self.max_depth,
            self.device,
        )

    def augment(self, dataset: Dataset) -> Dataset:
        aligned_entities = list(dataset.aligned_entities)
        if not aligned_entities:
            self.logger.warning("Dataset contains no aligned entities; returning unchanged dataset.")
            return dataset

        dataset_name = self.config.get("experiment", {}).get("dataset", "dataset") or "dataset"
        artifacts_dir = self._resolve_artifact_dir(dataset_name)

        interpolator = BartInterpolatorPLM(
            out_dir=str(artifacts_dir),
            device=str(self.device),
            **self.interpolator_settings,
        )

        self.section(f"PLM augmentation – dataset '{dataset_name}'")
        self.logger.info(
            "Starting PLM augmentation over %d aligned entity pairs (ratio=%.2f, max_depth=%d).",
            len(aligned_entities),
            self.aug_percentage,
            self.max_depth,
        )

        pairs = interpolator.build_pairs_from_dataset(
            dataset.knowledge_graph_source,
            dataset.knowledge_graph_target,
            aligned_entities,
        )
        if not pairs:
            self.logger.warning("Unable to build literal training pairs; returning unchanged dataset.")
            return dataset

        self.subsection("Fine-tuning interpolator")
        interpolator.fine_tune(
            pairs,
            epochs=self.ft_epochs,
            batch_size=self.ft_batch_size,
            lr=self.ft_lr,
            max_train_samples=self.ft_max_samples,
            val_split=self.ft_val_split,
            force_retrain=self.ft_force,
            num_proc=self.ft_workers,
            patience=self.ft_patience,
        )

        sample_size = min(int(len(aligned_entities) * self.aug_percentage), len(aligned_entities))
        if sample_size <= 0:
            self.logger.info(
                "Augmentation percentage produced zero candidates (size=%d); returning unchanged dataset.",
                sample_size,
            )
            return dataset

        candidates = self.random.sample(aligned_entities, sample_size)
        matches = match_relations(dataset)
        match_lnames = {(_clean_pred(src), _clean_pred(tgt)) for (src, tgt), _ in matches}

        new_src = set(dataset.knowledge_graph_source)
        new_tgt = set(dataset.knowledge_graph_target)
        new_align = set(dataset.aligned_entities)

        counter = 0
        visited_pairs = set()
        entity_map = {}
        pending_links = []

        def _get_entity_info(entity: URIRef, kg):
            triples = set()
            for triple in kg.triples((entity, None, None)):
                triples.add(triple)
            for triple in kg.triples((None, None, entity)):
                triples.add(triple)
            return triples

        def _build_kg(triples):
            kg = KnowledgeGraph()
            for triple in triples:
                kg.add(triple)
            return kg

        def _augment_entity_pair(e1: URIRef, e2: URIRef, depth: int = 0):
            nonlocal counter
            if depth > self.max_depth:
                return
            if (e1, e2) in visited_pairs:
                return
            visited_pairs.add((e1, e2))

            self.logger.debug("[Depth %d] Exploring (%s, %s)", depth, e1, e2)

            info1 = _get_entity_info(e1, dataset.knowledge_graph_source)
            info2 = _get_entity_info(e2, dataset.knowledge_graph_target)

            counter += 1
            new_uri_src = URIRef(f"{e1}_aug_{counter}")
            new_uri_tgt = URIRef(f"{e2}_aug_{counter}")
            entity_map[e1] = new_uri_src
            entity_map[e2] = new_uri_tgt

            new_triples_src = []
            new_triples_tgt = []
            next_pairs: list[Tuple[URIRef, URIRef]] = []
            local_seen_pairs = set()

            pred_map1 = {_clean_pred(p1): (p1, o1) for s1, p1, o1 in info1}
            pred_map2 = {_clean_pred(p2): (p2, o2) for s2, p2, o2 in info2}

            for lname1, lname2 in match_lnames:
                if lname1 not in pred_map1 or lname2 not in pred_map2:
                    continue

                p1, o1 = pred_map1[lname1]
                p2, o2 = pred_map2[lname2]

                if isinstance(o1, Literal) and isinstance(o2, Literal):
                    self.logger.debug("[Depth %d] Interpolating literal via %s", depth, p1)
                    aug_src, aug_tgt = interpolator.interpolate_pair(
                        str(o1),
                        str(o2),
                        max_new_tokens=32,
                        predicate=str(p1),
                    )
                    new_triples_src.append((new_uri_src, p1, Literal(aug_src)))
                    new_triples_tgt.append((new_uri_tgt, p2, Literal(aug_tgt)))
                elif isinstance(o1, URIRef) and isinstance(o2, URIRef):
                    if (o1, o2) not in dataset.aligned_entities:
                        self.logger.debug("[Depth %d] Skipping relation (%s, %s) not in alignment.", depth, o1, o2)
                        continue

                    if (o1, o2) not in visited_pairs and (o1, o2) not in local_seen_pairs:
                        local_seen_pairs.add((o1, o2))
                        next_pairs.append((o1, o2))

                    if o1 in entity_map and o2 in entity_map:
                        new_triples_src.append((new_uri_src, p1, entity_map[o1]))
                        new_triples_tgt.append((new_uri_tgt, p2, entity_map[o2]))
                    else:
                        pending_links.append((new_uri_src, p1, o1, new_uri_tgt, p2, o2))

            if new_triples_src and new_triples_tgt:
                new_src.update(new_triples_src)
                new_tgt.update(new_triples_tgt)
                new_align.add((new_uri_src, new_uri_tgt))
                self.logger.debug(
                    "[Depth %d] Generated augmented pair %s ↔ %s (%d attributes).",
                    depth,
                    new_uri_src,
                    new_uri_tgt,
                    len(new_triples_src),
                )

            for child1, child2 in next_pairs:
                _augment_entity_pair(child1, child2, depth + 1)

            for link in list(pending_links):
                parent_src, p1, o1, parent_tgt, p2, o2 = link
                if o1 == e1 and o2 == e2:
                    new_triples_src.append((parent_src, p1, new_uri_src))
                    new_triples_tgt.append((parent_tgt, p2, new_uri_tgt))
                    pending_links.remove(link)

        with logging_redirect_tqdm():
            for e1, e2 in tqdm(
                candidates,
                desc=f"BART latent interpolation (max_depth={self.max_depth})",
            ):
                _augment_entity_pair(e1, e2, depth=0)

        self.logger.info("Generated %d augmented entity pairs.", counter)
        self.logger.info("[SUCCESS] PLM augmentation completed for dataset '%s'", dataset_name)

        augmented_dataset = Dataset(
            knowledge_graph_source=_build_kg(new_src),
            knowledge_graph_target=_build_kg(new_tgt),
            aligned_entities=list(new_align),
        )
        return augmented_dataset

    def _resolve_artifact_dir(self, dataset_name: str) -> Path:
        safe_name = Path(dataset_name).name or "dataset"
        if self.output_dir_override:
            subdir = Path(self.output_dir_override).name
            return self.resolve_external_dir("plm", subdir, safe_name)
        return self.resolve_external_dir("plm", safe_name)
