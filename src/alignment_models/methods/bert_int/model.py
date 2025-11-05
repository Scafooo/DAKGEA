"""Two-phase BERT-INT alignment model (basic_unit + interaction_model)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from src.alignment_models.methods.bert_int import (
    load_basic_unit_data,
    load_bert_int_config,
)
from src.alignment_models.methods.bert_int.basic_unit import BasicBertUnit, BasicUnitTrainer
from src.alignment_models.methods.bert_int.interaction_model import (
    CandidateGenerator,
    DescriptionViewFeatureExtractor,
    InteractionDataset,
    InteractionMLP,
    InteractionTrainer,
    NeighborViewFeatureExtractor,
)
from src.alignment_models.registry import MODEL_REGISTRY
from src.logger import get_logger

logger = get_logger(__name__)

Pair = Tuple[int, int]


def _extract_overrides(stage_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract configuration overrides from stage_config.

    This function extracts only essential configuration that cannot be
    determined automatically by the model.
    """
    if not stage_config:
        return {}

    overrides = {}

    # Extract experiment-level overrides if present (from YAML)
    # These are optional overrides that the user wants to apply
    if "basic_unit" in stage_config:
        overrides["basic_unit"] = stage_config["basic_unit"]
    if "interaction_model" in stage_config:
        overrides["interaction_model"] = stage_config["interaction_model"]

    return overrides


@MODEL_REGISTRY.register("bert_int")
class BertIntAlignment:
    """Two-phase BERT-INT model: basic_unit (phase 1) + interaction_model (phase 2)."""

    def __init__(self, stage_config: Optional[Dict[str, Any]] = None) -> None:
        self.stage_config = stage_config or {}
        overrides = _extract_overrides(self.stage_config)
        self.config = load_bert_int_config(overrides=overrides)
        self.basic_cfg = self.config["basic_unit"]
        self.interaction_cfg = self.config["interaction_model"]

        # Determine paths from lineage
        lineage = self.stage_config.get("lineage", {})
        self.evaluation_root = Path(lineage.get("evaluation_root", "results/evaluation"))
        self.variant = lineage.get("variant", "reduced")

        # Create checkpoint directory for this variant
        self.checkpoint_dir = self.evaluation_root / "bert_int" / self.variant
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "[BERT-INT] Initialized two-phase model (encoder=%s, device=%s, "
            "basic_unit_epochs=%d, interaction_epochs=%d)",
            self.basic_cfg.get("encoder_name"),
            self.config.get("device"),
            self.basic_cfg.get("epochs"),
            self.interaction_cfg.get("epochs"),
        )
        logger.info(f"[BERT-INT] Checkpoint directory: {self.checkpoint_dir}")

    def evaluate(self, dataset_reduced, dataset_augmented):
        """Execute two-phase BERT-INT: basic_unit training → interaction_model training."""
        logger.info("=" * 80)
        logger.info("BERT-INT Phase 1: Basic Unit Training")
        logger.info("=" * 80)

        # Phase 1: Basic Unit
        basic_unit_results, data_bundle, entity_embeddings, eid2data = self._train_basic_unit(
            dataset_augmented
        )

        logger.info("=" * 80)
        logger.info("BERT-INT Phase 2: Interaction Model Training")
        logger.info("=" * 80)

        # Phase 2: Interaction Model
        interaction_results = self._train_interaction_model(
            dataset_augmented, data_bundle, entity_embeddings, eid2data
        )

        # Merge results from both phases
        final_results = self._merge_results(basic_unit_results, interaction_results)

        logger.info("=" * 80)
        logger.info("BERT-INT Completed: Two-Phase Results")
        logger.info(
            f"  Basic Unit  - Hits@1: {basic_unit_results.get('hits@1', 0):.2f}% | "
            f"Hits@10: {basic_unit_results.get('hits@10', 0):.2f}%"
        )
        logger.info(
            f"  Interaction - Hits@1: {interaction_results.get('hits@1', 0):.2f}% | "
            f"Hits@10: {interaction_results.get('hits@10', 0):.2f}%"
        )
        logger.info("=" * 80)

        return final_results

    def _train_basic_unit(self, dataset):
        """Train basic unit (phase 1) and return results + artifacts."""
        # Get dataset path from lineage (written by bert_int writer)
        lineage = self.stage_config.get("lineage", {})
        dataset_root = lineage.get("dataset_workspace")

        if not dataset_root:
            raise ValueError(
                "dataset_workspace not found in lineage. "
                "Make sure to use writer: bert_int in experiment config"
            )

        logger.info(f"[BERT-INT] Loading dataset from {dataset_root}")
        data_bundle = load_basic_unit_data(
            self.basic_cfg, {"dataset_root": dataset_root}
        )

        model = BasicBertUnit(self.basic_cfg)
        trainer = BasicUnitTrainer(
            model=model,
            config=self.basic_cfg,
            data=data_bundle,
            paths={"model_save_dir": str(self.checkpoint_dir), "model_save_prefix": "run"},
            device_spec=self.config.get("device"),
        )

        # Train
        skip_training = bool(self.stage_config.get("skip_training"))
        history: Sequence[Dict[str, float]] = []
        if not skip_training and self.basic_cfg.get("epochs", 0) > 0:
            history = trainer.fit()
        else:
            logger.info(
                "[BERT-INT] Skipping basic unit training (skip_training=%s, epochs=%d)",
                skip_training,
                self.basic_cfg.get("epochs", 0),
            )

        # Evaluate basic unit
        metrics = trainer.evaluate(
            self._evaluation_pairs(data_bundle),
            batch_size=self.basic_cfg.get("eval_batch_size"),
        )
        if history:
            metrics.update({"loss": history[-1]["loss"], "epochs_trained": len(history)})
        else:
            metrics.setdefault("epochs_trained", 0)

        logger.info(
            "[BERT-INT Phase 1] Basic unit completed: "
            f"hits@1={metrics.get('hits@1', 0.0):.2f}% "
            f"hits@10={metrics.get('hits@10', 0.0):.2f}% "
            f"mrr={metrics.get('mrr', 0.0):.4f}"
        )

        # Generate entity embeddings for phase 2
        device = torch.device(
            self.interaction_cfg["device"] if torch.cuda.is_available() else "cpu"
        )
        model.eval()
        model = model.to(device)
        entity_embeddings = []
        eid2data = data_bundle.ent2data

        for eid in range(len(eid2data)):
            token_input = torch.LongTensor([eid2data[eid][0]]).to(device)
            mask_input = torch.FloatTensor([eid2data[eid][1]]).to(device)
            with torch.no_grad():
                vec = model(token_input, mask_input)
            entity_embeddings.append(vec.cpu().numpy()[0])

        entity_embeddings = np.array(entity_embeddings)
        logger.info(f"[BERT-INT] Generated entity embeddings: {entity_embeddings.shape}")

        return metrics, data_bundle, entity_embeddings, eid2data

    def _train_interaction_model(self, dataset, data_bundle, entity_embeddings, eid2data):
        """Train interaction model (phase 2) and return results."""
        device = torch.device(
            self.interaction_cfg["device"] if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"[BERT-INT Phase 2] Using device: {device}")

        # Generate candidates
        candidate_topk = self.interaction_cfg["candidate_topk"]
        candidate_gen = CandidateGenerator(topk=candidate_topk, device=device)

        train_ids_1 = [e1 for e1, e2 in data_bundle.train_ill]
        train_ids_2 = [e2 for e1, e2 in data_bundle.train_ill]
        test_ids_1 = [e1 for e1, e2 in data_bundle.test_ill]
        test_ids_2 = [e2 for e1, e2 in data_bundle.test_ill]

        train_candidates = candidate_gen.generate(train_ids_1, train_ids_2, entity_embeddings)
        test_candidates = candidate_gen.generate(test_ids_1, test_ids_2, entity_embeddings)

        # Generate entity pairs
        entity_pairs = InteractionDataset.generate_all_entity_pairs(
            [train_candidates, test_candidates], [data_bundle.train_ill]
        )

        # Extract features
        logger.info("[BERT-INT Phase 2] Extracting neighbor-view features...")
        # Convert triples from URIs to indices using data_bundle mappings
        rel_triples = []
        for subj, pred, obj in dataset.knowledge_graph_source:
            subj_str = str(subj)
            pred_str = str(pred)
            obj_str = str(obj)
            # Only include triples where both subject and object are in our entity set
            if subj_str in data_bundle.entity2index and obj_str in data_bundle.entity2index:
                if pred_str in data_bundle.rel2index:
                    rel_triples.append((
                        data_bundle.entity2index[subj_str],
                        data_bundle.rel2index[pred_str],
                        data_bundle.entity2index[obj_str]
                    ))
        for subj, pred, obj in dataset.knowledge_graph_target:
            subj_str = str(subj)
            pred_str = str(pred)
            obj_str = str(obj)
            # Only include triples where both subject and object are in our entity set
            if subj_str in data_bundle.entity2index and obj_str in data_bundle.entity2index:
                if pred_str in data_bundle.rel2index:
                    rel_triples.append((
                        data_bundle.entity2index[subj_str],
                        data_bundle.rel2index[pred_str],
                        data_bundle.entity2index[obj_str]
                    ))

        pad_entity_id = len(entity_embeddings)
        entity_embeddings_padded = np.vstack(
            [entity_embeddings, np.zeros((1, entity_embeddings.shape[1]))]
        )

        neighbor_extractor = NeighborViewFeatureExtractor(
            kernel_num=self.interaction_cfg["kernel_num"],
            max_neighbors=self.interaction_cfg["entity_neigh_max_num"],
            device=device,
        )
        neighbor_dict = neighbor_extractor.build_neighbor_dict(rel_triples, pad_entity_id)
        neighbor_features = neighbor_extractor.extract_features(
            entity_pairs, entity_embeddings_padded, neighbor_dict, pad_entity_id, batch_size=2048
        )

        logger.info("[BERT-INT Phase 2] Extracting description-view features...")
        description_extractor = DescriptionViewFeatureExtractor(device=device)
        description_features = description_extractor.extract_features(
            entity_pairs, entity_embeddings, batch_size=512
        )

        # Placeholder for attribute features (TODO: implement if needed)
        attribute_features = np.zeros((len(entity_pairs), self.interaction_cfg["kernel_num"] * 2))

        # Concatenate features
        all_features = np.concatenate(
            [neighbor_features, attribute_features, description_features], axis=1
        )
        logger.info(f"[BERT-INT Phase 2] Feature shape: {all_features.shape}")

        # Create dataset and train
        interaction_dataset = InteractionDataset(
            entity_pairs=entity_pairs,
            features=all_features,
            train_ill=data_bundle.train_ill,
            test_ill=data_bundle.test_ill,
            train_candidates=train_candidates,
            test_candidates=test_candidates,
        )

        model = InteractionMLP(
            input_dim=all_features.shape[1], hidden_dim=self.interaction_cfg["mlp_hidden_dim"]
        )

        seed = self.stage_config.get("seed") or self.stage_config.get("experiment", {}).get("seed")
        trainer = InteractionTrainer(
            model=model,
            dataset=interaction_dataset,
            device=device,
            learning_rate=self.interaction_cfg["learning_rate"],
            margin=self.interaction_cfg["margin"],
            neg_num=self.interaction_cfg["neg_num"],
            batch_size=self.interaction_cfg["batch_size"],
            seed=seed,
        )

        # Train
        checkpoint_path = self.checkpoint_dir / "interaction_model.pt"
        training_results = trainer.train(
            epochs=self.interaction_cfg["epochs"],
            eval_every=self.interaction_cfg["eval_every"],
            save_path=checkpoint_path,
        )

        # Final evaluation
        final_results = trainer.evaluator.evaluate(topk=candidate_topk)

        logger.info(
            f"[BERT-INT Phase 2] Interaction model completed: "
            f"hits@1={final_results.get('hits@1', 0):.2f}% "
            f"hits@10={final_results.get('hits@10', 0):.2f}% "
            f"mrr={final_results.get('mrr', 0):.4f}"
        )

        return final_results

    def _merge_results(
        self, basic_unit_results: Dict[str, Any], interaction_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge results from both phases into final output."""
        final_results = {
            "model": "bert_int",
            "phases": {
                "basic_unit": basic_unit_results,
                "interaction_model": interaction_results,
            },
            # Top-level metrics from interaction model (final results)
            "hits@1": interaction_results.get("hits@1", 0.0),
            "hits@5": interaction_results.get("hits@5", 0.0),
            "hits@10": interaction_results.get("hits@10", 0.0),
            "hits@25": interaction_results.get("hits@25", 0.0),
            "hits@50": interaction_results.get("hits@50", 0.0),
            "mr": interaction_results.get("mr", 0.0),
            "mrr": interaction_results.get("mrr", 0.0),
            "evaluated": interaction_results.get("total", 0),
            "_note": "BERT-INT is a two-phase model. Top-level metrics are from phase 2 (interaction_model).",
        }
        return final_results

    @staticmethod
    def _evaluation_pairs(data_bundle) -> Sequence[Pair]:
        """Return evaluation alignment pairs."""
        if data_bundle.test_ill:
            return data_bundle.test_ill
        logger.warning("[BERT-INT] No test pairs available; falling back to training pairs.")
        return data_bundle.train_ill or data_bundle.ent_ill


__all__ = ("BertIntAlignment",)
