"""Two-phase BERT-INT-A alignment model: bert_int + attribute-view features.

This is bert_int with Phase 2 attribute-view features properly implemented,
following the original BERT-INT paper. The Phase 1 BERT model is reused to
encode each unique attribute value string, producing per-value embeddings that
feed the DualAggregation attribute-view interaction features (42 dims).

Differences from bert_int:
  - Phase 2 attribute_features computed via AttributeViewFeatureExtractor
    (bert_int leaves them as zeros)
  - _train_basic_unit also returns the trained model so Phase 2 can reuse it
    for value embedding generation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdflib import Literal
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from src.alignment_models.methods.bert_int import (
    load_basic_unit_data,
    load_bert_int_config,
)
from src.alignment_models.methods.bert_int.basic_unit import BasicBertUnit, BasicUnitTrainer
from src.alignment_models.methods.bert_int.interaction_model import (
    AttributeViewFeatureExtractor,
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
    if not stage_config:
        return {}
    overrides = {}
    if "basic_unit" in stage_config:
        overrides["basic_unit"] = stage_config["basic_unit"]
    if "interaction_model" in stage_config:
        overrides["interaction_model"] = stage_config["interaction_model"]
    return overrides


@MODEL_REGISTRY.register("bert_intA")
class BertIntAAlignment:
    """Two-phase BERT-INT-A: basic_unit (phase 1) + full interaction_model (phase 2).

    Identical to bert_int except attribute-view features in Phase 2 are computed
    using the Phase 1 BERT model to encode individual attribute value strings.
    """

    def __init__(self, stage_config: Optional[Dict[str, Any]] = None) -> None:
        self.stage_config = stage_config or {}
        overrides = _extract_overrides(self.stage_config)
        self.config = load_bert_int_config(overrides=overrides)
        self.basic_cfg = self.config["basic_unit"]
        self.interaction_cfg = self.config["interaction_model"]

        lineage = self.stage_config.get("lineage", {})
        self.variant = lineage.get("variant", "reduced")
        artifact_root = Path(lineage.get("artifact_root", "results/artifact"))
        artifact_root.mkdir(parents=True, exist_ok=True)
        self.artifact_root = artifact_root
        self.evaluation_root = Path(
            lineage.get("evaluation_root", (artifact_root / "evaluation"))
        )

        self.checkpoint_dir = artifact_root / "bert_intA" / self.variant
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "[BERT-INTA] Initialized (encoder=%s, device=%s, "
            "basic_unit_epochs=%d, interaction_epochs=%d)",
            self.basic_cfg.get("encoder_name"),
            self.config.get("device"),
            self.basic_cfg.get("epochs"),
            self.interaction_cfg.get("epochs"),
        )
        logger.info("[BERT-INTA] Checkpoint directory: %s", self.checkpoint_dir)

    def evaluate(self, dataset_reduced, dataset_augmented):
        """Execute two-phase BERT-INT-A."""
        logger.info("=" * 80)
        logger.info("BERT-INT-A Phase 1: Basic Unit Training")
        logger.info("=" * 80)

        basic_unit_results, data_bundle, entity_embeddings, eid2data, basic_bert_model = (
            self._train_basic_unit(dataset_augmented)
        )

        logger.info("=" * 80)
        logger.info("BERT-INT-A Phase 2: Interaction Model Training (with attribute-view)")
        logger.info("=" * 80)

        interaction_results = self._train_interaction_model(
            dataset_augmented, data_bundle, entity_embeddings, eid2data, basic_bert_model
        )

        final_results = self._merge_results(basic_unit_results, interaction_results)

        logger.info("=" * 80)
        logger.info("BERT-INT-A Completed")
        logger.info(
            "  Basic Unit  - Hits@1: %.4f | Hits@10: %.4f",
            basic_unit_results.get("hits@1", 0),
            basic_unit_results.get("hits@10", 0),
        )
        logger.info(
            "  Interaction - Hits@1: %.4f | Hits@10: %.4f",
            interaction_results.get("hits@1", 0),
            interaction_results.get("hits@10", 0),
        )
        logger.info("=" * 80)

        return final_results

    def _train_basic_unit(self, dataset):
        """Train Phase 1 and return (metrics, data_bundle, entity_embeddings, eid2data, model)."""
        lineage = self.stage_config.get("lineage", {})
        dataset_root = lineage.get("dataset_workspace")

        if not dataset_root:
            raise ValueError(
                "dataset_workspace not found in lineage. "
                "Make sure to use writer: bert_int in experiment config"
            )

        logger.info("[BERT-INTA] Loading dataset from %s", dataset_root)
        experiment_meta = self.stage_config.get("experiment", {})
        dataset_name = experiment_meta.get("dataset", "")
        basic_cfg_with_dataset = dict(self.basic_cfg)
        if dataset_name:
            basic_cfg_with_dataset.setdefault("dataset", {})["name"] = dataset_name
            logger.info("[BERT-INTA] Using dataset name: %s", dataset_name)

        data_bundle = load_basic_unit_data(
            basic_cfg_with_dataset, {"dataset_root": dataset_root}
        )

        model = BasicBertUnit(self.basic_cfg)
        trainer = BasicUnitTrainer(
            model=model,
            config=self.basic_cfg,
            data=data_bundle,
            paths={"model_save_dir": str(self.checkpoint_dir), "model_save_prefix": "run"},
            device_spec=self.config.get("device"),
        )

        skip_training = bool(self.stage_config.get("skip_training"))
        history: Sequence[Dict[str, float]] = []
        if not skip_training and self.basic_cfg.get("epochs", 0) > 0:
            history = trainer.fit()
        else:
            logger.info(
                "[BERT-INTA] Skipping basic unit training (skip_training=%s, epochs=%d)",
                skip_training,
                self.basic_cfg.get("epochs", 0),
            )

        metrics = trainer.evaluate(
            self._evaluation_pairs(data_bundle),
            batch_size=self.basic_cfg.get("eval_batch_size"),
        )
        if history:
            metrics.update({"loss": history[-1]["loss"], "epochs_trained": len(history)})
        else:
            metrics.setdefault("epochs_trained", 0)

        logger.info(
            "[BERT-INTA Phase 1] hits@1=%.4f  hits@10=%.4f  mrr=%.4f",
            metrics.get("hits@1", 0.0),
            metrics.get("hits@10", 0.0),
            metrics.get("mrr", 0.0),
        )

        eid2data = data_bundle.ent2data
        entity_embeddings = self._generate_entity_embeddings(model, eid2data)

        return metrics, data_bundle, entity_embeddings, eid2data, model

    def _generate_entity_embeddings(self, model: torch.nn.Module, eid2data) -> np.ndarray:
        if not eid2data:
            return np.zeros((0, 0), dtype=np.float32)

        device = torch.device(
            self.interaction_cfg["device"] if torch.cuda.is_available() else "cpu"
        )
        model.eval()
        model = model.to(device)

        key_iterable = eid2data.keys() if hasattr(eid2data, "keys") else range(len(eid2data))
        ordered_ids = sorted(key_iterable)
        token_tensor = torch.tensor(
            [eid2data[eid][0] for eid in ordered_ids], dtype=torch.long
        )
        mask_tensor = torch.tensor(
            [eid2data[eid][1] for eid in ordered_ids], dtype=torch.float
        )

        dataset = TensorDataset(token_tensor, mask_tensor)
        batch_size = int(self.interaction_cfg.get("embedding_batch_size") or 256)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        embeddings: List[torch.Tensor] = []
        with torch.no_grad():
            for tokens, masks in loader:
                embeddings.append(model(tokens.to(device), masks.to(device)).cpu())

        entity_embeddings = torch.cat(embeddings, dim=0).numpy()
        logger.info(
            "[BERT-INTA] Entity embeddings: %s (batches=%d, batch_size=%d)",
            entity_embeddings.shape, len(loader), batch_size,
        )
        return entity_embeddings

    def _generate_value_embeddings(
        self,
        model: torch.nn.Module,
        value_list: List[str],
        device: torch.device,
        batch_size: int = 256,
        max_length: int = 64,
    ) -> np.ndarray:
        """Encode each unique attribute value string with the Phase 1 BERT model."""
        if not value_list:
            return np.zeros((0, 1), dtype=np.float32)

        tokenizer = AutoTokenizer.from_pretrained(
            self.basic_cfg.get("encoder_name", "bert-base-multilingual-cased")
        )
        model.eval()
        model = model.to(device)

        all_embs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(value_list), batch_size):
                batch = value_list[i: i + batch_size]
                enc = tokenizer(
                    batch,
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                tokens = enc["input_ids"].to(device)
                masks = enc["attention_mask"].float().to(device)
                emb = model(tokens, masks)
                all_embs.append(emb.cpu().numpy())

        result = np.vstack(all_embs)
        logger.info("[BERT-INTA] Value embeddings: %s", result.shape)
        return result

    def _train_interaction_model(
        self, dataset, data_bundle, entity_embeddings, eid2data, basic_bert_model
    ):
        """Train Phase 2 with full 85-feature vectors (neighbor + attribute + description)."""
        device = torch.device(
            self.interaction_cfg["device"] if torch.cuda.is_available() else "cpu"
        )
        logger.info("[BERT-INTA Phase 2] Using device: %s", device)

        # ── candidate generation ──────────────────────────────────────────────
        candidate_topk = self.interaction_cfg["candidate_topk"]
        candidate_gen = CandidateGenerator(topk=candidate_topk, device=device)

        train_ids_1 = [e1 for e1, e2 in data_bundle.train_ill]
        train_ids_2 = [e2 for e1, e2 in data_bundle.train_ill]
        test_ids_1 = [e1 for e1, e2 in data_bundle.test_ill]
        test_ids_2 = [e2 for e1, e2 in data_bundle.test_ill]

        train_candidates = candidate_gen.generate(train_ids_1, train_ids_2, entity_embeddings)
        test_candidates = candidate_gen.generate(test_ids_1, test_ids_2, entity_embeddings)

        entity_pairs = InteractionDataset.generate_all_entity_pairs(
            [train_candidates, test_candidates], [data_bundle.train_ill]
        )

        # ── neighbor-view features ────────────────────────────────────────────
        logger.info("[BERT-INTA Phase 2] Extracting neighbor-view features...")
        rel_triples = []
        for kg in [dataset.knowledge_graph_source, dataset.knowledge_graph_target]:
            for subj, pred, obj in kg:
                s, p, o = str(subj), str(pred), str(obj)
                if (
                    s in data_bundle.entity2index
                    and o in data_bundle.entity2index
                    and p in data_bundle.rel2index
                ):
                    rel_triples.append((
                        data_bundle.entity2index[s],
                        data_bundle.rel2index[p],
                        data_bundle.entity2index[o],
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

        # ── attribute-view features ───────────────────────────────────────────
        logger.info("[BERT-INTA Phase 2] Building attribute value index...")
        value_set: set = set()
        ent_to_raw_values: Dict[int, List[str]] = {}
        for kg in [dataset.knowledge_graph_source, dataset.knowledge_graph_target]:
            for subj, pred, obj in kg:
                subj_str = str(subj)
                if subj_str in data_bundle.entity2index and isinstance(obj, Literal):
                    eid = data_bundle.entity2index[subj_str]
                    val_str = str(obj)
                    value_set.add(val_str)
                    ent_to_raw_values.setdefault(eid, []).append(val_str)

        value_list = sorted(value_set)
        value2index = {v: i for i, v in enumerate(value_list)}
        logger.info("[BERT-INTA Phase 2] Unique attribute values: %d", len(value_list))

        logger.info("[BERT-INTA Phase 2] Generating value embeddings...")
        value_embeddings = self._generate_value_embeddings(
            basic_bert_model, value_list, device
        )

        max_att = int(self.interaction_cfg.get("entity_attvalue_max_num", 20))
        pad_value_id = len(value_embeddings)
        value_embeddings_padded = np.vstack(
            [value_embeddings, np.zeros((1, value_embeddings.shape[1]))]
        )

        n_entities = len(data_bundle.entity2index)
        ent2valueids: Dict[int, List[int]] = {}
        for eid in range(n_entities):
            vals = list(dict.fromkeys(
                value2index[v] for v in ent_to_raw_values.get(eid, [])
            ))[:max_att]
            vals += [pad_value_id] * (max_att - len(vals))
            ent2valueids[eid] = vals

        logger.info("[BERT-INTA Phase 2] Extracting attribute-view features...")
        att_extractor = AttributeViewFeatureExtractor(
            kernel_num=self.interaction_cfg["kernel_num"],
            max_values=max_att,
            device=device,
        )
        attribute_features = att_extractor.extract_features(
            entity_pairs, value_embeddings_padded, ent2valueids, pad_value_id, batch_size=512
        )

        # ── description-view features ─────────────────────────────────────────
        logger.info("[BERT-INTA Phase 2] Extracting description-view features...")
        description_extractor = DescriptionViewFeatureExtractor(device=device)
        description_features = description_extractor.extract_features(
            entity_pairs, entity_embeddings, batch_size=512
        )

        # ── assemble feature matrix ───────────────────────────────────────────
        all_features = np.concatenate(
            [neighbor_features, attribute_features, description_features], axis=1
        )
        logger.info("[BERT-INTA Phase 2] Feature shape: %s", all_features.shape)

        # ── interaction MLP training ──────────────────────────────────────────
        interaction_dataset = InteractionDataset(
            entity_pairs=entity_pairs,
            features=all_features,
            train_ill=data_bundle.train_ill,
            test_ill=data_bundle.test_ill,
            train_candidates=train_candidates,
            test_candidates=test_candidates,
        )

        mlp = InteractionMLP(
            input_dim=all_features.shape[1],
            hidden_dim=self.interaction_cfg["mlp_hidden_dim"],
        )

        seed = self.stage_config.get("seed") or self.stage_config.get("experiment", {}).get("seed")
        trainer = InteractionTrainer(
            model=mlp,
            dataset=interaction_dataset,
            device=device,
            learning_rate=self.interaction_cfg["learning_rate"],
            margin=self.interaction_cfg["margin"],
            neg_num=self.interaction_cfg["neg_num"],
            batch_size=self.interaction_cfg["batch_size"],
            seed=seed,
        )

        checkpoint_path = self.checkpoint_dir / "interaction_model.pt"
        training_results = trainer.train(
            epochs=self.interaction_cfg["epochs"],
            eval_every=self.interaction_cfg["eval_every"],
            save_path=checkpoint_path,
        )

        if checkpoint_path.exists():
            trainer.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info(
                "[BERT-INTA] Loaded best model (best epoch: %d)",
                training_results["best_epoch"],
            )

        final_results = trainer.evaluator.evaluate(topk=candidate_topk)

        score_dist = trainer.evaluator.get_score_distributions()
        score_dist_path = self.checkpoint_dir / "score_distributions.json"
        with open(score_dist_path, "w") as _f:
            json.dump(score_dist, _f)

        final_results["best_epoch"] = training_results["best_epoch"]
        final_results["best_hits@1"] = training_results["best_hits@1"]

        logger.info(
            "[BERT-INTA Phase 2] hits@1=%.4f  hits@10=%.4f  mrr=%.4f  (best epoch: %d)",
            final_results.get("hits@1", 0),
            final_results.get("hits@10", 0),
            final_results.get("mrr", 0),
            training_results["best_epoch"],
        )
        return final_results

    def _merge_results(
        self, basic_unit_results: Dict[str, Any], interaction_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "model": "bert_intA",
            "phases": {
                "basic_unit": basic_unit_results,
                "interaction_model": interaction_results,
            },
            "hits@1": interaction_results.get("hits@1", 0.0),
            "hits@5": interaction_results.get("hits@5", 0.0),
            "hits@10": interaction_results.get("hits@10", 0.0),
            "hits@25": interaction_results.get("hits@25", 0.0),
            "hits@50": interaction_results.get("hits@50", 0.0),
            "mr": interaction_results.get("mr", 0.0),
            "mrr": interaction_results.get("mrr", 0.0),
            "precision": interaction_results.get("precision", 0.0),
            "recall": interaction_results.get("recall", 0.0),
            "f-measure": interaction_results.get("f-measure", 0.0),
            "evaluated": interaction_results.get("total", 0),
            "_note": "BERT-INT-A: Phase 2 uses full 85-feature vectors (neighbor + attribute + description).",
        }

    @staticmethod
    def _evaluation_pairs(data_bundle) -> Sequence[Pair]:
        if data_bundle.test_ill:
            return data_bundle.test_ill
        logger.warning("[BERT-INTA] No test pairs available; falling back to training pairs.")
        return data_bundle.train_ill or data_bundle.ent_ill


__all__ = ("BertIntAAlignment",)
