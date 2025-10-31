from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import random
import torch
import torch.nn.functional as F

from src.alignment_models.methods.bert_int.basic_training import BasicUnitArtifacts, train_basic_unit_model
from src.alignment_models.methods.bert_int.basic_unit_model import BasicBertUnitModel
from src.alignment_models.methods.bert_int.config import BertIntConfig
from src.alignment_models.methods.bert_int.data import BertIntDataset, build_dataset
from src.alignment_models.methods.bert_int.features import (
    attribute_features,
    attribute_value_embeddings,
    build_attribute_values,
    build_neighbor_dict,
    clean_attribute_triples,
    description_features,
    neighbor_features,
)
from src.alignment_models.methods.bert_int.interaction_training import train_interaction_model
from src.alignment_models.methods.bert_int.metrics import evaluate_alignment
from src.alignment_models.methods.bert_int.text import build_graph_entity_texts, friendly_name
from src.alignment_models.methods.bert_int.tokenization import encode_entities, normalise_uri
from src.alignment_models.registry import MODEL_REGISTRY
from src.config.loader import PROJECT_ROOT, load_yaml
from src.logger import get_logger

logger = get_logger(__name__)


@MODEL_REGISTRY.register("bert_int")
class Bert_int:
    """Integrated BERT-INT alignment model."""

    MODEL_CONFIG_PATH = PROJECT_ROOT / "config/models/bert_int.yaml"

    def __init__(self, config):
        self.stage_config = config or {}
        self.model_config = self._load_model_config()
        self._description_cache: Optional[Dict[str, str]] = None
        logger.debug("[BERT-INT] Loaded configuration: %s", self.model_config.to_dict())

    def evaluate(self, dataset_reduced, dataset_augmented):
        logger.info(
            "[BERT-INT] Evaluating dataset '%s'",
            self.stage_config.get("experiment", {}).get("dataset"),
        )
        logger.info("[STEP] BERT-INT evaluation starting")

        dataset = dataset_augmented or dataset_reduced
        aligned_pairs = list(self._normalise_pairs(dataset.aligned_entities))
        if len(aligned_pairs) < 2:
            logger.warning("[BERT-INT] Not enough aligned entities (%d)", len(aligned_pairs))
            return self._empty_metrics()

        self._seed_everything()

        lineage = self.stage_config.get("lineage")
        dataset_name = self.stage_config.get("experiment", {}).get("dataset", "")

        bert_dataset = build_dataset(
            dataset,
            self._train_ratio(),
            lineage=lineage,
            dataset_name=dataset_name,
        )
        entity_order = [bert_dataset.index2entity[idx] for idx in range(len(bert_dataset.index2entity))]
        logger.info(
            "[BERT-INT] Dataset prepared: |KG1|=%d entities, |KG2|=%d entities, train_pairs=%d, test_pairs=%d",
            len(bert_dataset.kg1.entities),
            len(bert_dataset.kg2.entities),
            len(bert_dataset.train_pairs),
            len(bert_dataset.test_pairs),
        )
        logger.info("[STEP] Tokenising entities")

        entity_texts = self._build_entity_texts(dataset, bert_dataset)
        token_ids, attention_masks, tokenizer = encode_entities(
            self.model_config.basic_unit.encoder_name,
            entity_texts,
            entity_order,
            max_length=self.model_config.basic_unit.max_seq_length,
            cache_dir=self.model_config.paths.cache_dir,
        )

        entid2data = {
            idx: (
                token_ids[idx].long(),
                attention_masks[idx].float(),
            )
            for idx in range(token_ids.size(0))
        }

        last_error: Optional[torch.cuda.OutOfMemoryError] = None
        for device in self._candidate_devices():
            try:
                self._seed_everything()
                logger.info("[STEP] Evaluating on device %s", device)
                return self._evaluate_on_device(dataset, bert_dataset, entid2data, tokenizer, device)
            except torch.cuda.OutOfMemoryError as err:
                last_error = err
                if device.type == "cuda":
                    logger.warning(
                        "[BERT-INT] CUDA out of memory on device '%s'. Error: %s. Retrying on CPU.",
                        device,
                        err,
                    )
                    torch.cuda.empty_cache()
                    continue
                raise

        if last_error is not None:
            raise last_error
        return self._empty_metrics()

    def _evaluate_on_device(
        self,
        dataset,
        bert_dataset: BertIntDataset,
        entid2data: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        tokenizer,
        device: torch.device,
    ) -> Dict[str, float]:
        logger.info("[BERT-INT] Using device '%s' for evaluation.", device)
        logger.info("[STEP] Training basic unit")

        model = BasicBertUnitModel(
            self.model_config.basic_unit.encoder_name,
            self.model_config.basic_unit.input_dim,
            self.model_config.basic_unit.output_dim,
            load_strategy=self.model_config.basic_unit.encoder_strategy,
            cache_dir=self.model_config.paths.cache_dir,
        ).to(device)

        artifacts = train_basic_unit_model(
            model,
            entid2data,
            bert_dataset.train_pairs,
            bert_dataset.test_pairs,
            bert_dataset.ent_ids_1,
            bert_dataset.ent_ids_2,
            epochs=self.model_config.basic_unit.epochs,
            batch_size=self.model_config.basic_unit.batch_size,
            learning_rate=self.model_config.basic_unit.learning_rate,
            margin=self.model_config.basic_unit.margin,
            negatives=self.model_config.basic_unit.negatives,
            candidate_topk=self.model_config.basic_unit.candidate_topk,
            eval_topk=self.model_config.interaction.candidate_topk,
            device=device,
            embedding_batch_size=self.model_config.basic_unit.test_batch_size,
        )
        logger.info("[BERT-INT] Basic unit training complete; starting interaction phase.")
        logger.info("[STEP] Running interaction stage")

        scored_predictions = self._run_interaction_stage(
            model,
            tokenizer,
            bert_dataset,
            artifacts,
            device,
        )
        logger.info("[BERT-INT] Interaction phase complete; evaluating metrics.")

        truth_pairs = [
            (bert_dataset.index2entity[src], bert_dataset.index2entity[tgt])
            for src, tgt in bert_dataset.test_pairs
        ]
        metrics = evaluate_alignment(scored_predictions, truth_pairs)
        result = metrics.to_dict()
        logger.info(
            "[BERT-INT] Metrics: precision=%.4f recall=%.4f f1=%.4f hits@1=%.4f hits@10=%.4f mrr=%.4f",
            result["precision"],
            result["recall"],
            result["f1"],
            result["hits@1"],
            result["hits@10"],
            result["mrr"],
        )
        return result

    def _candidate_devices(self) -> List[torch.device]:
        primary = self._resolve_device()
        devices = [primary]
        if primary.type == "cuda":
            devices.append(torch.device("cpu"))
        return devices

    def _description_overrides(self) -> Dict[str, str]:
        if self._description_cache is not None:
            return self._description_cache

        overrides: Dict[str, str] = {}
        for path in self._description_paths():
            data = self._load_description_dict(path)
            if not data:
                continue
            applied = sum(1 for key in data if key not in overrides)
            for key, value in data.items():
                overrides.setdefault(key, value)
            logger.info(
                "[BERT-INT] Loaded %d descriptions from '%s' (%d applied, %d total).",
                len(data),
                path,
                applied,
                len(overrides),
            )
        self._description_cache = overrides
        return overrides

    def _description_paths(self) -> List[Path]:
        configured: List[Path] = []
        for path_str in (
            self.model_config.paths.description_dict,
            self.model_config.paths.origin_description_dict,
        ):
            if not path_str:
                continue
            path = Path(path_str)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            configured.append(path)
        return configured

    def _load_description_dict(self, path: Path) -> Dict[str, str]:
        if not path.exists():
            logger.warning("[BERT-INT] Description dictionary not found: %s", path)
            return {}
        try:
            with path.open("rb") as handle:
                raw = pickle.load(handle)
        except Exception as exc:  # pragma: no cover - depends on external files
            logger.warning("[BERT-INT] Failed to load description dictionary '%s' (%s).", path, exc)
            return {}

        if not isinstance(raw, dict):
            logger.warning("[BERT-INT] Description dictionary '%s' is not a dict (type=%s).", path, type(raw))
            return {}

        cleaned: Dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8")
                except UnicodeDecodeError:
                    value = value.decode("utf-8", errors="ignore")
            cleaned[key] = str(value)
        return cleaned

    def _seed_everything(self) -> None:
        random.seed(self.model_config.seed)
        np.random.seed(self.model_config.seed)
        torch.manual_seed(self.model_config.seed)
        if torch.cuda.is_available():  # pragma: no cover - depends on hardware
            torch.cuda.manual_seed_all(self.model_config.seed)

    def _run_interaction_stage(
        self,
        model: BasicBertUnitModel,
        tokenizer,
        dataset: BertIntDataset,
        artifacts: BasicUnitArtifacts,
        device: torch.device,
    ) -> List[Tuple[str, str, float]]:
        entity_embeddings = torch.tensor(artifacts.entity_embeddings, dtype=torch.float32, device=device)
        entity_embeddings = F.normalize(entity_embeddings, p=2, dim=1)
        pad_entity_emb = torch.zeros(1, entity_embeddings.size(1), device=device)
        entity_embeddings = torch.cat([entity_embeddings, pad_entity_emb], dim=0)
        pad_entity_id = entity_embeddings.size(0) - 1

        all_rel_triples = dataset.kg1.relation_triples + dataset.kg2.relation_triples
        all_entities = list(range(len(dataset.index2entity)))
        neighbor_dict = build_neighbor_dict(
            all_rel_triples,
            all_entities,
            self.model_config.interaction.neighbor_max,
            pad_entity_id,
        )

        neighbor_feats = neighbor_features(
            artifacts.entity_pairs,
            entity_embeddings,
            neighbor_dict,
            pad_entity_id,
            self.model_config.interaction.kernel_num,
            device,
            batch_size=2048,
        )
        logger.debug("[BERT-INT] Neighbor features ready for %d pairs.", len(neighbor_feats))

        fallback_names = {
            idx: normalise_uri(dataset.index2entity[idx]) for idx in all_entities
        }
        all_attribute_triples = dataset.kg1.attribute_triples + dataset.kg2.attribute_triples
        before_clean = len(all_attribute_triples)
        all_attribute_triples = clean_attribute_triples(all_attribute_triples, threshold=3)
        logger.debug(
            "[BERT-INT] Attribute triples cleaned: %d -> %d",
            before_clean,
            len(all_attribute_triples),
        )
        ent2values = build_attribute_values(
            all_attribute_triples,
            all_entities,
            self.model_config.interaction.attribute_max,
            fallback_names,
            pad_token="<PAD>",
        )
        value_set = sorted({value for values in ent2values.values() for value in values if value != "<PAD>"})
        value_batch_size = max(32, min(512, self.model_config.basic_unit.test_batch_size * 4))
        value_embeddings, value_list = attribute_value_embeddings(
            model,
            value_set,
            tokenizer,
            batch_size=value_batch_size,
            device=device,
        )
        value_embeddings = torch.tensor(value_embeddings, dtype=torch.float32, device=device)
        value_embeddings = F.normalize(value_embeddings, p=2, dim=1)
        pad_value_emb = torch.zeros(1, value_embeddings.size(1), device=device)
        value_embeddings = torch.cat([value_embeddings, pad_value_emb], dim=0)
        pad_value_id = value_embeddings.size(0) - 1

        value2index = {value: idx for idx, value in enumerate(value_list)}
        ent2value_ids = {
            ent: [value2index.get(v, pad_value_id) if v != "<PAD>" else pad_value_id for v in values]
            for ent, values in ent2values.items()
        }

        model.eval()

        attribute_feats = attribute_features(
            artifacts.entity_pairs,
            value_embeddings,
            ent2value_ids,
            pad_value_id,
            self.model_config.interaction.kernel_num,
            device,
            batch_size=2048,
        )

        description_feats = description_features(
            artifacts.entity_pairs,
            entity_embeddings,
            device,
            batch_size=1024,
        )

        combined_features = [
            neighbor_feats[i] + attribute_feats[i] + description_feats[i]
            for i in range(len(artifacts.entity_pairs))
        ]
        logger.debug(
            "[BERT-INT] Combined feature vector dimension: %d (pairs=%d)",
            len(combined_features[0]) if combined_features else 0,
            len(combined_features),
        )

        logger.info(
            "[BERT-INT] Interaction training with %d feature vectors; train_pairs=%d, test_pairs=%d",
            len(combined_features),
            len(artifacts.train_pairs),
            len(artifacts.test_pairs),
        )
        interaction_artifacts = train_interaction_model(
            combined_features,
            artifacts.entity_pairs,
            artifacts.train_pairs,
            artifacts.test_pairs,
            artifacts.train_candidates,
            artifacts.test_candidates,
            epochs=self.model_config.interaction.epochs,
            batch_size=self.model_config.interaction.batch_size,
            learning_rate=self.model_config.interaction.learning_rate,
            margin=self.model_config.interaction.margin,
            neg_num=self.model_config.interaction.negatives,
            candidate_topk=self.model_config.interaction.candidate_topk,
            device=device,
        )

        scored_predictions: List[Tuple[str, str, float]] = []
        logger.info("[BERT-INT] Interaction artifacts scores: %d source entities", len(interaction_artifacts.scores))
        for src, candidates in interaction_artifacts.scores.items():
            src_uri = dataset.index2entity[src]
            allowed = set(artifacts.test_candidates.get(src, []))
            # If allowed is empty, include all candidates; otherwise only include allowed ones
            if allowed:
                filtered = [(tgt, score) for tgt, score in candidates if tgt in allowed]
            else:
                filtered = candidates
            logger.debug("[BERT-INT] Source %s: %d candidates, %d filtered", src_uri, len(candidates), len(filtered))
            for tgt, score in filtered:
                tgt_uri = dataset.index2entity[tgt]
                scored_predictions.append((src_uri, tgt_uri, score))
        logger.info("[BERT-INT] Total scored predictions: %d", len(scored_predictions))
        return scored_predictions

    def _build_entity_texts(self, dataset, bert_dataset: BertIntDataset) -> Dict[str, str]:
        dataset_name = self.stage_config.get("experiment", {}).get("dataset", "")

        source_texts = build_graph_entity_texts(
            dataset.knowledge_graph_source,
            dataset_name,
            kg_index=1,
        )
        target_texts = build_graph_entity_texts(
            dataset.knowledge_graph_target,
            dataset_name,
            kg_index=2,
        )

        texts = {**source_texts, **target_texts}
        overrides = self._description_overrides()
        if overrides:
            replaced = 0
            for entity in list(texts.keys()):
                if entity in overrides:
                    texts[entity] = overrides[entity]
                    replaced += 1
            logger.debug("[BERT-INT] Description dictionary applied to %d entities.", replaced)
        for entity in bert_dataset.index2entity.values():
            if entity not in texts:
                texts[entity] = friendly_name(entity, dataset_name)
        return texts

    def _train_ratio(self) -> float:
        ratio = self.model_config.basic_unit.train_ill_rate
        if ratio <= 0 or ratio >= 1:
            return 0.7
        return ratio

    def _load_model_config(self) -> BertIntConfig:
        base_cfg: Dict[str, object] = {}
        if Path(self.MODEL_CONFIG_PATH).exists():
            base_cfg = load_yaml(self.MODEL_CONFIG_PATH).get("model", {})
        overrides = self.stage_config.get("models", {}).get("bert_int", {})
        merged = {**base_cfg, **overrides}
        return BertIntConfig.from_dict(merged)

    @staticmethod
    def _normalise_pairs(pairs: Iterable[Tuple[URIRef, URIRef]]):
        for left, right in pairs:
            yield (str(left), str(right))

    def _resolve_device(self) -> torch.device:
        desired = self.model_config.device
        if desired.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("[BERT-INT] CUDA requested but unavailable, falling back to CPU")
            return torch.device("cpu")
        return torch.device(desired)

    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "hits@1": 0.0,
            "hits@10": 0.0,
            "mrr": 0.0,
        }
