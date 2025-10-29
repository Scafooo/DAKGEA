from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from rdflib import URIRef

from src.alignment_models.methods.bert_int.config import BertIntConfig
from src.alignment_models.methods.bert_int.metrics import evaluate_alignment
from src.alignment_models.registry import MODEL_REGISTRY
from src.config.loader import PROJECT_ROOT, load_yaml
from src.logger import get_logger

logger = get_logger(__name__)

AlignmentPair = Tuple[str, str]
ScoredAlignment = Tuple[str, str, float]


@MODEL_REGISTRY.register("bert_int")
class Bert_int:
    """Wrapper integrating the BERT-INT alignment model into the experiment runner."""

    MODEL_CONFIG_PATH = PROJECT_ROOT / "config/models/bert_int.yaml"

    def __init__(self, config):
        self.stage_config = config or {}
        self.model_config = self._load_model_config()
        logger.debug("[BERT-INT] Loaded configuration: %s", self.model_config.to_dict())

    def evaluate(self, dataset_reduced, dataset_augmented):
        """Compute alignment metrics for the provided datasets."""
        logger.info("[BERT-INT] Evaluating dataset '%s'", self.stage_config.get("experiment", {}).get("dataset"))

        truth_pairs = list(self._normalise_pairs(dataset_reduced.aligned_entities))
        predictions_source = dataset_augmented or dataset_reduced
        predicted_pairs = list(self._normalise_pairs(predictions_source.aligned_entities))

        scored_predictions = self._score_predictions(predicted_pairs)
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

    def _load_model_config(self) -> BertIntConfig:
        """Merge base configuration with stage-level overrides."""
        base_cfg: Dict[str, object] = {}
        if Path(self.MODEL_CONFIG_PATH).exists():
            base_cfg = load_yaml(self.MODEL_CONFIG_PATH).get("model", {})

        overrides = self.stage_config.get("models", {}).get("bert_int", {})
        merged = {**base_cfg, **overrides}
        return BertIntConfig.from_dict(merged)

    @staticmethod
    def _normalise_pairs(pairs: Iterable[Tuple[URIRef, URIRef]]) -> Iterable[AlignmentPair]:
        for left, right in pairs:
            yield (str(left), str(right))

    @staticmethod
    def _score_predictions(pairs: Iterable[AlignmentPair]) -> Iterable[ScoredAlignment]:
        """Assign a default confidence score to predicted pairs.

        Replace this placeholder once the full BERT-INT scoring pipeline is
        integrated, yielding calibrated similarity values instead of a flat
        confidence score.
        """
        default_score = 1.0
        for left, right in pairs:
            yield left, right, default_score
