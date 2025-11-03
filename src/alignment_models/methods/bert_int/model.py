"""Integration entry point for the BERT-INT alignment model."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from src.alignment_models.methods.bert_int import (
    load_basic_unit_data,
    load_bert_int_config,
)
from src.alignment_models.methods.bert_int.basic_unit import BasicBertUnit, BasicUnitTrainer
from src.alignment_models.registry import MODEL_REGISTRY
from src.logger import get_logger

logger = get_logger(__name__)

Pair = Tuple[int, int]


def _extract_overrides(stage_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return model-specific overrides from the stage configuration."""
    if not stage_config:
        return {}
    if "model" in stage_config:
        # Allow direct overrides under "model"
        return stage_config.get("model") or {}
    models_section = stage_config.get("models") or {}
    return models_section.get("bert_int") or {}


@MODEL_REGISTRY.register("bert_int")
class BertIntAlignment:
    """Coordinate loading, training, and evaluation for the BERT-INT basic unit."""

    def __init__(self, stage_config: Optional[Dict[str, Any]] = None) -> None:
        self.stage_config = stage_config or {}
        overrides = _extract_overrides(self.stage_config)
        self.config = load_bert_int_config(overrides=overrides)
        self.paths = self.config["paths_resolved"]
        self.basic_cfg = self.config["basic_unit"]
        logger.info(
            "[BERT-INT] Initialised (encoder=%s, device=%s, epochs=%d, pretrained=%s)",
            self.basic_cfg.get("encoder_name"),
            self.config.get("device"),
            self.basic_cfg.get("epochs"),
            self.basic_cfg.get("load_pretrained"),
        )

    def evaluate(self, dataset_reduced, dataset_augmented):
        """Execute the basic unit training/evaluation loop and return metrics."""
        data_bundle = load_basic_unit_data(self.basic_cfg, self.config["paths_resolved"])
        model = BasicBertUnit(self.basic_cfg)
        trainer = BasicUnitTrainer(
            model=model,
            config=self.basic_cfg,
            data=data_bundle,
            paths=self.paths,
            device_spec=self.config.get("device"),
        )

        skip_training = bool(self.stage_config.get("skip_training"))
        history: Sequence[Dict[str, float]] = []
        if not skip_training and self.basic_cfg.get("epochs", 0) > 0:
            history = trainer.fit()
        else:
            logger.info(
                "[BERT-INT] Skipping training (skip_training=%s, epochs=%d)",
                skip_training,
                self.basic_cfg.get("epochs", 0),
            )

        metrics = trainer.evaluate(
            self._evaluation_pairs(data_bundle),
            batch_size=self.basic_cfg.get("eval_batch_size"),
        )
        if history:
            metrics.update(
                {
                    "loss": history[-1]["loss"],
                    "epochs_trained": len(history),
                }
            )
        else:
            metrics.setdefault("epochs_trained", 0)
        logger.info(
            "[BERT-INT] Completed evaluation: hits@1=%.4f hits@10=%.4f mrr=%.4f",
            metrics.get("hits@1", 0.0),
            metrics.get("hits@10", 0.0),
            metrics.get("mrr", 0.0),
        )
        return metrics

    @staticmethod
    def _evaluation_pairs(data_bundle) -> Sequence[Pair]:
        """Return evaluation alignment pairs."""
        if data_bundle.test_ill:
            return data_bundle.test_ill
        logger.warning("[BERT-INT] No test pairs available; falling back to training pairs.")
        return data_bundle.train_ill or data_bundle.ent_ill


__all__ = ("BertIntAlignment",)
