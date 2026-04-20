"""AttrE alignment model — pure PyTorch implementation.

Implements the two-view entity alignment architecture from:
  Trisedya et al. (2019) "Entity Alignment between Knowledge Graphs Using
  Attribute Embeddings". AAAI 2019.

The model operates directly on the project-standard Dataset object (no
subprocess, no pickle files) via the data_pipeline module.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.alignment_models.methods.attrE.data_pipeline import build_attre_data
from src.alignment_models.methods.attrE.model_core import AttrEModel
from src.alignment_models.methods.attrE.trainer import AttrETrainer
from src.alignment_models.registry import MODEL_REGISTRY
from src.config.loader import PROJECT_ROOT, load_yaml
from src.core.dataset import Dataset
from src.logger import get_logger

logger = get_logger(__name__)


@MODEL_REGISTRY.register("attrE")
class AttrEAlignment:
    """PyTorch AttrE model: dual-view entity alignment via TransE + character n-grams.

    Training uses ``dataset_augmented`` when available, otherwise falls back to
    ``dataset_reduced``, following the same convention as other models in this
    project.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.stage_config = config or {}
        self.model_config = self._load_model_config()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset_reduced: Dataset,
        dataset_augmented: Optional[Dataset],
    ) -> Dict[str, float]:
        """Train AttrE on *dataset* and return alignment metrics.

        Args:
            dataset_reduced: The reduced (label-suppressed) dataset.
            dataset_augmented: The augmented dataset, if available.  When
                present this is used for training; otherwise *dataset_reduced*
                is used.

        Returns:
            ``{"hits@1": float, "hits@10": float, "mrr": float}``
        """
        dataset = dataset_augmented if dataset_augmented is not None else dataset_reduced

        logger.info("[AttrE] Building data bundle from Dataset object …")
        train_ratio = float(self.model_config.get("train_ratio", 0.3))
        char_seq_len = int(self.model_config.get("char_seq_len", 10))
        filter_noise_attr = bool(self.model_config.get("filter_noise_attr", True))
        bundle = build_attre_data(
            dataset,
            train_ratio=train_ratio,
            char_seq_len=char_seq_len,
            filter_noise_attr=filter_noise_attr,
        )

        logger.info(
            "[AttrE] Vocabulary: %d entities, %d predicates, %d chars",
            bundle.num_entities,
            bundle.num_predicates,
            bundle.num_chars,
        )

        model = AttrEModel(
            num_entities=bundle.num_entities,
            num_predicates=bundle.num_predicates,
            num_chars=bundle.num_chars,
            hidden_dim=int(self.model_config.get("hidden_dim", 100)),
            char_seq_len=bundle.char_seq_len,
        )

        trainer = AttrETrainer(
            model=model,
            data=bundle,
            config=self.model_config,
            device_spec=self.model_config.get("device"),
        )

        logger.info("[AttrE] Starting training …")
        history = trainer.fit()

        if history:
            last = history[-1]
            logger.info(
                "[AttrE] Training complete — last eval: "
                "hits@1=%.4f  hits@10=%.4f  mrr=%.4f",
                last["hits@1"], last["hits@10"], last["mrr"],
            )

        logger.info("[AttrE] Running final evaluation …")
        metrics = trainer.evaluate()
        logger.info(
            "[AttrE] Final: hits@1=%.4f  hits@10=%.4f  mrr=%.4f",
            metrics["hits@1"], metrics["hits@10"], metrics["mrr"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_model_config(self) -> Dict[str, Any]:
        path = PROJECT_ROOT / "config/models/attrE.yaml"
        model_section: Dict[str, Any] = {}
        if path.exists():
            payload = load_yaml(path)
            model_section = payload.get("model", {})

        # Allow per-experiment overrides under ``models.attrE`` in stage config
        stage_override = self.stage_config.get("models", {}).get("attrE", {})
        if stage_override:
            model_section = {**model_section, **stage_override}

        return model_section


__all__ = ["AttrEAlignment"]
