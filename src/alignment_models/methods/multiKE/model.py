"""MultiKE (Multi-view Knowledge Embedding) alignment model.

TF2 in-process implementation using the core/ package.
Replaces the legacy subprocess-based wrapper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from src.alignment_models.registry import MODEL_REGISTRY
from src.config.loader import PROJECT_ROOT, load_yaml
from src.logger import get_logger

if TYPE_CHECKING:
    from src.core.dataset import Dataset

logger = get_logger(__name__)


class _Args:
    """Simple namespace that mirrors the original MultiKE ARGs class."""
    def __init__(self, dic: Dict[str, Any]):
        for k, v in dic.items():
            setattr(self, k, v)


@MODEL_REGISTRY.register("multiKE")
class MultiKEAlignment:
    """MultiKE alignment model (TF2, in-process)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.stage_config = config or {}
        self.model_config = self._load_model_config()

    def evaluate(
        self,
        dataset_reduced: Dataset,
        dataset_augmented: Optional[Dataset],
    ) -> Dict[str, float]:
        from .core.data_model import DataModel
        from .core.predicate_alignment import PredicateAlignModel
        from .core.trainer import MultiKETrainer

        dataset = dataset_augmented or dataset_reduced
        cfg = self.model_config

        logger.info("[MultiKE] Evaluating (aligned=%d)", len(dataset.aligned_entities))

        # Build args namespace
        all_pairs = list(dataset.aligned_entities)
        n = len(all_pairs)
        train_ratio = cfg.get("train_ratio", 0.2)
        valid_ratio = cfg.get("valid_ratio", 0.1)

        args = _Args({
            "dim":                        cfg.get("dim", 75),
            "learning_rate":              cfg.get("learning_rate", 0.001),
            "ITC_learning_rate":          cfg.get("ITC_learning_rate", 0.004),
            "optimizer":                  cfg.get("optimizer", "Adagrad"),
            "max_epoch":                  cfg.get("max_epoch", 200),
            "shared_learning_max_epoch":  cfg.get("shared_learning_max_epoch", 200),
            "batch_size":                 cfg.get("batch_size", 5000),
            "entity_batch_size":          cfg.get("entity_batch_size", 5000),
            "attribute_batch_size":       cfg.get("attribute_batch_size", 5000),
            "neg_triple_num":             cfg.get("neg_triple_num", 10),
            "neg_sampling":               cfg.get("neg_sampling", "truncated"),
            "truncated_epsilon":          cfg.get("truncated_epsilon", 0.98),
            "truncated_freq":             cfg.get("truncated_freq", 20),
            "test_threads_num":           cfg.get("test_threads_num", 4),
            "start_valid":                cfg.get("start_valid", 100),
            "eval_freq":                  cfg.get("eval_freq", 10),
            "top_k":                      cfg.get("top_k", [1, 5, 10]),
            "orthogonal_weight":          cfg.get("orthogonal_weight", 2.0),
            "cv_name_weight":             cfg.get("cv_name_weight", 1.0),
            "cv_weight":                  cfg.get("cv_weight", 1.0),
            "start_predicate_soft_alignment": cfg.get("start_predicate_soft_alignment", 10),
            "predicate_soft_sim":         cfg.get("predicate_soft_sim", 0.85),
            "predicate_init_sim":         cfg.get("predicate_init_sim", 0.90),
        })

        w2v_raw = cfg.get("word2vec_path") or ""
        w2v_path = str(PROJECT_ROOT / w2v_raw) if w2v_raw else None

        data = DataModel(
            dataset=dataset,
            word2vec_path=w2v_path,
            dim=args.dim,
            encoder_epochs=cfg.get("encoder_epoch", 100),
            encoder_active=cfg.get("encoder_active", "tanh"),
            encoder_normalize=cfg.get("encoder_normalize", True),
            literal_normalize=cfg.get("literal_normalize", True),
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            batch_size=512,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            mode=cfg.get("alignment_module", "swapping"),
        )

        pred_align = PredicateAlignModel(
            kgs=data.kgs,
            predicate_init_sim=args.predicate_init_sim,
            predicate_soft_sim=args.predicate_soft_sim,
        )

        trainer = MultiKETrainer(data, pred_align, args)
        metrics = trainer.run()

        logger.info("[MultiKE] Metrics: hits@1=%.4f hits@10=%.4f mrr=%.4f",
                    metrics.get("hits@1", 0), metrics.get("hits@10", 0), metrics.get("mrr", 0))
        return metrics

    def _load_model_config(self) -> Dict[str, Any]:
        path = PROJECT_ROOT / "config/models/multiKE.yaml"
        if not path.exists():
            return {}
        payload = load_yaml(path)
        model_section: Dict[str, Any] = payload.get("model", {})
        stage_override = self.stage_config.get("models", {}).get("multiKE", {})
        if stage_override:
            model_section = {**model_section, **stage_override}
        return model_section


__all__ = ["MultiKEAlignment"]
