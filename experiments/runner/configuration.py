"""Normalization helpers for experiment configuration payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


def _ensure_sequence(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


@dataclass(frozen=True)
class ExperimentConfig:
    """Normalized experiment configuration derived from the raw YAML payload."""

    name: str
    suite: Optional[str]  # Optional suite name for grouping experiments
    dataset: Any
    ratio: Optional[float]
    augmentation: Optional[str]
    models: List[str]
    reduction_method: str
    reduction_writer: Optional[str]
    reduction_save_dataset: bool
    reduction_save_model: bool
    reduction_eval: bool
    augmentation_writer: Optional[str]
    augmentation_save_dataset: bool
    augmentation_save_model: bool
    augmentation_eval: bool
    clear_intermediate: bool
    overwrite_existing: bool

    @property
    def resume(self) -> bool:
        return not self.overwrite_existing

    @property
    def direct_mode(self) -> bool:
        return self.ratio is None

    @classmethod
    def from_payload(
        cls,
        payload: Dict[str, Any],
        *,
        cli_overwrite: Optional[bool],
        default_overwrite: bool,
    ) -> "ExperimentConfig":
        name = cls._extract_name(payload)
        suite = cls._extract_suite(payload)
        datasets = cls._extract_datasets(payload)
        if len(datasets) != 1:
            raise ValueError(
                f"Exactly one dataset must be defined per experiment (found {len(datasets)})."
            )
        dataset_entry = datasets[0]

        ratios = cls._extract_ratios(payload)
        if len(ratios) > 1:
            raise ValueError(
                f"Only one reduction ratio is supported per experiment (found {ratios})."
            )
        ratio_value = ratios[0] if ratios else None

        augmentations = cls._extract_augmentations(payload)
        if len(augmentations) > 1:
            raise ValueError(
                f"Only one augmentation method is supported per experiment (found {augmentations})."
            )
        augmentation_entry = augmentations[0] if augmentations else None

        models = cls._extract_models(payload)
        (
            reduction_method,
            reduction_writer,
            reduction_save_dataset,
            reduction_save_model,
            reduction_eval,
        ) = cls._extract_reduction(payload)
        (
            augmentation_writer,
            augmentation_save_dataset,
            augmentation_save_model,
            augmentation_eval,
        ) = cls._extract_augmentation(payload)
        clear_intermediate = bool(payload.get("clear", False))

        effective_overwrite = cls._resolve_overwrite(
            payload.get("overwrite_existing"),
            cli_overwrite,
            default_overwrite,
        )

        return cls(
            name=name,
            suite=suite,
            dataset=dataset_entry,
            ratio=ratio_value,
            augmentation=augmentation_entry,
            models=models,
            reduction_method=reduction_method,
            reduction_writer=reduction_writer,
            reduction_save_dataset=reduction_save_dataset,
            reduction_save_model=reduction_save_model,
            reduction_eval=reduction_eval,
            augmentation_writer=augmentation_writer,
            augmentation_save_dataset=augmentation_save_dataset,
            augmentation_save_model=augmentation_save_model,
            augmentation_eval=augmentation_eval,
            clear_intermediate=clear_intermediate,
            overwrite_existing=effective_overwrite,
        )

    @staticmethod
    def _extract_name(payload: Dict[str, Any]) -> str:
        try:
            return payload["name"]
        except KeyError as exc:
            raise KeyError(
                f"Missing required experiment configuration key: 'name'. "
                f"Available keys: {list(payload.keys())}"
            ) from exc

    @staticmethod
    def _extract_suite(payload: Dict[str, Any]) -> Optional[str]:
        """Extract optional suite name for grouping experiments.

        Suite allows organizing related experiments into a common directory.
        For example: suite="quality_evaluation_bert_int" groups all quality
        evaluation experiments for bert_int model together.
        """
        return payload.get("suite", None)

    @staticmethod
    def _extract_datasets(payload: Dict[str, Any]) -> List[Any]:
        if "datasets" in payload:
            datasets_cfg = payload["datasets"]
        elif "dataset" in payload:
            datasets_cfg = payload["dataset"]
        else:
            raise KeyError(
                f"Experiment configuration must define 'dataset' or 'datasets'. "
                f"Available keys: {list(payload.keys())}"
            )
        return _ensure_sequence(datasets_cfg)

    @staticmethod
    def _extract_ratios(payload: Dict[str, Any]) -> List[float]:
        if "reduction_ratios" in payload:
            ratios: Sequence[Any] = payload["reduction_ratios"]
        elif "reduction_ratio" in payload:
            ratios = [payload["reduction_ratio"]]
        elif isinstance(payload.get("reduction"), dict):
            red_cfg = payload["reduction"]
            ratio_value = red_cfg.get("ratio")
            if ratio_value is None:
                return []
            ratios = ratio_value if isinstance(ratio_value, (list, tuple)) else [ratio_value]
        elif isinstance(payload.get("augmentation"), dict):
            aug_cfg = payload["augmentation"]
            ratio_value = aug_cfg.get("reduction")
            if ratio_value is None:
                return []
            ratios = [ratio_value]
        else:
            return []
        return [float(r) for r in _ensure_sequence(ratios)]

    @staticmethod
    def _extract_augmentations(payload: Dict[str, Any]) -> List[str]:
        if "augmentation_methods" in payload:
            augmentations = payload.get("augmentation_methods", [])
        elif payload.get("augmentation_method"):
            augmentations = [payload["augmentation_method"]]
        elif isinstance(payload.get("augmentation"), dict):
            method = payload["augmentation"].get("method")
            augmentations = [method] if method else []
        else:
            augmentations = []
        return [a for a in _ensure_sequence(augmentations) if a]

    @staticmethod
    def _extract_models(payload: Dict[str, Any]) -> List[str]:
        if "models_to_run" in payload:
            models = payload["models_to_run"]
        elif "model" in payload:
            models = [payload["model"]]
        else:
            raise KeyError(
                f"Experiment configuration must define 'model' or 'models_to_run'. "
                f"Available keys: {list(payload.keys())}"
            )
        return [m for m in _ensure_sequence(models) if m]

    @staticmethod
    def _extract_reduction(payload: Dict[str, Any]) -> tuple[str, Optional[str], bool, bool, bool]:
        """Extract reduction configuration: method, writer, save_dataset, save_model, eval.

        Supports both legacy 'save' flag and new 'save_dataset'/'save_model' flags.
        If 'save' is specified, it applies to both dataset and model.
        """
        if isinstance(payload.get("reduction"), dict):
            red_cfg = payload["reduction"]
            method = red_cfg.get("method", "random_entities")
            writer = red_cfg.get("writer")

            # Support both legacy 'save' and new granular flags
            if "save" in red_cfg:
                # Legacy mode: 'save' controls both
                save_value = bool(red_cfg["save"])
                save_dataset = save_value
                save_model = save_value
            else:
                # New granular mode
                save_dataset = bool(red_cfg.get("save_dataset", True))
                save_model = bool(red_cfg.get("save_model", True))

            eval_flag = bool(red_cfg.get("eval", True))
            return method, writer, save_dataset, save_model, eval_flag
        return payload.get("reduction_method", "random_entities"), None, True, True, True

    @staticmethod
    def _extract_augmentation(payload: Dict[str, Any]) -> tuple[Optional[str], bool, bool, bool]:
        """Extract augmentation configuration: writer, save_dataset, save_model, eval.

        Supports both legacy 'save' flag and new 'save_dataset'/'save_model' flags.
        If 'save' is specified, it applies to both dataset and model.
        """
        aug_cfg = payload.get("augmentation")
        if isinstance(aug_cfg, dict):
            writer = aug_cfg.get("writer")

            # Support both legacy 'save' and new granular flags
            if "save" in aug_cfg:
                # Legacy mode: 'save' controls both
                save_value = bool(aug_cfg["save"])
                save_dataset = save_value
                save_model = save_value
            else:
                # New granular mode
                save_dataset = bool(aug_cfg.get("save_dataset", True))
                save_model = bool(aug_cfg.get("save_model", True))

            eval_flag = bool(aug_cfg.get("eval", True))
            return writer, save_dataset, save_model, eval_flag
        return None, True, True, True

    @staticmethod
    def _resolve_overwrite(
        config_value: Optional[bool],
        cli_value: Optional[bool],
        default_value: bool,
    ) -> bool:
        if cli_value is not None:
            return bool(cli_value)
        if config_value is not None:
            return bool(config_value)
        return bool(default_value)
