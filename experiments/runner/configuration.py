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
    datasets: List[Any]
    ratios: List[float]
    augmentations: List[str]
    models: List[str]
    reduction_method: str
    reduction_writer: Optional[str]
    reduction_save: bool
    reduction_eval: bool
    augmentation_writer: Optional[str]
    augmentation_save: bool
    augmentation_eval: bool
    clear_intermediate: bool
    overwrite_existing: bool

    @property
    def resume(self) -> bool:
        return not self.overwrite_existing

    @property
    def direct_mode(self) -> bool:
        return len(self.ratios) == 0

    @classmethod
    def from_payload(
        cls,
        payload: Dict[str, Any],
        *,
        cli_overwrite: Optional[bool],
        default_overwrite: bool,
    ) -> "ExperimentConfig":
        name = cls._extract_name(payload)
        datasets = cls._extract_datasets(payload)
        ratios = cls._extract_ratios(payload)
        augmentations = cls._extract_augmentations(payload)
        models = cls._extract_models(payload)
        reduction_method, reduction_writer, reduction_save, reduction_eval = cls._extract_reduction(payload)
        augmentation_writer, augmentation_save, augmentation_eval = cls._extract_augmentation(payload)
        clear_intermediate = bool(payload.get("clear", False))

        effective_overwrite = cls._resolve_overwrite(
            payload.get("overwrite_existing"),
            cli_overwrite,
            default_overwrite,
        )

        return cls(
            name=name,
            datasets=datasets,
            ratios=ratios,
            augmentations=augmentations,
            models=models,
            reduction_method=reduction_method,
            reduction_writer=reduction_writer,
            reduction_save=reduction_save,
            reduction_eval=reduction_eval,
            augmentation_writer=augmentation_writer,
            augmentation_save=augmentation_save,
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
    def _extract_reduction(payload: Dict[str, Any]) -> tuple[str, Optional[str], bool, bool]:
        """Extract reduction configuration: method, writer, save, eval."""
        if isinstance(payload.get("reduction"), dict):
            red_cfg = payload["reduction"]
            method = red_cfg.get("method", "random_entities")
            writer = red_cfg.get("writer")
            save = bool(red_cfg.get("save", False))
            eval_flag = bool(red_cfg.get("eval", False))
            return method, writer, save, eval_flag
        return payload.get("reduction_method", "random_entities"), None, False, False

    @staticmethod
    def _extract_augmentation(payload: Dict[str, Any]) -> tuple[Optional[str], bool, bool]:
        """Extract augmentation configuration: writer, save, eval."""
        aug_cfg = payload.get("augmentation")
        if isinstance(aug_cfg, dict):
            writer = aug_cfg.get("writer")
            save = bool(aug_cfg.get("save", False))
            eval_flag = bool(aug_cfg.get("eval", False))
            return writer, save, eval_flag
        return None, False, False

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
