"""Service for orchestrating BART operations (training + interpolation).

This service provides a unified interface for all BART-related operations.
"""

from typing import Tuple, Optional, Dict, Any

from src.core.dataset import Dataset
from src.logger import get_logger

from ..bart.trainer import BARTTrainer
from ..bart.interpolator import BARTInterpolator
from ..models import InterpolationConfig

logger = get_logger(__name__)


class BARTService:
    """Service for managing BART fine-tuning and interpolation.

    This service coordinates the BARTTrainer and BARTInterpolator,
    providing a clean facade for the PLMAugmenter.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the BART service.

        Args:
            config: Full configuration dictionary
        """
        self.config = config
        bart_cfg = config.get("bart", {})
        augmentation_cfg = config.get("augmentation", config)

        # Determine output directory
        stage_root = augmentation_cfg.get("stage_root")
        if stage_root:
            from pathlib import Path
            out_dir = str(Path(stage_root) / "model")
        else:
            out_dir = bart_cfg.get("out_dir", "./bart_plm_model")

        # Extract parameters
        model_name = bart_cfg.get("model_name", "facebook/bart-base")
        seed = config.get("experiment", {}).get("seed", config.get("seed", 0))

        # Initialize trainer
        self.trainer = BARTTrainer(
            model_name=model_name,
            out_dir=out_dir,
            seed=int(seed),
            config=config,
        )

        # Initialize interpolation config
        self.interpolation_config = InterpolationConfig.from_config(config)

        # Interpolator will be created after training
        self._interpolator: Optional[BARTInterpolator] = None

        # Track if fine-tuning is enabled
        self.enable_finetuning = bool(bart_cfg.get("enable_finetuning", True))

        logger.info(f"[BARTService] Initialized (finetuning={'enabled' if self.enable_finetuning else 'disabled'})")

    def fine_tune_if_enabled(self, dataset: Dataset) -> None:
        """Fine-tune BART on the dataset if enabled.

        Args:
            dataset: Dataset to train on
        """
        if not self.enable_finetuning:
            logger.info("[BARTService] Fine-tuning disabled in configuration")
            return

        logger.info("[BARTService] Starting BART fine-tuning...")
        self.trainer.fine_tune_on_dataset(dataset)
        logger.info("[BARTService] BART fine-tuning complete")

        # Initialize interpolator after training
        self._initialize_interpolator()

    def _initialize_interpolator(self) -> None:
        """Initialize the interpolator with the trained model."""
        if self._interpolator is not None:
            return  # Already initialized

        logger.info("[BARTService] Initializing interpolator...")

        self._interpolator = BARTInterpolator(
            model=self.trainer.model,
            tokenizer=self.trainer.tokenizer,
            config=self.interpolation_config,
            device=self.trainer.device,
            max_len_in=96,  # From config if needed
        )

        logger.info("[BARTService] Interpolator initialized")

    def interpolate_values(
        self,
        val_src: str,
        val_tgt: str,
        predicate: str = "",
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[str, str]:
        """Interpolate between source and target values.

        Args:
            val_src: Source attribute value
            val_tgt: Target attribute value
            predicate: Predicate name (for conservative alpha)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (interpolated_src, interpolated_tgt)
        """
        # Ensure interpolator is initialized
        if self._interpolator is None:
            self._initialize_interpolator()

        return self._interpolator.interpolate_pair(
            val_src, val_tgt, predicate, max_new_tokens
        )

    def get_trainer(self) -> BARTTrainer:
        """Get the BART trainer.

        Returns:
            BARTTrainer instance
        """
        return self.trainer

    def get_interpolator(self) -> Optional[BARTInterpolator]:
        """Get the BART interpolator.

        Returns:
            BARTInterpolator instance or None if not initialized
        """
        return self._interpolator

    def is_ready(self) -> bool:
        """Check if the service is ready for interpolation.

        Returns:
            True if interpolator is initialized
        """
        return self._interpolator is not None
