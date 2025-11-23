"""BART trainer for fine-tuning on knowledge graph attribute pairs.

This module handles the training logic extracted from bart_interpolator.py.
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path

import torch
from transformers import BartForConditionalGeneration, BartTokenizer

from src.core.dataset import Dataset
from src.utils.reproducibility import set_random_seeds
from src.logger import get_logger

# Import the existing BartInterpolatorPLM for now
# We'll gradually migrate functionality
from ..bart_interpolator import BartInterpolatorPLM

logger = get_logger(__name__)


class BARTTrainer:
    """Handles BART model fine-tuning for attribute value generation.

    This class wraps and delegates to BartInterpolatorPLM for now,
    providing a cleaner interface while maintaining all existing functionality.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        out_dir: str = "./bart_plm_model",
        device: Optional[str] = None,
        seed: int = 42,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the BART trainer.

        Args:
            model_name: Pretrained model name or path
            out_dir: Output directory for fine-tuned model
            device: Device to use ('cpu' or 'cuda')
            seed: Random seed for reproducibility
            config: Full configuration dictionary
        """
        self.model_name = model_name
        self.out_dir = out_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.config = config or {}

        # Set reproducibility
        set_random_seeds(seed)

        # Extract BART-specific config
        bart_cfg = self.config.get("bart", {})

        # Training parameters
        self.epochs = int(bart_cfg.get("epochs", 10))
        self.batch_size = int(bart_cfg.get("batch_size", 16))
        self.learning_rate = float(bart_cfg.get("learning_rate", 5e-5))
        self.force_retrain = bool(bart_cfg.get("force_retrain", False))

        # Advanced training config
        self.advanced_training_config = bart_cfg.get("advanced_training", {})
        self.generation_config = bart_cfg.get("generation", {})

        # Initialize the underlying BartInterpolatorPLM
        # This maintains backward compatibility
        self._interpolator = BartInterpolatorPLM(
            model_name=self.model_name,
            out_dir=self.out_dir,
            device=self.device,
            seed=self.seed,
            base_alpha=float(bart_cfg.get("base_alpha", 0.35)),
            alpha_spread=float(bart_cfg.get("alpha_spread", 0.25)),
            max_len_in=int(bart_cfg.get("max_len_in", 96)),
            max_len_out=int(bart_cfg.get("max_len_out", 48)),
            reuse_if_available=True,
            advanced_training_config=self.advanced_training_config,
            generation_config=self.generation_config,
            training_config=bart_cfg,
        )

        logger.info(f"[BARTTrainer] Initialized with model {self.model_name}")

    @property
    def model(self) -> BartForConditionalGeneration:
        """Get the underlying BART model."""
        return self._interpolator.model

    @property
    def tokenizer(self) -> BartTokenizer:
        """Get the BART tokenizer."""
        return self._interpolator.tokenizer

    def fine_tune_on_dataset(self, dataset: Dataset) -> None:
        """Fine-tune BART on attribute pairs from the dataset.

        Args:
            dataset: Dataset containing source and target knowledge graphs
        """
        logger.info("[BARTTrainer] Building training pairs from dataset...")

        # Build training pairs using existing logic
        pairs = self._interpolator.build_pairs_from_dataset(
            dataset.knowledge_graph_source,
            dataset.knowledge_graph_target,
            dataset.aligned_entities,
        )

        logger.info(f"[BARTTrainer] Built {len(pairs)} training pairs")

        if len(pairs) == 0:
            logger.warning("[BARTTrainer] No training pairs found. Skipping fine-tuning.")
            return

        # Check if model already exists
        if self._should_skip_training():
            logger.info(f"[BARTTrainer] Model already exists at {self.out_dir}, skipping training")
            return

        # Fine-tune
        logger.info("[BARTTrainer] Starting fine-tuning...")
        self._interpolator.fine_tune(
            pairs,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.learning_rate,
            force_retrain=self.force_retrain,
        )

        logger.info("[BARTTrainer] Fine-tuning complete")

    def _should_skip_training(self) -> bool:
        """Check if training should be skipped (model already exists)."""
        if self.force_retrain:
            return False

        model_path = Path(self.out_dir)
        if not model_path.exists():
            return False

        # Check if model files exist
        required_files = ["pytorch_model.bin", "config.json"]
        has_files = any(f in os.listdir(self.out_dir) for f in required_files)

        return has_files

    def get_interpolator(self):
        """Get the underlying interpolator for backward compatibility.

        Returns:
            The BartInterpolatorPLM instance
        """
        return self._interpolator

    def save(self, path: Optional[str] = None) -> None:
        """Save the fine-tuned model.

        Args:
            path: Path to save to (defaults to self.out_dir)
        """
        save_path = path or self.out_dir
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"[BARTTrainer] Model saved to {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """Load a fine-tuned model.

        Args:
            path: Path to load from (defaults to self.out_dir)
        """
        load_path = path or self.out_dir

        if not Path(load_path).exists():
            raise FileNotFoundError(f"Model not found at {load_path}")

        self._interpolator.model = BartForConditionalGeneration.from_pretrained(
            load_path
        ).to(self.device)
        self._interpolator.tokenizer = BartTokenizer.from_pretrained(load_path)

        logger.info(f"[BARTTrainer] Model loaded from {load_path}")
