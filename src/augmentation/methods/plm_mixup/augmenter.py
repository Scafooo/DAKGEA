"""PLM Mix-up Augmenter: Data Augmentation via Latent Space Interpolation.

Uses Mix-up of encoder hidden states with predicate conditioning via special tokens.
Training: DAE + Identity Mapping with Seq2SeqTrainer.
Inference: Asymmetric latent interpolation (one encode, two decode).

Supports multiple backends:
- BART (facebook/bart-base)
- FLAN-T5-XL (google/flan-t5-xl) with LoRA
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional, Union

from rdflib import URIRef

from src.augmentation.base import AugmentationMethod
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.core.dataset import Dataset
from src.logger import get_logger

from src.augmentation.methods.plm.mixup_bart_interpolator import MixupBartInterpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder
from src.augmentation.methods.plm.plm_augmenter import PLMAugmenter

logger = get_logger(__name__)

# Type alias for interpolator
Interpolator = Union[MixupBartInterpolator, "MixupT5XLInterpolator"]


@AUGMENTATION_REGISTRY.register("plm_mixup")
class PLMMixupAugmenter(PLMAugmenter):
    """Augmenter based on Mix-up of encoder hidden states.

    Inherits BFS expansion logic from PLMAugmenter, overrides model
    initialization and training to use:
    - Predicate conditioning via special tokens
    - DAE + Identity training with Seq2SeqTrainer
    - Asymmetric latent interpolation for inference

    Supports two backends:
    - BART (facebook/bart-base) - default
    - FLAN-T5-XL (google/flan-t5-xl) with LoRA - for better quality

    Configuration:
        augmentation:
          method: plm_mixup
          ratio: 0.5
          backbone: "bart"  # or "flan-t5-xl"
          pretrained_model_dir: null  # Path to pre-trained model (skip fine-tuning)

          bart:  # Config for BART backbone
            model_name: "facebook/bart-base"
            epochs: 5
            batch_size: 16
            learning_rate: 5.0e-5

          flan_t5:  # Config for FLAN-T5 backbone
            model_name: "google/flan-t5-xl"
            epochs: 3
            batch_size: 8
            learning_rate: 1.0e-3
            lora:
              r: 16
              alpha: 32
              dropout: 0.05
            generation:
              temperature: 1.0
              num_beams: 4
              repetition_penalty: 2.0
              top_p: 0.9
              latent_noise_std: 0.05
    """

    registry_name = "plm_mixup_augmentation"
    _DEFAULT_CONFIG_PATH = "config/augmentation/plm_mixup.yaml"

    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)

        # Parse backbone configuration
        aug_cfg = self.config.get("augmentation", {})
        self.backbone = aug_cfg.get("backbone", "bart").lower()
        self.pretrained_model_dir = aug_cfg.get("pretrained_model_dir")

        # Parse backbone-specific config
        if "t5" in self.backbone:
            self.t5_cfg = aug_cfg.get("flan_t5", {})
            self.t5_model_name = self.t5_cfg.get("model_name", "google/flan-t5-xl")
            self.t5_epochs = int(self.t5_cfg.get("epochs", 3))
            self.t5_batch_size = int(self.t5_cfg.get("batch_size", 8))
            self.t5_learning_rate = float(self.t5_cfg.get("learning_rate", 1e-3))
            self.t5_generation_config = self.t5_cfg.get("generation", {})

            # Determine output dir
            stage_root = aug_cfg.get("stage_root")
            if stage_root:
                self.t5_out_dir = str(Path(stage_root) / "model")
            else:
                self.t5_out_dir = self.t5_cfg.get("out_dir", "./flan_t5_mixup_model")

        logger.info(f"[PLM-MIXUP] Using backbone: {self.backbone}")
        if self.pretrained_model_dir:
            logger.info(f"[PLM-MIXUP] Will load pre-trained model from: {self.pretrained_model_dir}")

    def _build_canonical_mapping(self, dataset: Dataset) -> Dict[str, str]:
        """Generate predicate URI -> special token mapping from attribute_matches.

        Groups matched predicates under the same canonical token derived
        from the source predicate's local name.

        Example:
            attribute_matches = {"http://schema.org/name": ["http://xmlns.com/foaf/0.1/name"]}
            -> {"http://schema.org/name": "<name>", "http://xmlns.com/foaf/0.1/name": "<name>"}
        """
        mapping: Dict[str, str] = {}

        if not dataset.attribute_matches:
            logger.warning("[PLM-MIXUP] No attribute_matches available, using empty mapping.")
            return mapping

        for src_uri, tgt_uris in dataset.attribute_matches.items():
            local = src_uri.split("/")[-1].split("#")[-1]
            token = f"<{local}>"
            mapping[src_uri] = token
            for tgt_uri in tgt_uris:
                mapping[tgt_uri] = token

        logger.info(f"[PLM-MIXUP] Built canonical mapping: {len(mapping)} URIs -> {len(set(mapping.values()))} tokens")
        for token in sorted(set(mapping.values())):
            uris = [u for u, t in mapping.items() if t == token]
            logger.info(f"  {token}: {len(uris)} predicates")

        return mapping

    # ------------------------------------------------------------------
    # Override: Model Initialization and Fine-tuning
    # ------------------------------------------------------------------
    def _initialize_and_finetune_bart(self, dataset: Dataset) -> None:
        """Initialize interpolator and fine-tune with DAE + Identity.

        Supports two backends:
        - BART: MixupBartInterpolator
        - FLAN-T5-XL: MixupT5XLInterpolator with LoRA

        If pretrained_model_dir is set, loads pre-trained model instead of training.
        """
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        # Build canonical mapping from attribute_matches
        canonical_mapping = self._build_canonical_mapping(dataset)

        # Route to appropriate backend
        if "t5" in self.backbone:
            self._initialize_t5_backend(dataset, device, canonical_mapping)
        else:
            self._initialize_bart_backend(dataset, device, canonical_mapping)

    def _initialize_t5_backend(self, dataset: Dataset, device: str, canonical_mapping: Dict[str, str]) -> None:
        """Initialize FLAN-T5-XL backend with LoRA."""
        from src.augmentation.methods.plm.mixup_t5_xl_interpolator import MixupT5XLInterpolator
        import json

        # Check if using pre-trained model
        if self.pretrained_model_dir and Path(self.pretrained_model_dir).exists():
            self.logger.info(f"[PLM-MIXUP] Loading pre-trained T5 from {self.pretrained_model_dir}")

            # Load best config from validation sweep (if available)
            best_config_path = Path(self.pretrained_model_dir) / "best_config.json"
            generation_config = self.t5_generation_config.copy() if self.t5_generation_config else {}

            if best_config_path.exists():
                with open(best_config_path) as f:
                    best_config = json.load(f)
                self.logger.info(f"[PLM-MIXUP] Loaded best config from sweep: {best_config}")
                # Apply best params to generation config
                if "temp" in best_config:
                    generation_config["temperature"] = best_config["temp"]
                if "noise" in best_config:
                    generation_config["latent_noise_std"] = best_config["noise"]
                if "alpha" in best_config:
                    generation_config["alpha"] = best_config["alpha"]
            else:
                self.logger.warning("[PLM-MIXUP] No best_config.json found, using default params")

            self.bart_interpolator = MixupT5XLInterpolator(
                model_name=self.t5_model_name,
                out_dir=self.t5_out_dir,
                device=device,
                max_len_in=self.t5_cfg.get("max_len_in", 128),
                pretrained_path=self.pretrained_model_dir,
                generation_config=generation_config,
            )
            self.logger.info("[PLM-MIXUP] Pre-trained T5 model loaded.")
            return

        # Initialize fresh model for fine-tuning
        self.logger.info("[PLM-MIXUP] Initializing MixupT5XLInterpolator for fine-tuning...")
        self.bart_interpolator = MixupT5XLInterpolator(
            model_name=self.t5_model_name,
            out_dir=self.t5_out_dir,
            device=device,
            max_len_in=self.t5_cfg.get("max_len_in", 128),
            generation_config=self.t5_generation_config,
        )

        # Build training data
        self._build_and_finetune(dataset, canonical_mapping, is_t5=True)

    def _initialize_bart_backend(self, dataset: Dataset, device: str, canonical_mapping: Dict[str, str]) -> None:
        """Initialize BART backend."""
        # Check if using pre-trained model
        if self.pretrained_model_dir and Path(self.pretrained_model_dir).exists():
            self.logger.info(f"[PLM-MIXUP] Loading pre-trained BART from {self.pretrained_model_dir}")
            self.bart_interpolator = MixupBartInterpolator(
                model_name=self.bart_model_name,
                out_dir=self.pretrained_model_dir,
                device=device,
                seed=self.seed,
                max_len_in=self.bart_cfg.get("max_len_in", 96),
                generation_config=self.bart_generation_config,
                reuse_if_available=True,
            )
            self.bart_interpolator.set_predicate_mapping(canonical_mapping)
            self.logger.info("[PLM-MIXUP] Pre-trained BART model loaded.")
            return

        # Initialize fresh model for fine-tuning
        self.logger.info("[PLM-MIXUP] Initializing MixupBartInterpolator...")
        self.bart_interpolator = MixupBartInterpolator(
            model_name=self.bart_model_name,
            out_dir=self.bart_out_dir,
            device=device,
            seed=self.seed,
            max_len_in=self.bart_cfg.get("max_len_in", 96),
            generation_config=self.bart_generation_config,
            reuse_if_available=not self.bart_force_retrain,
        )
        self.bart_interpolator.set_predicate_mapping(canonical_mapping)

        # Build training data and fine-tune
        self._build_and_finetune(dataset, canonical_mapping, is_t5=False)

    def _build_and_finetune(self, dataset: Dataset, canonical_mapping: Dict[str, str], is_t5: bool) -> None:
        """Build training data and fine-tune the model."""
        self.logger.info("[PLM-MIXUP] Building training data (DAE + Identity)...")
        builder = MixupDataBuilder()

        if is_t5:
            max_pairs = self.t5_cfg.get("max_pairs_per_pred", 2000)
        else:
            max_pairs = self.bart_cfg.get("max_pairs_per_pred", 2000)

        training_rows, _ = builder.build_training_data(
            dataset=dataset,
            max_pairs_per_pred=max_pairs,
        )

        if not training_rows:
            self.logger.warning("[PLM-MIXUP] No training data generated. Skipping fine-tuning.")
            return

        self.logger.info(f"[PLM-MIXUP] Generated {len(training_rows)} training rows.")

        # Fine-tune
        if is_t5:
            self.bart_interpolator.fine_tune(
                training_rows=training_rows,
                epochs=self.t5_epochs,
                batch_size=self.t5_batch_size,
                lr=self.t5_learning_rate,
            )
        else:
            lr = float(self.bart_cfg.get("learning_rate", 5e-5))
            self.bart_interpolator.fine_tune(
                training_rows=training_rows,
                epochs=self.bart_epochs,
                batch_size=self.bart_batch_size,
                lr=lr,
                force_retrain=self.bart_force_retrain,
            )

        self.logger.info("[PLM-MIXUP] Fine-tuning complete.")

    def _initialize_bart_only(self, dataset: Dataset) -> None:
        """Initialize interpolator without fine-tuning.

        Used when enable_finetuning=false. Must have a pre-trained model available.
        """
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        canonical_mapping = self._build_canonical_mapping(dataset)

        # Route to appropriate backend
        if "t5" in self.backbone:
            from src.augmentation.methods.plm.mixup_t5_xl_interpolator import MixupT5XLInterpolator

            model_path = self.pretrained_model_dir or self.t5_out_dir
            self.bart_interpolator = MixupT5XLInterpolator(
                model_name=self.t5_model_name,
                out_dir=self.t5_out_dir,
                device=device,
                max_len_in=self.t5_cfg.get("max_len_in", 128),
                pretrained_path=model_path,
                generation_config=self.t5_generation_config,
            )
        else:
            model_path = self.pretrained_model_dir or self.bart_out_dir
            self.bart_interpolator = MixupBartInterpolator(
                model_name=self.bart_model_name,
                out_dir=model_path,
                device=device,
                seed=self.seed,
                max_len_in=self.bart_cfg.get("max_len_in", 96),
                generation_config=self.bart_generation_config,
                reuse_if_available=True,
            )
            self.bart_interpolator.set_predicate_mapping(canonical_mapping)

        self.logger.info(f"[PLM-MIXUP] Pre-trained {self.backbone} model ready (no fine-tuning).")
