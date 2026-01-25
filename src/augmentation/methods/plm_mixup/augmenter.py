"""PLM Mix-up Augmenter: Data Augmentation via BART Last Hidden State Interpolation.

Uses Mix-up of encoder hidden states with predicate conditioning via special tokens.
Training: DAE + Identity Mapping with Seq2SeqTrainer.
Inference: Asymmetric latent interpolation (one encode, two decode).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

from rdflib import URIRef

from src.augmentation.base import AugmentationMethod
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.core.dataset import Dataset
from src.logger import get_logger

from src.augmentation.methods.plm.mixup_bart_interpolator import MixupBartInterpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder
from src.augmentation.methods.plm.plm_augmenter import PLMAugmenter

logger = get_logger(__name__)


@AUGMENTATION_REGISTRY.register("plm_mixup")
class PLMMixupAugmenter(PLMAugmenter):
    """Augmenter based on Mix-up of BART Last Hidden States.

    Inherits BFS expansion logic from PLMAugmenter, overrides BART
    initialization and training to use:
    - Predicate conditioning via special tokens
    - DAE + Identity training with Seq2SeqTrainer
    - Asymmetric latent interpolation for inference

    Configuration:
        augmentation:
          method: plm_mixup
          ratio: 0.5
          bart:
            model_name: "facebook/bart-base"
            epochs: 5
            batch_size: 16
            learning_rate: 5.0e-5
            base_alpha: 0.5
            max_pairs_per_pred: 2000
            generation:
              max_new_tokens: 32
              temperature: 0.9
    """

    registry_name = "plm_mixup_augmentation"
    _DEFAULT_CONFIG_PATH = "config/augmentation/plm_mixup.yaml"

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
    # Override: BART Initialization and Fine-tuning
    # ------------------------------------------------------------------
    def _initialize_and_finetune_bart(self, dataset: Dataset) -> None:
        """Initialize MixupBartInterpolator and fine-tune with DAE + Identity.

        Overrides PLMAugmenter._initialize_and_finetune_bart() to use:
        1. Predicate conditioning via special tokens
        2. DAE + Identity training data
        3. Seq2SeqTrainer for fine-tuning
        """
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        # Build canonical mapping from attribute_matches
        canonical_mapping = self._build_canonical_mapping(dataset)

        # Determine output directory
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

        # Register predicate special tokens
        self.bart_interpolator.set_predicate_mapping(canonical_mapping)

        # Build training data (DAE + Identity)
        self.logger.info("[PLM-MIXUP] Building training data (DAE + Identity)...")
        builder = MixupDataBuilder()
        max_pairs = self.bart_cfg.get("max_pairs_per_pred", 2000)

        training_rows = builder.build_training_data(
            dataset=dataset,
            canonical_mapping=canonical_mapping,
            max_pairs_per_pred=max_pairs,
        )

        if not training_rows:
            self.logger.warning("[PLM-MIXUP] No training data generated. Skipping fine-tuning.")
            return

        self.logger.info(f"[PLM-MIXUP] Generated {len(training_rows)} training rows.")

        # Fine-tune with Seq2SeqTrainer
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
        """Initialize MixupBartInterpolator without fine-tuning.

        Used when bart.enable_finetuning=false.
        """
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        canonical_mapping = self._build_canonical_mapping(dataset)

        self.bart_interpolator = MixupBartInterpolator(
            model_name=self.bart_model_name,
            out_dir=self.bart_out_dir,
            device=device,
            seed=self.seed,
            max_len_in=self.bart_cfg.get("max_len_in", 96),
            generation_config=self.bart_generation_config,
            reuse_if_available=True,
        )

        self.bart_interpolator.set_predicate_mapping(canonical_mapping)
        self.logger.info("[PLM-MIXUP] Pretrained model ready (no fine-tuning).")
