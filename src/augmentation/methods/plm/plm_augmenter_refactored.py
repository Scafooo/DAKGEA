"""Refactored PLM augmenter using service-oriented architecture.

This is a cleaner version of PLMAugmenter that delegates to specialized services.
All original logic is maintained but organized in a more maintainable way.
"""

import random
import yaml
from pathlib import Path
from typing import Optional

from src.augmentation.base import AugmentationMethod
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.core.dataset import Dataset

from .services import AttributeMatchingService, BARTService, GraphExpansionService
from .set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph


@AUGMENTATION_REGISTRY.register("plm_refactored")
class PLMAugmenterRefactored(AugmentationMethod):
    """PLM-based data augmentation using service-oriented architecture.

    This refactored version maintains all original functionality while providing
    better separation of concerns through specialized services:
    - AttributeMatchingService: Handles attribute correspondences
    - BARTService: Manages BART fine-tuning and interpolation
    - GraphExpansionService: Coordinates BFS expansion and entity generation

    The augmentation process consists of three main phases:
    1. Attribute Matching: Establish correspondences between attributes
    2. BART Fine-tuning: Adapt BART to domain-specific patterns
    3. Graph Expansion: Generate synthetic entity pairs via BFS
    """

    registry_name = "plm_augmentation_refactored"
    _DEFAULT_CONFIG_PATH = "config/augmentation/plm.yaml"

    def __init__(self, config: Optional[dict] = None):
        """Initialize the PLM augmenter with service architecture.

        Args:
            config: Configuration dictionary (optional)
        """
        # Load and merge configuration
        default_cfg = self._load_default_config()
        if config:
            merged_cfg = self._merge_configs(default_cfg, config)
        else:
            merged_cfg = default_cfg

        super().__init__(merged_cfg)

        # Log configuration source
        if config:
            self.logger.info("[PLM] Using merged configuration (default + user-provided)")
        elif default_cfg:
            self.logger.info(f"[PLM] Using default configuration from {self._DEFAULT_CONFIG_PATH}")
        else:
            self.logger.info("[PLM] Using hardcoded default configuration")

        # Initialize services
        self._init_services()

    @staticmethod
    def _load_default_config() -> dict:
        """Load default configuration from config/augmentation/plm.yaml."""
        config_path = Path(PLMAugmenterRefactored._DEFAULT_CONFIG_PATH)

        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    default_cfg = yaml.safe_load(f)
                return default_cfg or {}
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load default config from {config_path}: {e}")
                return {}
        return {}

    @staticmethod
    def _merge_configs(base: dict, override: dict) -> dict:
        """Deep merge two configuration dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PLMAugmenterRefactored._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def _init_services(self) -> None:
        """Initialize the service layer."""
        self.logger.info("[PLM] Initializing services...")

        # 1. Attribute Matching Service
        self.matching_service = AttributeMatchingService(self.config)

        # 2. BART Service
        self.bart_service = BARTService(self.config)

        # 3. Graph Expansion Service
        self.expansion_service = GraphExpansionService(
            bart_service=self.bart_service,
            matching_service=self.matching_service,
            config=self.config,
        )

        self.logger.info("[PLM] Services initialized successfully")

    def augment(self, dataset: Dataset) -> Dataset:
        """Augment a dataset by spawning aligned synthetic entities.

        This orchestrates the three-phase augmentation process:
        1. Compute attribute correspondences (ground-truth + semantic)
        2. Fine-tune BART on domain-specific attribute pairs
        3. Expand graph via BFS, generating synthetic entities

        Args:
            dataset: Input dataset to augment

        Returns:
            Augmented dataset with new aligned entity pairs
        """
        initial_pairs = len(dataset.aligned_entities)
        dataset_augmented = dataset.clone()

        if not dataset.aligned_entities:
            self.logger.warning("[PLM] No aligned entities available, skipping augmentation")
            return dataset_augmented

        # ------------------------------------------------------------------
        # Phase 1: Attribute Matching
        # ------------------------------------------------------------------
        self.section("Attribute Matching")
        correspondences = self.matching_service.compute_correspondences(dataset_augmented)

        if correspondences:
            ground_truth_count = sum(1 for c in correspondences if c.is_ground_truth)
            semantic_count = sum(1 for c in correspondences if c.is_semantic)
            self.logger.info(
                f"[PLM] Computed {len(correspondences)} attribute correspondences "
                f"({ground_truth_count} ground-truth + {semantic_count} semantic)"
            )
        else:
            self.logger.warning("[PLM] No attribute correspondences found")

        # ------------------------------------------------------------------
        # Phase 2: BART Fine-tuning
        # ------------------------------------------------------------------
        self.section("BART Fine-tuning")
        self.bart_service.fine_tune_if_enabled(dataset_augmented)

        # ------------------------------------------------------------------
        # Phase 3: Graph Expansion
        # ------------------------------------------------------------------
        self.section("Graph Expansion")

        # Create Set Knowledge Graph
        self.logger.info("[PLM] Creating Set Knowledge Graph...")
        set_graph = SetKnowledgeGraph.from_dataset(dataset_augmented)
        set_nodes = list(set_graph.iter_set_nodes())

        if not set_nodes:
            self.logger.warning("[PLM] No set nodes found, skipping expansion")
            return dataset_augmented

        self.logger.info(f"[PLM] Created Set KG with {len(set_nodes)} set nodes")

        # Perform expansion
        dataset_augmented = self.expansion_service.expand_dataset(
            dataset_augmented, set_graph
        )

        # Log final statistics
        final_pairs = len(dataset_augmented.aligned_entities)
        added_pairs = final_pairs - initial_pairs

        self.logger.info(
            f"[PLM] Augmentation complete: {initial_pairs} → {final_pairs} pairs "
            f"(+{added_pairs} new pairs)"
        )

        return dataset_augmented

    def section(self, title: str) -> None:
        """Log a section header."""
        self.logger.info("=" * 80)
        self.logger.info(f"  {title}")
        self.logger.info("=" * 80)
