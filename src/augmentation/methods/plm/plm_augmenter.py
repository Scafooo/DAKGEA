"""PLM-based augmentation strategy leveraging latent interpolation scaffolding.

Refactored version with improved modularity and clarity.
"""

from __future__ import annotations

import math
import os
import random
import yaml
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

from rdflib import Literal, URIRef

from src.augmentation.base import AugmentationMethod
from src.augmentation.registry import AUGMENTATION_REGISTRY
from src.core.dataset import Dataset

from .bart_interpolator import BartInterpolatorPLM
from .expansion_node import ExpansionNode
from .neighbor_handler import NeighborHandler
from .node_expander import NodeExpander
from .set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph


@AUGMENTATION_REGISTRY.register("plm_augmentation")
class PLMAugmenter(AugmentationMethod):
    """Augment datasets via PLM-based expansion of fused entity sets.

    The augmentation process:
    1. Creates a SetKnowledgeGraph where aligned entities are fused
    2. Performs BFS expansion starting from random set nodes
    3. For each set node, creates an aligned pair of augmented entities
    4. Bootstraps literal attributes from original entities
    5. Expands to neighbors while maintaining structural coherence
    6. Handles bridging through non-set nodes for 2-hop expansion

    Configuration:
        - max_depth: Maximum BFS depth (default: 1)
        - ratio: Augmentation ratio (e.g., 0.5 = 50% more entities)
        - max_pairs: Absolute maximum number of pairs to generate
        - seed: Random seed for reproducibility
        - add_derived_predicate: Whether to add derivedFrom triples (default: False)
    """

    registry_name = "plm_augmentation"
    _DEFAULT_DERIVED_PREDICATE = URIRef("http://dakgea.org/augmentation/derivedFrom")
    _DEFAULT_CONFIG_PATH = "config/augmentation/plm.yaml"

    @staticmethod
    def _load_default_config() -> dict:
        """Load default configuration from config/augmentation/plm.yaml.

        Returns:
            Default configuration dictionary
        """
        config_path = Path(PLMAugmenter._DEFAULT_CONFIG_PATH)

        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    default_cfg = yaml.safe_load(f)
                return default_cfg or {}
            except Exception as e:
                # If loading fails, return empty dict (will use hardcoded defaults)
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load default config from {config_path}: {e}")
                return {}
        else:
            # Config file doesn't exist, use hardcoded defaults
            return {}

    @staticmethod
    def _merge_configs(base: dict, override: dict) -> dict:
        """Deep merge two configuration dictionaries.

        Args:
            base: Base configuration
            override: Override configuration (takes precedence)

        Returns:
            Merged configuration
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PLMAugmenter._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def __init__(self, config: Optional[dict] = None):
        # Load default config and merge with provided config
        default_cfg = self._load_default_config()
        if config:
            merged_cfg = self._merge_configs(default_cfg, config)
        else:
            merged_cfg = default_cfg

        super().__init__(merged_cfg)
        cfg = self.config or {}
        augmentation_cfg = cfg.get("augmentation", cfg)
        experiment_cfg = cfg.get("experiment", cfg)

        # Log config source
        if config:
            self.logger.info("[PLM] Using merged configuration (default + user-provided)")
        elif default_cfg:
            self.logger.info(f"[PLM] Using default configuration from {self._DEFAULT_CONFIG_PATH}")
        else:
            self.logger.info("[PLM] Using hardcoded default configuration")

        # Expansion parameters
        self.max_depth = int(augmentation_cfg.get("max_depth", 1))
        max_pairs = augmentation_cfg.get("max_pairs")
        self.max_pairs_config = max(1, int(max_pairs)) if max_pairs is not None else None

        # Augmentation ratio
        ratio = augmentation_cfg.get("ratio")
        self.augmentation_ratio: Optional[float] = None
        if ratio is not None:
            try:
                ratio_value = float(ratio)
                if ratio_value > 0:
                    self.augmentation_ratio = ratio_value
                else:
                    self.logger.warning("augmentation.ratio must be > 0; ignoring '%s'.", ratio)
            except (TypeError, ValueError):
                self.logger.warning("Invalid augmentation.ratio '%s'; ignoring.", ratio)

        # Reproducibility
        seed = experiment_cfg.get("seed", cfg.get("seed", 0))
        self.seed = int(seed) if seed is not None else 0

        # Provenance tracking
        self.add_derived_predicate = bool(augmentation_cfg.get("add_derived_predicate", False))
        derived_predicate = augmentation_cfg.get("derived_predicate", cfg.get("derived_predicate"))
        self.derived_predicate = (
            URIRef(derived_predicate) if derived_predicate else self._DEFAULT_DERIVED_PREDICATE
        )

        # BART fine-tuning parameters
        bart_cfg = augmentation_cfg.get("bart", {})
        self.bart_cfg = bart_cfg  # Store full config for NodeExpander
        self.enable_bart_finetuning = bool(bart_cfg.get("enable_finetuning", True))
        self.bart_model_name = bart_cfg.get("model_name", "facebook/bart-base")
        self.bart_advanced_training_config = bart_cfg.get("advanced_training", {})

        # Determine BART output directory
        # Priority: stage_root/model > configured out_dir > default ./bart_plm_model
        stage_root = augmentation_cfg.get("stage_root")
        if stage_root:
            self.bart_out_dir = str(Path(stage_root) / "model")
        else:
            self.bart_out_dir = bart_cfg.get("out_dir", "./bart_plm_model")

        self.bart_epochs = int(bart_cfg.get("epochs", 10))
        self.bart_batch_size = int(bart_cfg.get("batch_size", 16))
        self.bart_force_retrain = bool(bart_cfg.get("force_retrain", False))

        # BART interpolation parameters
        self.bart_base_alpha = float(bart_cfg.get("base_alpha", 0.35))
        self.bart_alpha_spread = float(bart_cfg.get("alpha_spread", 0.25))

        # Generation parameters
        self.bart_generation_config = bart_cfg.get("generation", {})

        # Predicate matching configuration
        self.predicate_matcher_config = bart_cfg.get("predicate_matching", {})

        # Initialize helper classes
        self.neighbor_handler = NeighborHandler()

        # Initialize BART interpolator (lazy initialization - only if fine-tuning is enabled)
        self.bart_interpolator: Optional[BartInterpolatorPLM] = None

        # Node expander will be initialized after BART (needs interpolator reference)
        self.node_expander: Optional[NodeExpander] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def augment(self, dataset: Dataset) -> Dataset:
        """Augment a dataset by spawning aligned synthetic entities.

        Args:
            dataset: Input dataset to augment

        Returns:
            Augmented dataset with new aligned entity pairs
        """
        initial_pairs = len(dataset.aligned_entities)
        dataset_augmented = dataset.clone()

        if not dataset.aligned_entities:
            self.logger.warning("Skipping PLM augmentation: no aligned entities available.")
            return dataset_augmented

        # ------------------------------------------------------------------
        # Step 1: BART Fine-tuning (if enabled)
        # ------------------------------------------------------------------
        if self.enable_bart_finetuning:
            self.section("BART Fine-tuning")
            self._initialize_and_finetune_bart(dataset)
        else:
            self.logger.info("[PLM] BART fine-tuning disabled in configuration.")

        # ------------------------------------------------------------------
        # Step 1.5: Pre-compute predicate alignments (if enabled)
        # ------------------------------------------------------------------
        alignment_cache = None
        use_value_based_matching = self.predicate_matcher_config.get("use_value_similarity", False)

        if use_value_based_matching:
            self.section("Predicate Alignment Pre-computation")
            try:
                from .predicate_alignment import PredicateAlignmentCache

                name_weight = self.predicate_matcher_config.get("name_weight", 0.7)
                value_weight = self.predicate_matcher_config.get("value_weight", 0.3)
                sample_size = self.predicate_matcher_config.get("alignment_sample_size", 100)

                self.logger.info(f"[PLM] Initializing alignment cache (name_weight={name_weight}, value_weight={value_weight}, sample_size={sample_size})")

                alignment_cache = PredicateAlignmentCache(
                    predicate_matcher_config=self.predicate_matcher_config,
                    name_weight=name_weight,
                    value_weight=value_weight,
                    sample_size=sample_size,
                )

                self.logger.info("[PLM] Computing predicate alignments...")
                # Pre-compute alignments
                alignments = alignment_cache.compute_alignments(dataset_augmented)
                self.logger.info(f"[PLM] ✓ Predicate alignment pre-computation complete: {len(alignments)} alignments found")

                if alignments:
                    avg_combined = sum(a.combined_score for a in alignments) / len(alignments)
                    self.logger.info(f"[PLM] Average combined score: {avg_combined:.3f}")
                    self.logger.info(f"[PLM] Top 3 alignments:")
                    for i, align in enumerate(alignments[:3], 1):
                        self.logger.info(f"  {i}. {align.src_uri} ↔ {align.tgt_uri}")
                        self.logger.info(f"     name={align.name_similarity:.3f}, value={align.value_similarity:.3f}, combined={align.combined_score:.3f}")
                else:
                    self.logger.warning("[PLM] No alignments found! Check threshold or dataset.")
            except Exception as e:
                self.logger.error(f"[PLM] Failed to pre-compute alignments: {e}", exc_info=True)
                self.logger.warning("[PLM] Falling back to on-the-fly matching")
                alignment_cache = None

        # Initialize NodeExpander with BART interpolator and predicate alignment cache
        self.node_expander = NodeExpander(
            self.derived_predicate,
            self.add_derived_predicate,
            self.bart_interpolator,
            self.predicate_matcher_config,
            alignment_cache,  # Pass the cache
            advanced_training_config=self.bart_advanced_training_config,
            bart_config=self.bart_cfg,  # Pass full BART config for unmatched generation
        )

        # ------------------------------------------------------------------
        # Step 2: Create SetKnowledgeGraph and prepare for expansion
        # ------------------------------------------------------------------
        # Create fused set graph
        set_graph = SetKnowledgeGraph.from_dataset(dataset)
        set_nodes = sorted(set_graph.iter_set_nodes(), key=lambda uri: str(uri))

        if not set_nodes:
            self.logger.warning("SetKnowledgeGraph is empty; nothing to augment.")
            return dataset_augmented

        # Calculate expansion budget
        pair_budget = self._compute_pair_budget(initial_pairs)
        self._log_augmentation_start(initial_pairs, pair_budget)

        # Randomize starting order
        rng = random.Random(self.seed)
        rng.shuffle(set_nodes)

        # ------------------------------------------------------------------
        # Step 3: Perform BFS expansion
        # ------------------------------------------------------------------
        self.section("PLM BFS Expansion")
        expanded_pairs = self._bfs_expansion(dataset_augmented, set_graph, set_nodes, pair_budget)

        self.logger.info(
            "[PLM] Augmentation complete: expanded %d pairs (target: %s)",
            expanded_pairs,
            pair_budget if pair_budget is not None else "unbounded",
        )

        return dataset_augmented

    # ------------------------------------------------------------------
    # BART Fine-tuning
    # ------------------------------------------------------------------
    def _initialize_and_finetune_bart(self, dataset: Dataset) -> None:
        """Initialize BART interpolator and perform fine-tuning.

        Args:
            dataset: Dataset to extract training pairs from
        """
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
            self.logger.warning("[BART] PyTorch not available, using CPU.")

        self.logger.info("[BART] Initializing BartInterpolatorPLM...")

        self.bart_interpolator = BartInterpolatorPLM(
            model_name=self.bart_model_name,
            out_dir=self.bart_out_dir,
            device=device,
            seed=self.seed,
            base_alpha=self.bart_base_alpha,
            alpha_spread=self.bart_alpha_spread,
            advanced_training_config=self.bart_advanced_training_config,
            generation_config=self.bart_generation_config,
        )

        # Build training pairs from aligned entities
        self.logger.info("[BART] Building training pairs from dataset...")
        pairs = self.bart_interpolator.build_pairs_from_dataset(
            dataset.knowledge_graph_source,
            dataset.knowledge_graph_target,
            dataset.aligned_entities,
        )
        self.logger.info(f"[BART] Built {len(pairs)} training pairs.")

        # Fine-tune BART
        if len(pairs) > 0:
            self.logger.info("[BART] Starting fine-tuning...")
            self.bart_interpolator.fine_tune(
                pairs,
                epochs=self.bart_epochs,
                batch_size=self.bart_batch_size,
                force_retrain=self.bart_force_retrain,
            )
            self.logger.info("[BART] Fine-tuning complete.")
        else:
            self.logger.warning("[BART] No training pairs found. Skipping fine-tuning.")

    # ------------------------------------------------------------------
    # BFS Expansion
    # ------------------------------------------------------------------
    def _bfs_expansion(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        set_nodes: List[URIRef],
        pair_budget: Optional[int],
    ) -> int:
        """Perform BFS expansion of set nodes.

        Args:
            dataset: Dataset to augment
            set_graph: Set knowledge graph
            set_nodes: List of set nodes to potentially expand
            pair_budget: Maximum number of pairs to create (None = unlimited)

        Returns:
            Number of pairs actually expanded
        """
        visited: set[URIRef] = set()
        expansion_chain: List[str] = []
        queue: Deque[ExpansionNode] = deque()
        remaining_seeds: Deque[URIRef] = deque(set_nodes)
        expanded_pairs = 0
        # Track mapping from set_node to its augmented entities
        set_node_to_augmented: dict[URIRef, tuple[URIRef, URIRef]] = {}

        while (queue or remaining_seeds) and (pair_budget is None or expanded_pairs < pair_budget):
            # Refill queue from remaining seeds if empty
            if not queue:
                while remaining_seeds and remaining_seeds[0] in visited:
                    remaining_seeds.popleft()
                if not remaining_seeds:
                    break
                next_seed = remaining_seeds.popleft()
                queue.append(ExpansionNode(uri=next_seed, depth=0, node_type="set"))

            # Process next node in queue
            exp_node = queue.popleft()

            if exp_node.uri in visited:
                continue

            visited.add(exp_node.uri)
            expansion_chain.append(str(exp_node.uri))

            # Expand the node
            if exp_node.is_set_node:
                self._expand_set_node_with_neighbors(
                    dataset, set_graph, exp_node, queue, visited, set_node_to_augmented
                )
                expanded_pairs += 1
            else:
                # Non-set nodes will be handled by PLM expansion (future work)
                self.logger.debug("[PLM] Non-set node encountered (future expansion): %s", exp_node.uri)

        # Log expansion summary
        self._log_expansion_summary(expanded_pairs, pair_budget, expansion_chain)

        return expanded_pairs

    def _expand_set_node_with_neighbors(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        exp_node: ExpansionNode,
        queue: Deque[ExpansionNode],
        visited: set[URIRef],
        set_node_to_augmented: dict[URIRef, tuple[URIRef, URIRef]],
    ) -> None:
        """Expand a set node and process its neighbors.

        Args:
            dataset: Dataset to augment
            set_graph: Set knowledge graph
            exp_node: Expansion node to process
            queue: BFS queue to add new nodes to
            visited: Set of already visited nodes
            set_node_to_augmented: Mapping from set nodes to their augmented entities
        """
        set_node = exp_node.uri
        depth = exp_node.depth

        # Log connections for debugging
        self._log_node_connections(set_graph, set_node)

        # Create augmented pair
        src_aug, tgt_aug, src_original, tgt_original = self.node_expander.expand_set_node(
            dataset, set_graph, set_node
        )

        # Save mapping for future relation creation
        set_node_to_augmented[set_node] = (src_aug, tgt_aug)

        # Bootstrap literal attributes
        self.node_expander.bootstrap_literals(
            dataset, src_original, tgt_original, src_aug, tgt_aug
        )

        # TODO: PLM-driven expansion (future implementation)
        # This will use a language model to generate variations of literals
        # self._plm_expansion(dataset, set_graph, set_node, src_aug, tgt_aug)

        # Process neighbors if we haven't reached max depth
        if depth < self.max_depth:
            self._process_neighbors(
                dataset, set_graph, set_node, src_aug, tgt_aug, depth, queue, visited, set_node_to_augmented
            )

    def _process_neighbors(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        current_node: URIRef,
        src_aug: URIRef,
        tgt_aug: URIRef,
        current_depth: int,
        queue: Deque[ExpansionNode],
        visited: set[URIRef],
        set_node_to_augmented: dict[URIRef, tuple[URIRef, URIRef]],
    ) -> None:
        """Process all neighbors of the current node.

        Args:
            dataset: Dataset to augment
            set_graph: Set knowledge graph
            current_node: Current set node being processed
            src_aug: Augmented source entity
            tgt_aug: Augmented target entity
            current_depth: Current BFS depth
            queue: BFS queue
            visited: Set of visited nodes
            set_node_to_augmented: Mapping from set nodes to their augmented entities
        """
        direct_enqueued: set[URIRef] = set()

        for direction, predicate, neighbor in self.neighbor_handler.iter_neighbors(
            set_graph, current_node
        ):
            # Handle literal neighbors
            if isinstance(neighbor, Literal):
                self.neighbor_handler.attach_literal_to_augmented(
                    dataset, predicate, neighbor, direction, src_aug, tgt_aug
                )
                continue

            # Skip non-URI neighbors
            if not isinstance(neighbor, URIRef):
                continue

            # Handle set node neighbors (direct expansion)
            if set_graph.is_set_node(neighbor):
                self._enqueue_set_neighbor(
                    dataset, set_graph, neighbor, predicate, direction, current_node, src_aug, tgt_aug,
                    current_depth, queue, visited, direct_enqueued, set_node_to_augmented
                )
                continue

            # Handle non-set node neighbors
            self._handle_non_set_neighbor(
                dataset,
                set_graph,
                neighbor,
                predicate,
                direction,
                current_node,
                src_aug,
                tgt_aug,
                current_depth,
                queue,
                visited,
            )

    def _enqueue_set_neighbor(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        neighbor: URIRef,
        predicate: URIRef,
        direction: str,
        current_node: URIRef,
        src_aug: URIRef,
        tgt_aug: URIRef,
        current_depth: int,
        queue: Deque[ExpansionNode],
        visited: set[URIRef],
        direct_enqueued: set[URIRef],
        set_node_to_augmented: dict[URIRef, tuple[URIRef, URIRef]],
    ) -> None:
        """Enqueue a set node neighbor and create relations between augmented entities.

        Args:
            dataset: Dataset to augment
            set_graph: Set knowledge graph
            neighbor: The set node neighbor to enqueue
            predicate: The predicate connecting current_node to neighbor
            direction: Direction of the edge ("out" or "in")
            current_node: Current node being processed
            src_aug: Augmented source entity of current node
            tgt_aug: Augmented target entity of current node
            current_depth: Current BFS depth
            queue: BFS queue
            visited: Set of visited nodes
            direct_enqueued: Set of nodes already enqueued in this iteration
            set_node_to_augmented: Mapping from set nodes to their augmented entities
        """
        # If neighbor was already expanded, create relations immediately
        if neighbor in set_node_to_augmented:
            neighbor_src_aug, neighbor_tgt_aug = set_node_to_augmented[neighbor]
            self._create_set_to_set_relations(
                dataset, set_graph, current_node, neighbor, src_aug, tgt_aug,
                neighbor_src_aug, neighbor_tgt_aug, predicate, direction
            )
            # Show the relation as an explicit RDF triple for clarity
            if direction == "out":
                self.logger.info("[PLM][Relation] Created: %s --%s--> %s", current_node, predicate, neighbor)
            else:
                self.logger.info("[PLM][Relation] Created: %s <--%s-- %s", current_node, predicate, neighbor)

        # Enqueue for future expansion if not visited yet
        next_depth = current_depth + 1
        if (
            next_depth <= self.max_depth
            and neighbor not in visited
            and neighbor not in direct_enqueued
        ):
            queue.append(ExpansionNode(uri=neighbor, depth=next_depth, node_type="set", parent=current_node))
            direct_enqueued.add(neighbor)

            self.logger.info("[PLM][Queue] Direct set neighbor enqueued")
            self.logger.info("    • current → %s", current_node)
            self.logger.info("    • neighbor → %s", neighbor)
            self.logger.info("    • depth → %d", next_depth)

    def _create_set_to_set_relations(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        current_node: URIRef,
        neighbor_node: URIRef,
        src_aug: URIRef,
        tgt_aug: URIRef,
        neighbor_src_aug: URIRef,
        neighbor_tgt_aug: URIRef,
        predicate: URIRef,
        direction: str,
    ) -> None:
        """Create relations between augmented entities from two set nodes.

        IMPORTANT: Only creates relations in graphs where they originally existed.

        Example:
            If in original graphs:
                kg_source: e1 --r1--> e3
                kg_target: e2 --r1--> e4
            Then in set graph: {e1,e2} --r1--> {e3,e4}
            Creates:
                e1_aug1 --r1--> e3_aug1 (in kg_source)
                e2_aug1 --r1--> e4_aug1 (in kg_target)

            But if kg_target doesn't have e2--r1-->e4, then we DON'T create e2_aug1--r1-->e4_aug1

        Args:
            dataset: Dataset to augment
            set_graph: Set knowledge graph (for checking relation origins)
            current_node: Current set node
            neighbor_node: Neighbor set node
            src_aug: Augmented source entity from current set node
            tgt_aug: Augmented target entity from current set node
            neighbor_src_aug: Augmented source entity from neighbor set node
            neighbor_tgt_aug: Augmented target entity from neighbor set node
            predicate: Predicate connecting the set nodes
            direction: Direction of the edge ("out" or "in")
        """
        # Get relation origins for the current node
        origins = set_graph.get_relation_origins(current_node)

        # Check if this specific relation exists in source/target graphs
        # The relation is stored as (predicate, neighbor, direction)
        relation_key = (predicate, neighbor_node, direction)

        exists_in_source = relation_key in origins.get("source", set())
        exists_in_target = relation_key in origins.get("target", set())

        # Create relations only in graphs where they originally existed
        if direction == "out":
            # current --predicate--> neighbor
            if exists_in_source:
                dataset.knowledge_graph_source.add((src_aug, predicate, neighbor_src_aug))
            if exists_in_target:
                dataset.knowledge_graph_target.add((tgt_aug, predicate, neighbor_tgt_aug))
        else:
            # neighbor --predicate--> current (incoming edge)
            if exists_in_source:
                dataset.knowledge_graph_source.add((neighbor_src_aug, predicate, src_aug))
            if exists_in_target:
                dataset.knowledge_graph_target.add((neighbor_tgt_aug, predicate, tgt_aug))

        # Log removed - too verbose and not essential for tracking progress

        if not exists_in_source and not exists_in_target:
            self.logger.warning(
                "[PLM][Relation] Skipped relation (not in original graphs): %s --%s(%s)--> %s",
                current_node, predicate, direction, neighbor_node
            )

    def _handle_non_set_neighbor(
        self,
        dataset: Dataset,
        set_graph: SetKnowledgeGraph,
        neighbor: URIRef,
        predicate: URIRef,
        direction: str,
        current_node: URIRef,
        src_aug: URIRef,
        tgt_aug: URIRef,
        current_depth: int,
        queue: Deque[ExpansionNode],
        visited: set[URIRef],
    ) -> None:
        """Handle a non-set node neighbor.

        Non-set nodes are connected to the augmented entities and checked for bridging
        to other set nodes.

        Args:
            dataset: Dataset to augment
            set_graph: Set knowledge graph
            neighbor: The non-set neighbor node
            predicate: Predicate connecting to the neighbor
            direction: Direction of the edge ("out" or "in")
            current_node: Current set node
            src_aug: Augmented source entity
            tgt_aug: Augmented target entity
            current_depth: Current BFS depth
            queue: BFS queue
            visited: Set of visited nodes
        """
        # Connect the non-set node to augmented entities
        self.neighbor_handler.connect_non_set_to_augmented(
            dataset, neighbor, predicate, direction, src_aug, tgt_aug
        )

        # Check for bridging to other set nodes (2-hop expansion)
        if current_depth + 2 <= self.max_depth:
            self._check_bridging(
                set_graph, neighbor, current_node, current_depth, queue, visited
            )

    def _check_bridging(
        self,
        set_graph: SetKnowledgeGraph,
        non_set_node: URIRef,
        current_node: URIRef,
        current_depth: int,
        queue: Deque[ExpansionNode],
        visited: set[URIRef],
    ) -> None:
        """Check if a non-set node bridges to other set nodes.

        When a set node is found from a non-set node, it starts a NEW expansion
        with depth=0 (acts as a new seed).

        Args:
            set_graph: Set knowledge graph
            non_set_node: The intermediate non-set node
            current_node: Current set node
            current_depth: Current BFS depth (ignored for bridged nodes)
            queue: BFS queue
            visited: Set of visited nodes
        """
        bridged_nodes = self.neighbor_handler.find_bridged_set_neighbors(
            set_graph, non_set_node, current_node
        )

        bridged_enqueued: set[URIRef] = set()

        for bridged_node in bridged_nodes:
            if bridged_node in visited or bridged_node in bridged_enqueued:
                continue

            # Depth restarts at 0 - this set node becomes a new seed
            queue.append(
                ExpansionNode(
                    uri=bridged_node,
                    depth=0,
                    node_type="set",
                    parent=non_set_node,
                )
            )
            bridged_enqueued.add(bridged_node)

            self.logger.info("[PLM][Queue] Bridged set node (new seed, depth=0)")
            self.logger.info("    • from set node → %s", current_node)
            self.logger.info("    • via non-set → %s", non_set_node)
            self.logger.info("    • to set node → %s (depth restarted)", bridged_node)

    # ------------------------------------------------------------------
    # Budget Calculation
    # ------------------------------------------------------------------
    def _compute_pair_budget(self, initial_pairs: int) -> Optional[int]:
        """Compute the maximum number of pairs to generate.

        Takes the minimum of ratio-based and config-based limits.

        Args:
            initial_pairs: Number of aligned pairs in input dataset

        Returns:
            Maximum number of pairs to generate, or None for unlimited
        """
        ratio_limit: Optional[int] = None
        if self.augmentation_ratio is not None:
            ratio_limit = max(1, math.ceil(initial_pairs * self.augmentation_ratio))

        config_limit = self.max_pairs_config

        if ratio_limit is None:
            return config_limit
        if config_limit is None:
            return ratio_limit

        return min(ratio_limit, config_limit)

    # ------------------------------------------------------------------
    # Logging Helpers
    # ------------------------------------------------------------------
    def _log_augmentation_start(self, initial_pairs: int, pair_budget: Optional[int]) -> None:
        """Log augmentation initialization parameters."""
        budget_display = pair_budget if pair_budget is not None else "unbounded"
        self.logger.info(
            "[PLM] Starting augmentation: budget=%s (initial=%d, ratio=%s, max_pairs=%s, seed=%d, derived=%s)",
            budget_display,
            initial_pairs,
            f"{self.augmentation_ratio:.2f}" if self.augmentation_ratio is not None else "-",
            self.max_pairs_config if self.max_pairs_config is not None else "-",
            self.seed,
            "enabled" if self.add_derived_predicate else "disabled",
        )

    def _log_expansion_summary(
        self, expanded_pairs: int, pair_budget: Optional[int], expansion_chain: List[str]
    ) -> None:
        """Log expansion summary."""
        budget_display = pair_budget if pair_budget is not None else "unbounded"
        self.logger.info(
            "[PLM] Expanded %d/%s set nodes (max_depth=%d)",
            expanded_pairs,
            budget_display,
            self.max_depth,
        )

        if expansion_chain:
            chain_repr = " -> ".join(expansion_chain[:10])
            if len(expansion_chain) > 10:
                chain_repr += f" ... (+{len(expansion_chain) - 10} more)"
            self.logger.info("[PLM] Expansion chain: %s", chain_repr)
        else:
            self.logger.info("[PLM] Expansion chain: (none)")

    def _log_node_connections(self, graph: SetKnowledgeGraph, node: URIRef) -> None:
        """Log all connections of a node for debugging."""
        outgoing: list[str] = []
        incoming: list[str] = []

        for direction, predicate, neighbor in self.neighbor_handler.iter_neighbors(graph, node):
            if isinstance(neighbor, Literal):
                value = neighbor.toPython()
                if not isinstance(value, str):
                    value = str(value)
                neighbor_repr = f'"{value}"'
            else:
                neighbor_repr = str(neighbor)

            if direction == "out":
                outgoing.append(f"{predicate} -> {neighbor_repr}")
            else:
                incoming.append(f"{predicate} <- {neighbor_repr}")

        self.logger.debug("[PLM][Node] %s", node)
        if outgoing:
            self.logger.debug("    • outgoing (%d)", len(outgoing))
            for rel in outgoing:
                self.logger.debug("        -> %s", rel)
        if incoming:
            self.logger.debug("    • incoming (%d)", len(incoming))
            for rel in incoming:
                self.logger.debug("        <- %s", rel)
