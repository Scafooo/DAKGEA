"""Evaluation utilities for BERT-INT interaction model."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from .dataset import InteractionDataset
from .model import InteractionMLP
from src.logger import get_logger

logger = get_logger(__name__)


class InteractionEvaluator:
    """Evaluator for interaction model using Hits@K and MRR metrics."""

    def __init__(
        self,
        model: InteractionMLP,
        dataset: InteractionDataset,
        device: torch.device,
        batch_size: int = 2048,
    ):
        """Initialize evaluator.

        Args:
            model: Trained interaction model
            dataset: Interaction dataset
            device: Device to run evaluation on
            batch_size: Batch size for scoring
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

    def _score_all_test_pairs(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Score all test candidate pairs.

        Returns:
            Tuple of (test_pairs, scores) where test_pairs is list of (e1, e2) tuples
            and scores is list of corresponding scores
        """
        self.model.eval()

        # Generate all test pairs from candidates
        test_ill_set = set(self.dataset.test_ill)
        test_pairs = []

        for e1, e2 in self.dataset.test_ill:
            candidates = self.dataset.get_test_candidates_for(e1)
            for cand_e2 in candidates:
                test_pairs.append((e1, cand_e2))

        logger.debug(f"Scoring {len(test_pairs)} test pairs")

        # Score all pairs in batches
        all_scores = []
        with torch.no_grad():
            for start_idx in range(0, len(test_pairs), self.batch_size):
                batch_pairs = test_pairs[start_idx:start_idx + self.batch_size]

                # Get feature indices for batch
                batch_feature_indices = [
                    self.dataset.pair_to_feature_idx[pair] for pair in batch_pairs
                ]

                # Get features and compute scores
                batch_features = self.dataset.features[batch_feature_indices].to(self.device)
                batch_scores = self.model(batch_features)
                all_scores.extend(batch_scores.cpu().tolist())

        return test_pairs, all_scores

    def evaluate(self, topk: int = 50) -> Dict[str, float]:
        """Evaluate the model on test set.

        Args:
            topk: Maximum K for Hits@K metric

        Returns:
            Dictionary containing evaluation metrics:
                - hits@1, hits@5, hits@10: Hit rate at K
                - mr: Mean rank
                - mrr: Mean reciprocal rank
        """
        # Score all test pairs
        test_pairs, scores = self._score_all_test_pairs()

        # Group scores by source entity
        test_ill_set = set(self.dataset.test_ill)
        entity_to_candidates: Dict[int, List[Tuple[int, float, int]]] = {}

        for i, (e1, e2) in enumerate(test_pairs):
            score = scores[i]
            label = 1 if (e1, e2) in test_ill_set else 0

            if e1 not in entity_to_candidates:
                entity_to_candidates[e1] = []

            entity_to_candidates[e1].append((e2, score, label))

        # Sort candidates by score (descending) for each entity
        for e1 in entity_to_candidates:
            entity_to_candidates[e1].sort(key=lambda x: x[1], reverse=True)

        # Compute metrics
        all_test_num = len(entity_to_candidates)
        result_labels = []

        for e1, candidates in entity_to_candidates.items():
            # Extract labels of top-K candidates
            label_list = [label for e2, score, label in candidates[:topk]]
            result_labels.append(label_list)

        result_labels = np.array(result_labels)

        # Count how many test entities have at least one correct match in top-K
        count_found = 0
        for i in range(len(result_labels)):
            if np.sum(result_labels[i]) > 0:
                count_found += 1

        # Compute Hits@K
        hits_per_position = result_labels.sum(axis=0).tolist()
        topk_metrics = {}
        for k in [1, 5, 10, 25, 50]:
            if k <= topk:
                cumulative_hits = sum(hits_per_position[:k])
                # Store as fraction (0-1) for consistency with Basic Unit
                topk_metrics[f"hits@{k}"] = cumulative_hits / all_test_num

        # Compute MR and MRR
        mr_sum = 0
        mrr_sum = 0

        for i in range(len(hits_per_position)):
            rank = i + 1
            num_correct_at_rank = hits_per_position[i]
            mr_sum += rank * num_correct_at_rank
            mrr_sum += (1.0 / rank) * num_correct_at_rank

        mr = mr_sum / all_test_num
        mrr = mrr_sum / all_test_num

        metrics = {
            "hits@1": topk_metrics.get("hits@1", 0.0),
            "hits@5": topk_metrics.get("hits@5", 0.0),
            "hits@10": topk_metrics.get("hits@10", 0.0),
            "mr": mr,
            "mrr": mrr,
            "found": count_found,
            "total": all_test_num,
        }

        # Add optional metrics if available
        if "hits@25" in topk_metrics:
            metrics["hits@25"] = topk_metrics["hits@25"]
        if "hits@50" in topk_metrics:
            metrics["hits@50"] = topk_metrics["hits@50"]

        return metrics

    def get_best_alignments(self) -> List[Tuple[int, int]]:
        """Get best alignment predictions (argmax) for all test entities.

        Returns:
            List of (e1, e2) pairs representing best predictions
        """
        # Score all test pairs
        test_pairs, scores = self._score_all_test_pairs()

        # Find best scoring e2 for each e1
        best_alignments: Dict[int, Tuple[int, float]] = {}

        for i, (e1, e2) in enumerate(test_pairs):
            score = scores[i]

            if e1 not in best_alignments or score > best_alignments[e1][1]:
                best_alignments[e1] = (e2, score)

        # Convert to list of pairs
        result = [(e1, e2) for e1, (e2, score) in best_alignments.items()]
        return result
