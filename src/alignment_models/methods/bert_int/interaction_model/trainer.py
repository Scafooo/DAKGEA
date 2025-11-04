"""Training utilities for BERT-INT interaction model."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .dataset import InteractionDataset
from .model import InteractionMLP
from .evaluator import InteractionEvaluator
from src.logger import get_logger, get_structured_logger

logger = get_logger(__name__)
slogger = get_structured_logger(__name__)


class TrainingBatchGenerator:
    """Generate training batches with negative sampling."""

    def __init__(
        self,
        dataset: InteractionDataset,
        neg_num: int = 5,
        batch_size: int = 256,
        seed: Optional[int] = None,
    ):
        """Initialize batch generator.

        Args:
            dataset: Interaction dataset
            neg_num: Number of negative samples per positive sample
            batch_size: Batch size
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.neg_num = neg_num
        self.batch_size = batch_size
        self.iter_count = 0

        if seed is not None:
            np.random.seed(seed)

        # Convert train candidates to numpy arrays for efficient sampling
        self.train_candidates_np = {}
        for e1, candidates in dataset.train_candidates.items():
            self.train_candidates_np[e1] = np.array(candidates)

        # Generate initial training pairs
        self.train_pair_indices, self.batch_num = self._generate_train_pairs()

        logger.info(f"Training batch generator initialized:")
        logger.info(f"  Training ILL pairs: {len(dataset.train_ill)}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Negative samples per positive: {neg_num}")
        logger.info(f"  Total training pairs: {len(self.train_pair_indices)}")
        logger.info(f"  Number of batches per epoch: {self.batch_num}")

    def _generate_train_pairs(self) -> Tuple[List[Tuple[int, int, int, int]], int]:
        """Generate training pairs with negative sampling.

        Returns:
            Tuple of (train_pair_indices, batch_num) where train_pair_indices is a
            list of (pos_e1, pos_e2, neg_e1, neg_e2) tuples
        """
        train_pairs = []

        for pos_e1, pos_e2 in self.dataset.train_ill:
            # Sample negative examples from candidates
            if pos_e1 not in self.train_candidates_np:
                continue

            candidates = self.train_candidates_np[pos_e1]
            if len(candidates) == 0:
                continue

            # Randomly sample neg_num negative entities
            neg_indices = np.random.randint(len(candidates), size=self.neg_num)
            neg_e2_list = candidates[neg_indices].tolist()

            for neg_e2 in neg_e2_list:
                # Skip if negative is actually the positive
                if neg_e2 == pos_e2:
                    continue

                neg_e1 = pos_e1  # Negative pair shares the same e1
                train_pairs.append((pos_e1, pos_e2, neg_e1, neg_e2))

        # Shuffle training pairs
        np.random.shuffle(train_pairs)
        np.random.shuffle(train_pairs)
        np.random.shuffle(train_pairs)

        batch_num = int(np.ceil(len(train_pairs) / self.batch_size))
        return train_pairs, batch_num

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[List[int], List[int]]:
        """Get next batch of positive and negative feature indices.

        Returns:
            Tuple of (pos_feature_indices, neg_feature_indices)
        """
        if self.iter_count < self.batch_num:
            batch_idx = self.iter_count
            self.iter_count += 1

            # Get batch of training pairs
            start_idx = batch_idx * self.batch_size
            end_idx = (batch_idx + 1) * self.batch_size
            batch_pairs = self.train_pair_indices[start_idx:end_idx]

            # Extract positive and negative pairs
            pos_pairs = [(pos_e1, pos_e2) for pos_e1, pos_e2, neg_e1, neg_e2 in batch_pairs]
            neg_pairs = [(neg_e1, neg_e2) for pos_e1, pos_e2, neg_e1, neg_e2 in batch_pairs]

            # Get feature indices
            pos_feature_indices = [
                self.dataset.pair_to_feature_idx[pair] for pair in pos_pairs
            ]
            neg_feature_indices = [
                self.dataset.pair_to_feature_idx[pair] for pair in neg_pairs
            ]

            return pos_feature_indices, neg_feature_indices
        else:
            # Reset for next epoch
            self.iter_count = 0
            self.train_pair_indices, self.batch_num = self._generate_train_pairs()
            raise StopIteration()


class InteractionTrainer:
    """Trainer for BERT-INT interaction model."""

    def __init__(
        self,
        model: InteractionMLP,
        dataset: InteractionDataset,
        device: torch.device,
        learning_rate: float = 0.001,
        margin: float = 1.0,
        neg_num: int = 5,
        batch_size: int = 256,
        seed: Optional[int] = None,
    ):
        """Initialize trainer.

        Args:
            model: Interaction MLP model
            dataset: Interaction dataset
            device: Device to train on
            learning_rate: Learning rate for optimizer
            margin: Margin for ranking loss
            neg_num: Number of negative samples per positive
            batch_size: Batch size
            seed: Random seed for reproducibility
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.seed = seed

        # Setup optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction="mean")

        # Setup batch generator
        self.batch_generator = TrainingBatchGenerator(
            dataset, neg_num=neg_num, batch_size=batch_size, seed=seed
        )

        # Setup evaluator
        self.evaluator = InteractionEvaluator(model, dataset, device)

        logger.info(f"Trainer initialized:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Margin: {margin}")
        logger.info(f"  Optimizer: Adam")

    def train_one_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for pos_feature_indices, neg_feature_indices in self.batch_generator:
            self.optimizer.zero_grad()

            # Get features
            pos_features = self.dataset.features[pos_feature_indices].to(self.device)
            neg_features = self.dataset.features[neg_feature_indices].to(self.device)

            # Forward pass
            pos_scores = self.model(pos_features)
            neg_scores = self.model(neg_features)

            # Reshape for loss computation
            pos_scores = pos_scores.unsqueeze(-1)  # [B, 1]
            neg_scores = neg_scores.unsqueeze(-1)  # [B, 1]

            # Compute ranking loss (pos_scores should be higher than neg_scores)
            label_y = torch.ones_like(pos_scores).to(self.device)
            loss = self.criterion(pos_scores, neg_scores, label_y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * pos_scores.size(0)
            num_batches += 1

        avg_loss = total_loss / (num_batches * self.batch_generator.batch_size)
        return avg_loss

    def train(
        self,
        epochs: int,
        eval_every: int = 10,
        save_path: Optional[Path] = None,
    ) -> Dict:
        """Train the model for multiple epochs.

        Args:
            epochs: Number of epochs to train
            eval_every: Evaluate every N epochs
            save_path: Path to save the trained model

        Returns:
            Dictionary containing training results
        """
        slogger.section("Interaction Model Training")

        best_hits_at_1 = 0.0
        best_epoch = -1
        training_history = []

        for epoch in range(epochs):
            start_time = time.time()

            # Train one epoch
            epoch_loss = self.train_one_epoch()
            epoch_time = time.time() - start_time

            # Log training progress
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f} - Time: {epoch_time:.2f}s"
            )

            # Evaluate periodically
            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                eval_start = time.time()
                eval_results = self.evaluator.evaluate(topk=50)
                eval_time = time.time() - eval_start

                hits_at_1 = eval_results["hits@1"]
                hits_at_5 = eval_results["hits@5"]
                hits_at_10 = eval_results["hits@10"]
                mrr = eval_results["mrr"]

                slogger.subsection(f"Evaluation at Epoch {epoch + 1}")
                slogger.table("Metrics", {
                    "Hits@1": f"{hits_at_1:.2f}%",
                    "Hits@5": f"{hits_at_5:.2f}%",
                    "Hits@10": f"{hits_at_10:.2f}%",
                    "MRR": f"{mrr:.4f}",
                    "Eval Time": f"{eval_time:.2f}s",
                })

                # Track best model
                if hits_at_1 > best_hits_at_1:
                    best_hits_at_1 = hits_at_1
                    best_epoch = epoch + 1
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                        logger.info(f"Saved best model to {save_path}")

                training_history.append({
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "hits@1": hits_at_1,
                    "hits@5": hits_at_5,
                    "hits@10": hits_at_10,
                    "mrr": mrr,
                })

        slogger.success(f"Training completed! Best Hits@1: {best_hits_at_1:.2f}% at epoch {best_epoch}")

        return {
            "best_hits@1": best_hits_at_1,
            "best_epoch": best_epoch,
            "training_history": training_history,
        }
