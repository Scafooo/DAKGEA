"""RREA (Relational Reflection Entity Alignment) model implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.alignment_models.methods.RREA import load_rrea_config
from src.alignment_models.methods.RREA.layer import RREAEncoder
from src.alignment_models.methods.RREA.csls import eval_alignment_batched
from src.alignment_models.methods.RREA.utils import build_matrices_from_triples
from src.alignment_models.methods.RREA.data_loader import load_rrea_data
from src.alignment_models.registry import MODEL_REGISTRY
from src.core.dataset import Dataset
from src.logger import get_logger

logger = get_logger(__name__)

Pair = Tuple[int, int]


def margin_loss(
    embeddings: torch.Tensor,
    pos_pairs: torch.Tensor,
    neg_pairs: torch.Tensor,
    gamma: float = 3.0,
) -> torch.Tensor:
    """Compute margin-based ranking loss.

    Args:
        embeddings: Entity embeddings [num_entities, dim]
        pos_pairs: Positive pairs [batch_size, 2]
        neg_pairs: Negative pairs [batch_size, neg_num, 2]
        gamma: Margin parameter

    Returns:
        Loss value
    """
    # Positive pair distances
    pos_src = embeddings[pos_pairs[:, 0]]
    pos_tgt = embeddings[pos_pairs[:, 1]]
    pos_dist = torch.sum((pos_src - pos_tgt) ** 2, dim=1)

    # Negative pair distances
    neg_src = embeddings[neg_pairs[:, :, 0]]  # [batch_size, neg_num, dim]
    neg_tgt = embeddings[neg_pairs[:, :, 1]]
    neg_dist = torch.sum((neg_src - neg_tgt) ** 2, dim=2)  # [batch_size, neg_num]

    # Margin loss: max(0, gamma + pos_dist - neg_dist)
    loss = torch.relu(gamma + pos_dist.unsqueeze(1) - neg_dist)
    loss = torch.mean(loss)

    return loss


@MODEL_REGISTRY.register("rrea")
class RREAAlignment:
    """RREA: Relational Reflection Entity Alignment model."""

    def __init__(self, stage_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RREA model.

        Args:
            stage_config: Stage configuration from experiment YAML
        """
        self.stage_config = stage_config or {}
        self.config = load_rrea_config(overrides=self.stage_config.get("model", {}))
        self.model_cfg = self.config["model"]

        # Determine paths from lineage
        lineage = self.stage_config.get("lineage", {})
        self.variant = lineage.get("variant", "reduced")
        artifact_root = Path(lineage.get("artifact_root", "results/artifact"))
        artifact_root.mkdir(parents=True, exist_ok=True)
        self.artifact_root = artifact_root
        self.evaluation_root = Path(
            lineage.get("evaluation_root", (artifact_root / "evaluation"))
        )

        # Create checkpoint directory
        self.checkpoint_dir = artifact_root / "rrea" / self.variant
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device configuration
        self.device = self.model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Model will be initialized during training
        self.model: Optional[RREAEncoder] = None

        logger.info(
            "[RREA] Initialized model (device=%s, embedding_dim=%d, depth=%d, epochs=%d)",
            self.device,
            self.model_cfg.get("embedding_dim"),
            self.model_cfg.get("depth"),
            self.model_cfg.get("epochs"),
        )
        logger.info(f"[RREA] Checkpoint directory: {self.checkpoint_dir}")

    def evaluate(self, dataset_reduced: Dataset, dataset_augmented: Dataset) -> Dict[str, Any]:
        """Train and evaluate RREA model.

        Args:
            dataset_reduced: Original (non-augmented) dataset
            dataset_augmented: Augmented dataset to use for training

        Returns:
            Dictionary with evaluation results
        """
        logger.info("=" * 80)
        logger.info("[STEP] RREA: RELATIONAL REFLECTION ENTITY ALIGNMENT")
        logger.info("=" * 80)

        # Load pre-processed data from dataset workspace
        lineage = self.stage_config.get("lineage", {})
        dataset_root = lineage.get("dataset_workspace")

        if not dataset_root:
            raise ValueError(
                "dataset_workspace not found in lineage. "
                "Make sure to use writer: openea in experiment config"
            )

        logger.info(f"[RREA] Loading dataset from {dataset_root}")
        data_bundle = load_rrea_data(dataset_root)

        # Build graph structures from pre-processed triples
        logger.info("[STEP] Building graph structures...")
        num_entities = len(data_bundle.entity2id)
        num_relations = len(data_bundle.relation2id)
        adj_matrix, r_index, r_val, adj_features, rel_features, adj_indices_np = build_matrices_from_triples(
            data_bundle.triples,
            num_entities,
            num_relations,
        )

        # Split alignments into train/test
        train_ratio = self.model_cfg.get("train_ratio", 0.3)
        all_pairs = data_bundle.aligned_pairs.copy()
        np.random.shuffle(all_pairs)

        n_train = int(len(all_pairs) * train_ratio)
        train_pairs = all_pairs[:n_train]
        test_pairs = all_pairs[n_train:]

        logger.info(f"[RREA] Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")
        kg2_offset = data_bundle.kg2_offset

        # Convert to PyTorch tensors
        adj_indices = torch.LongTensor(adj_indices_np).to(self.device)
        r_index_tensor = torch.LongTensor(r_index).to(self.device)
        r_val_tensor = torch.FloatTensor(r_val).to(self.device)

        # Initialize model
        node_size = adj_matrix.shape[0]
        rel_size = rel_features.shape[1]
        triple_size = len(r_index)

        self.model = RREAEncoder(
            node_size=node_size,
            rel_size=rel_size,
            triple_size=triple_size,
            embedding_dim=self.model_cfg["embedding_dim"],
            depth=self.model_cfg["depth"],
            attn_heads=self.model_cfg["attn_heads"],
            dropout=self.model_cfg["dropout_rate"],
        ).to(self.device)

        logger.info(f"[RREA] Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Train model
        logger.info("[STEP] Starting training...")
        train_results = self._train(
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            adj_indices=adj_indices,
            r_index=r_index_tensor,
            r_val=r_val_tensor,
            kg2_offset=kg2_offset,
        )

        # Load best model for final evaluation
        best_checkpoint_path = self.checkpoint_dir / "rrea_best.pt"
        if best_checkpoint_path.exists():
            logger.info(f"[RREA] Loading best model from {best_checkpoint_path}")
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"[RREA] Best model loaded (epoch {checkpoint['epoch']})")
        else:
            logger.warning("[RREA] Best model checkpoint not found, using final model")

        # Final evaluation
        logger.info("=" * 80)
        logger.info("[STEP] Final evaluation on test set with best model...")
        final_results = eval_alignment_batched(
            model=self.model,
            test_pairs=test_pairs,
            adj_indices=adj_indices,
            sparse_indices=r_index_tensor,
            sparse_val=r_val_tensor,
            kg2_offset=kg2_offset,
            top_k=tuple(self.model_cfg["eval_top_k"]),
            csls_k=self.model_cfg["csls_k"],
            use_csls=self.model_cfg["use_csls"],
            device=self.device,
        )

        logger.info("=" * 80)
        logger.info("[SUCCESS] RREA evaluation completed")
        logger.info("=" * 80)

        # Return results - framework will save them automatically to evaluation/results.json
        return final_results

    def _train(
        self,
        train_pairs: np.ndarray,
        test_pairs: np.ndarray,
        adj_indices: torch.Tensor,
        r_index: torch.Tensor,
        r_val: torch.Tensor,
        kg2_offset: int,
    ) -> Dict[str, Any]:
        """Train RREA model.

        Args:
            train_pairs: Training alignment pairs
            test_pairs: Test alignment pairs
            adj_indices: Adjacency indices
            r_index: Relation indices
            r_val: Relation values
            kg2_offset: Offset for KG2 entity IDs

        Returns:
            Training results
        """
        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.model_cfg["learning_rate"],
            weight_decay=self.model_cfg["weight_decay"],
        )

        # Training parameters
        epochs = self.model_cfg["epochs"]
        batch_size = self.model_cfg["batch_size"]
        gamma = self.model_cfg["gamma"]
        neg_num = self.model_cfg["neg_num"]
        eval_freq = self.model_cfg["eval_frequency"]

        best_hits1 = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()

            # Generate negative samples
            neg_pairs = self._generate_negatives(train_pairs, kg2_offset, neg_num)

            # Create batches
            n_batches = (len(train_pairs) + batch_size - 1) // batch_size
            total_loss = 0.0

            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(train_pairs))

                pos_batch = torch.LongTensor(train_pairs[start:end]).to(self.device)
                neg_batch = torch.LongTensor(neg_pairs[start:end]).to(self.device)

                # Forward pass
                embeddings = self.model(adj_indices, r_index, r_val)

                # Compute loss
                loss = margin_loss(embeddings, pos_batch, neg_batch, gamma)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.model_cfg["max_grad_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.model_cfg["max_grad_norm"]
                    )

                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / n_batches

            # Evaluation
            if (epoch + 1) % eval_freq == 0:
                logger.info(f"[PROGRESS] Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

                metrics = eval_alignment_batched(
                    model=self.model,
                    test_pairs=test_pairs,
                    adj_indices=adj_indices,
                    sparse_indices=r_index,
                    sparse_val=r_val,
                    kg2_offset=kg2_offset,
                    top_k=tuple(self.model_cfg["eval_top_k"]),
                    csls_k=self.model_cfg["csls_k"],
                    use_csls=self.model_cfg["use_csls"],
                    device=self.device,
                )

                hits1 = metrics["hits@1"]

                # Early stopping
                if self.model_cfg["early_stop"]:
                    if hits1 > best_hits1:
                        best_hits1 = hits1
                        patience_counter = 0
                        # Save best model
                        self._save_checkpoint(epoch, "best")
                        logger.info(f"[SUCCESS] New best model saved (Hits@1: {hits1:.2f}%)")
                    else:
                        patience_counter += 1

                    if patience_counter >= self.model_cfg["early_stop_patience"]:
                        logger.info(f"[IMPORTANT] Early stopping at epoch {epoch + 1}")
                        break

        # Save final model
        self._save_checkpoint(epochs - 1, "final")
        logger.info(f"[SUCCESS] Training completed (Best Hits@1: {best_hits1:.2f}%)")

        return {"best_hits@1": best_hits1}

    def _generate_negatives(
        self,
        pos_pairs: np.ndarray,
        kg2_offset: int,
        neg_num: int
    ) -> np.ndarray:
        """Generate negative samples for training.

        Args:
            pos_pairs: Positive alignment pairs
            kg2_offset: Offset for KG2 entity IDs
            neg_num: Number of negative samples per positive

        Returns:
            Negative pairs [batch_size, neg_num, 2]
        """
        batch_size = len(pos_pairs)
        neg_pairs = np.zeros((batch_size, neg_num, 2), dtype=np.int64)

        kg1_size = kg2_offset
        kg2_size = self.model.node_size - kg2_offset

        for i in range(batch_size):
            src, tgt = pos_pairs[i]

            for j in range(neg_num):
                # Randomly corrupt source or target
                if np.random.rand() < 0.5:
                    # Corrupt source
                    neg_src = np.random.randint(0, kg1_size)
                    neg_pairs[i, j] = [neg_src, tgt]
                else:
                    # Corrupt target
                    neg_tgt = np.random.randint(kg2_offset, kg2_offset + kg2_size)
                    neg_pairs[i, j] = [src, neg_tgt]

        return neg_pairs

    def _save_checkpoint(self, epoch: int, name: str) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch
            name: Checkpoint name
        """
        checkpoint_path = self.checkpoint_dir / f"rrea_{name}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }, checkpoint_path)
        logger.info(f"[RREA] Checkpoint saved to {checkpoint_path}")
