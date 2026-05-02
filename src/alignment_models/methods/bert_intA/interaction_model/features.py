"""Feature extraction for BERT-INT interaction model.

This module implements the Dual Aggregation mechanism and various view-based
interaction feature extractors (neighbor-view, attribute-view, description-view).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from src.logger import get_logger

logger = get_logger(__name__)


class DualAggregation:
    """Dual Aggregation mechanism using Gaussian kernels.

    This implements the core interaction feature extraction using multiple
    Gaussian kernels with different mu and sigma values to capture multi-scale
    similarity patterns between two sets of elements.
    """

    def __init__(self, kernel_num: int = 21):
        """Initialize Dual Aggregation with specified number of kernels.

        Args:
            kernel_num: Number of Gaussian kernels to use (default: 21)
        """
        self.kernel_num = kernel_num
        self.mus = self._kernel_mus(kernel_num)
        self.sigmas = self._kernel_sigmas(kernel_num)

    @staticmethod
    def _kernel_mus(kernel_num: int) -> torch.Tensor:
        """Generate mu values for Gaussian kernels.

        Linearly spaced values from -1 to 1.
        """
        mu_step = 2.0 / (kernel_num - 1)
        return torch.FloatTensor([1.0 - mu_step * i for i in range(kernel_num)])

    @staticmethod
    def _kernel_sigmas(kernel_num: int) -> torch.Tensor:
        """Generate sigma values for Gaussian kernels.

        All sigmas are set to 0.1 (following original implementation).
        """
        return torch.FloatTensor([0.1] * kernel_num)

    def compute_features(
        self,
        similarity_matrix: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Compute dual aggregation features from similarity matrix.

        Args:
            similarity_matrix: Similarity matrix of shape [B, n1, n2]
                where B is batch size, n1 is number of elements in set 1,
                n2 is number of elements in set 2
            mask1: Mask for set 1 of shape [B, n1, 1] (1 for valid, 0 for padding)
            mask2: Mask for set 2 of shape [B, n2, 1] (1 for valid, 0 for padding)
            device: Device to perform computation on

        Returns:
            Features tensor of shape [B, kernel_num * 2] where the first kernel_num
            features are from sum-pooling and the next kernel_num are from max-pooling
        """
        if device is None:
            device = similarity_matrix.device

        mus = self.mus.to(device).view(1, 1, -1)  # [1, 1, K]
        sigmas = self.sigmas.to(device).view(1, 1, -1)  # [1, 1, K]

        # Apply masks to similarity matrix
        # Set masked positions to very low value so they don't contribute
        sim_masked = similarity_matrix.unsqueeze(-1)  # [B, n1, n2, 1]
        mask1_expanded = mask1.unsqueeze(2)  # [B, n1, 1, 1]
        mask2_expanded = mask2.unsqueeze(1)  # [B, 1, n2, 1]

        # Apply both masks
        valid_mask = mask1_expanded * mask2_expanded  # [B, n1, n2, 1]
        sim_masked = sim_masked * valid_mask + (1 - valid_mask) * (-1e9)

        # Compute Gaussian kernel responses for each similarity value
        # kernel_response[b, i, j, k] = exp(-(sim[b,i,j] - mu[k])^2 / sigma[k]^2)
        diff = sim_masked.unsqueeze(-1) - mus.unsqueeze(0).unsqueeze(0)  # [B, n1, n2, 1, K]
        kernel_response = torch.exp(-(diff ** 2) / (sigmas.unsqueeze(0).unsqueeze(0) ** 2))
        kernel_response = kernel_response.squeeze(3)  # [B, n1, n2, K]

        # Apply mask to kernel responses
        kernel_response = kernel_response * valid_mask

        # Sum pooling: aggregate across n2 dimension
        sum_pooled = torch.sum(kernel_response, dim=2)  # [B, n1, K]
        # Then aggregate across n1 dimension
        sum_features = torch.sum(sum_pooled, dim=1)  # [B, K]

        # Max pooling: aggregate across n2 dimension
        max_pooled, _ = torch.max(kernel_response, dim=2)  # [B, n1, K]
        # Then aggregate across n1 dimension
        max_features, _ = torch.max(max_pooled, dim=1)  # [B, K]

        # Concatenate sum and max features
        features = torch.cat([sum_features, max_features], dim=1)  # [B, K*2]

        return features


class NeighborViewFeatureExtractor:
    """Extract interaction features based on entity neighborhoods."""

    def __init__(
        self,
        kernel_num: int = 21,
        max_neighbors: int = 50,
        device: Optional[torch.device] = None,
    ):
        """Initialize neighbor-view feature extractor.

        Args:
            kernel_num: Number of Gaussian kernels for dual aggregation
            max_neighbors: Maximum number of neighbors to consider
            device: Device to perform computation on
        """
        self.dual_aggregation = DualAggregation(kernel_num)
        self.max_neighbors = max_neighbors
        self.device = device or torch.device("cpu")

    def build_neighbor_dict(
        self,
        triples: List[Tuple[int, int, int]],
        pad_id: int,
    ) -> Dict[int, List[int]]:
        """Build dictionary mapping entities to their neighbors.

        Args:
            triples: List of (subject, predicate, object) triples
            pad_id: ID to use for padding

        Returns:
            Dictionary mapping entity ID to list of neighbor IDs (padded to max_neighbors)
        """
        # Collect all neighbors for each entity
        entity_neighbors: Dict[int, List[int]] = {}

        for subj, pred, obj in triples:
            # Subject's neighbor is object
            if subj not in entity_neighbors:
                entity_neighbors[subj] = []
            entity_neighbors[subj].append(obj)

            # Object's neighbor is subject (bidirectional)
            if obj not in entity_neighbors:
                entity_neighbors[obj] = []
            entity_neighbors[obj].append(subj)

        # Pad or truncate to max_neighbors
        neighbor_dict = {}
        for entity, neighbors in entity_neighbors.items():
            # Remove duplicates
            neighbors = list(set(neighbors))

            # Truncate if too many
            if len(neighbors) > self.max_neighbors:
                neighbors = neighbors[:self.max_neighbors]

            # Pad if too few
            while len(neighbors) < self.max_neighbors:
                neighbors.append(pad_id)

            neighbor_dict[entity] = neighbors

        return neighbor_dict

    def extract_features(
        self,
        entity_pairs: List[Tuple[int, int]],
        entity_embeddings: np.ndarray,
        neighbor_dict: Dict[int, List[int]],
        pad_id: int,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Extract neighbor-view interaction features for entity pairs.

        Args:
            entity_pairs: List of (entity1, entity2) pairs
            entity_embeddings: Entity embeddings array of shape [num_entities, embed_dim]
            neighbor_dict: Dictionary mapping entity ID to neighbor IDs
            pad_id: Padding entity ID
            batch_size: Batch size for processing

        Returns:
            Features array of shape [num_pairs, kernel_num * 2]
        """
        all_features = []

        # Convert embeddings to tensor and normalize
        emb_tensor = torch.FloatTensor(entity_embeddings).to(self.device)
        emb_tensor = F.normalize(emb_tensor, p=2, dim=-1)

        logger.info(f"Extracting neighbor-view features for {len(entity_pairs)} pairs")

        for start_idx in range(0, len(entity_pairs), batch_size):
            batch_pairs = entity_pairs[start_idx:start_idx + batch_size]

            e1_list = [e1 for e1, e2 in batch_pairs]
            e2_list = [e2 for e1, e2 in batch_pairs]

            # Get neighbors for each entity
            e1_neighbors = [neighbor_dict.get(e1, [pad_id] * self.max_neighbors) for e1 in e1_list]
            e2_neighbors = [neighbor_dict.get(e2, [pad_id] * self.max_neighbors) for e2 in e2_list]

            # Create masks (1 for valid, 0 for padding)
            e1_masks = torch.FloatTensor(
                [[1.0 if n != pad_id else 0.0 for n in neighbors] for neighbors in e1_neighbors]
            ).to(self.device).unsqueeze(-1)  # [B, n1, 1]

            e2_masks = torch.FloatTensor(
                [[1.0 if n != pad_id else 0.0 for n in neighbors] for neighbors in e2_neighbors]
            ).to(self.device).unsqueeze(-1)  # [B, n2, 1]

            # Get neighbor embeddings
            e1_neighbor_ids = torch.LongTensor(e1_neighbors).to(self.device)  # [B, n1]
            e2_neighbor_ids = torch.LongTensor(e2_neighbors).to(self.device)  # [B, n2]

            e1_neighbor_emb = emb_tensor[e1_neighbor_ids]  # [B, n1, embed_dim]
            e2_neighbor_emb = emb_tensor[e2_neighbor_ids]  # [B, n2, embed_dim]

            # Compute similarity matrix
            sim_matrix = torch.bmm(e1_neighbor_emb, e2_neighbor_emb.transpose(1, 2))  # [B, n1, n2]

            # Apply dual aggregation
            features = self.dual_aggregation.compute_features(
                sim_matrix, e1_masks, e2_masks, device=self.device
            )

            all_features.append(features.cpu().numpy())

        result = np.vstack(all_features)
        logger.info(f"Neighbor-view features shape: {result.shape}")
        return result


class AttributeViewFeatureExtractor:
    """Extract interaction features based on entity attribute values."""

    def __init__(
        self,
        kernel_num: int = 21,
        max_values: int = 20,
        device: Optional[torch.device] = None,
    ):
        """Initialize attribute-view feature extractor.

        Args:
            kernel_num: Number of Gaussian kernels for dual aggregation
            max_values: Maximum number of attribute values to consider
            device: Device to perform computation on
        """
        self.dual_aggregation = DualAggregation(kernel_num)
        self.max_values = max_values
        self.device = device or torch.device("cpu")

    def build_entity_to_values(
        self,
        attribute_triples: List[Tuple[int, int, str]],
        value_to_index: Dict[str, int],
        pad_value_id: int,
        entity_ids: Iterable[int],
    ) -> Dict[int, List[int]]:
        """Build dictionary mapping entities to their attribute value indices.

        Args:
            attribute_triples: List of (entity, attribute, value) triples
            value_to_index: Dictionary mapping value strings to indices
            pad_value_id: ID to use for padding
            entity_ids: Set of entity IDs to consider

        Returns:
            Dictionary mapping entity ID to list of value indices (padded to max_values)
        """
        # Collect all attribute values for each entity
        entity_values: Dict[int, List[int]] = {eid: [] for eid in entity_ids}

        for entity, attribute, value in attribute_triples:
            if entity in entity_values and value in value_to_index:
                value_idx = value_to_index[value]
                entity_values[entity].append(value_idx)

        # Pad or truncate to max_values
        entity_to_value_ids = {}
        for entity, values in entity_values.items():
            # Remove duplicates
            values = list(set(values))

            # Truncate if too many
            if len(values) > self.max_values:
                values = values[:self.max_values]

            # Pad if too few
            while len(values) < self.max_values:
                values.append(pad_value_id)

            entity_to_value_ids[entity] = values

        return entity_to_value_ids

    def extract_features(
        self,
        entity_pairs: List[Tuple[int, int]],
        value_embeddings: np.ndarray,
        entity_to_value_ids: Dict[int, List[int]],
        pad_value_id: int,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Extract attribute-view interaction features for entity pairs.

        Args:
            entity_pairs: List of (entity1, entity2) pairs
            value_embeddings: Value embeddings array of shape [num_values, embed_dim]
            entity_to_value_ids: Dictionary mapping entity ID to value indices
            pad_value_id: Padding value ID
            batch_size: Batch size for processing

        Returns:
            Features array of shape [num_pairs, kernel_num * 2]
        """
        all_features = []

        # Convert embeddings to tensor and normalize
        value_tensor = torch.FloatTensor(value_embeddings).to(self.device)
        value_tensor = F.normalize(value_tensor, p=2, dim=-1)

        logger.info(f"Extracting attribute-view features for {len(entity_pairs)} pairs")

        for start_idx in range(0, len(entity_pairs), batch_size):
            batch_pairs = entity_pairs[start_idx:start_idx + batch_size]

            e1_list = [e1 for e1, e2 in batch_pairs]
            e2_list = [e2 for e1, e2 in batch_pairs]

            # Get attribute values for each entity
            e1_values = [entity_to_value_ids.get(e1, [pad_value_id] * self.max_values) for e1 in e1_list]
            e2_values = [entity_to_value_ids.get(e2, [pad_value_id] * self.max_values) for e2 in e2_list]

            # Create masks (1 for valid, 0 for padding)
            e1_masks = torch.FloatTensor(
                [[1.0 if v != pad_value_id else 0.0 for v in values] for values in e1_values]
            ).to(self.device).unsqueeze(-1)  # [B, n1, 1]

            e2_masks = torch.FloatTensor(
                [[1.0 if v != pad_value_id else 0.0 for v in values] for values in e2_values]
            ).to(self.device).unsqueeze(-1)  # [B, n2, 1]

            # Get value embeddings
            e1_value_ids = torch.LongTensor(e1_values).to(self.device)  # [B, n1]
            e2_value_ids = torch.LongTensor(e2_values).to(self.device)  # [B, n2]

            e1_value_emb = value_tensor[e1_value_ids]  # [B, n1, embed_dim]
            e2_value_emb = value_tensor[e2_value_ids]  # [B, n2, embed_dim]

            # Compute similarity matrix
            sim_matrix = torch.bmm(e1_value_emb, e2_value_emb.transpose(1, 2))  # [B, n1, n2]

            # Apply dual aggregation
            features = self.dual_aggregation.compute_features(
                sim_matrix, e1_masks, e2_masks, device=self.device
            )

            all_features.append(features.cpu().numpy())

        result = np.vstack(all_features)
        logger.info(f"Attribute-view features shape: {result.shape}")
        return result


class DescriptionViewFeatureExtractor:
    """Extract interaction features based on entity descriptions/names.

    This simply computes cosine similarity between entity embeddings.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize description-view feature extractor.

        Args:
            device: Device to perform computation on
        """
        self.device = device or torch.device("cpu")

    def extract_features(
        self,
        entity_pairs: List[Tuple[int, int]],
        entity_embeddings: np.ndarray,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Extract description-view interaction features for entity pairs.

        Args:
            entity_pairs: List of (entity1, entity2) pairs
            entity_embeddings: Entity embeddings array of shape [num_entities, embed_dim]
            batch_size: Batch size for processing

        Returns:
            Features array of shape [num_pairs, 1] containing cosine similarities
        """
        all_features = []

        # Convert embeddings to tensor
        emb_tensor = torch.FloatTensor(entity_embeddings).to(self.device)

        logger.info(f"Extracting description-view features for {len(entity_pairs)} pairs")

        for start_idx in range(0, len(entity_pairs), batch_size):
            batch_pairs = entity_pairs[start_idx:start_idx + batch_size]

            e1_ids = torch.LongTensor([e1 for e1, e2 in batch_pairs]).to(self.device)
            e2_ids = torch.LongTensor([e2 for e1, e2 in batch_pairs]).to(self.device)

            e1_embs = emb_tensor[e1_ids]  # [B, embed_dim]
            e2_embs = emb_tensor[e2_ids]  # [B, embed_dim]

            # Compute cosine similarity
            cos_sim = F.cosine_similarity(e1_embs, e2_embs, dim=1)  # [B]
            cos_sim = cos_sim.unsqueeze(-1)  # [B, 1]

            all_features.append(cos_sim.cpu().numpy())

        result = np.vstack(all_features)
        logger.info(f"Description-view features shape: {result.shape}")
        return result
