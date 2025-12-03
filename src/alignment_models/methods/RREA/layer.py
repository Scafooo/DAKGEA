"""Graph Attention layers for RREA model (PyTorch implementation)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.logger import get_logger

logger = get_logger(__name__)


class NR_GraphAttention(nn.Module):
    """Neighborhood-aware Relational Graph Attention layer.

    This implements the graph attention mechanism with relational reflection
    as described in the RREA paper.

    Args:
        node_size: Number of nodes in the graph
        rel_size: Number of relations
        triple_size: Number of triples
        in_features: Input feature dimension
        out_features: Output feature dimension
        depth: Number of attention layers
        attn_heads: Number of attention heads
        attn_heads_reduction: How to combine attention heads ('concat' or 'average')
        dropout: Dropout rate
        activation: Activation function
        use_bias: Whether to use bias
        use_w: Whether to use weight transformation
    """

    def __init__(
        self,
        node_size: int,
        rel_size: int,
        triple_size: int,
        in_features: int,
        out_features: int,
        depth: int = 1,
        attn_heads: int = 1,
        attn_heads_reduction: str = 'concat',
        dropout: float = 0.3,
        activation: str = 'relu',
        use_bias: bool = False,
        use_w: bool = False,
    ):
        super(NR_GraphAttention, self).__init__()

        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possible reduction methods: concat, average')

        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.in_features = in_features
        self.out_features = out_features
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.dropout = dropout
        self.use_bias = use_bias
        self.use_w = use_w
        self.depth = depth

        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = lambda x: x

        # Attention kernels for each layer and each head
        self.attn_kernels = nn.ModuleList()
        for l in range(depth):
            layer_kernels = nn.ModuleList()
            for head in range(attn_heads):
                # Attention kernel for [self || neighbor || relation]
                attn_kernel = nn.Linear(3 * in_features, 1, bias=False)
                layer_kernels.append(attn_kernel)
            self.attn_kernels.append(layer_kernels)

        # Weight transformation if enabled
        if use_w:
            self.weight_transforms = nn.ModuleList()
            for l in range(depth):
                if attn_heads_reduction == 'concat':
                    w = nn.Linear(in_features * attn_heads, in_features, bias=use_bias)
                else:
                    w = nn.Linear(in_features, in_features, bias=use_bias)
                self.weight_transforms.append(w)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        rel_emb: torch.Tensor,
        adj_indices: torch.Tensor,
        sparse_indices: torch.Tensor,
        sparse_val: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Node features [num_nodes, in_features]
            rel_emb: Relation embeddings [rel_size, in_features]
            adj_indices: Adjacency list indices [num_edges, 2]
            sparse_indices: Sparse relation indices [num_triples, 2]
            sparse_val: Sparse relation values [num_triples]

        Returns:
            Output features [num_nodes, out_features]
        """
        outputs = []
        features = self.activation(features)
        outputs.append(features)

        for l in range(self.depth):
            features_list = []

            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels[l][head]

                # Compute relation sum using sparse operations
                # Create sparse tensor for relation aggregation
                rel_indices = sparse_indices.long()
                rel_values = sparse_val

                # Aggregate relations by triple ID
                rels_sum_by_triple = torch.zeros(self.triple_size, self.rel_size, device=features.device)
                rels_sum_by_triple.index_add_(0, rel_indices[:, 0],
                                   F.one_hot(rel_indices[:, 1], self.rel_size).float() * rel_values.unsqueeze(-1))

                # Transform to embedding space
                rels_sum_by_triple = torch.mm(rels_sum_by_triple, rel_emb)
                rels_sum_by_triple = F.normalize(rels_sum_by_triple, p=2, dim=1)

                # Map triple IDs to edges (now adj_indices and r_index have same length)
                # Each edge in adj_indices corresponds to an entry in r_index
                edge_triple_ids = rel_indices[:, 0]  # Triple ID for each edge
                rels_sum = rels_sum_by_triple[edge_triple_ids]

                # Gather neighbor and self features
                adj_src = adj_indices[:, 0].long()
                adj_dst = adj_indices[:, 1].long()

                selfs = features[adj_src]
                neighs = features[adj_dst]

                # Apply relational reflection
                # Project neighbor onto relation direction and reflect
                bias = (neighs * rels_sum).sum(dim=1, keepdim=True) * rels_sum
                neighs_reflected = neighs - 2 * bias

                # Compute attention scores
                att_input = torch.cat([selfs, neighs_reflected, rels_sum], dim=1)
                att_scores = attention_kernel(att_input).squeeze(-1)

                # Apply softmax per source node
                att_scores_exp = torch.exp(att_scores - att_scores.max())
                att_sum = torch.zeros(self.node_size, device=features.device)
                att_sum.index_add_(0, adj_src, att_scores_exp)

                # Normalize attention scores
                att_normalized = att_scores_exp / (att_sum[adj_src] + 1e-16)

                # Aggregate neighbor features
                new_features = torch.zeros(self.node_size, self.in_features, device=features.device)
                weighted_neighs = neighs_reflected * att_normalized.unsqueeze(-1)
                new_features.index_add_(0, adj_src, weighted_neighs)

                features_list.append(new_features)

            # Combine attention heads
            if self.attn_heads_reduction == 'concat':
                features = torch.cat(features_list, dim=-1)
            else:
                features = torch.stack(features_list, dim=0).mean(dim=0)

            # Apply weight transformation if enabled
            if self.use_w:
                features = self.weight_transforms[l](features)

            # Apply activation and dropout
            features = self.activation(features)
            features = self.dropout_layer(features)

            outputs.append(features)

        # Concatenate all layer outputs
        outputs = torch.cat(outputs, dim=-1)

        return outputs

    def get_output_dim(self) -> int:
        """Get output feature dimension."""
        if self.attn_heads_reduction == 'concat':
            per_layer_dim = self.in_features * self.attn_heads if not self.use_w else self.in_features
        else:
            per_layer_dim = self.in_features

        # First layer (input) + depth layers
        return self.in_features + self.depth * per_layer_dim


class RREAEncoder(nn.Module):
    """RREA encoder combining entity and relation embeddings with GAT.

    Args:
        node_size: Number of nodes
        rel_size: Number of relations
        triple_size: Number of triples
        embedding_dim: Dimension of embeddings
        depth: Number of GAT layers
        attn_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        node_size: int,
        rel_size: int,
        triple_size: int,
        embedding_dim: int = 100,
        depth: int = 2,
        attn_heads: int = 1,
        dropout: float = 0.3,
    ):
        super(RREAEncoder, self).__init__()

        self.node_size = node_size
        self.rel_size = rel_size
        self.embedding_dim = embedding_dim

        # Entity and relation embeddings
        self.entity_emb = nn.Embedding(node_size, embedding_dim)
        self.relation_emb = nn.Embedding(rel_size, embedding_dim)

        # Graph Attention layer
        self.gat = NR_GraphAttention(
            node_size=node_size,
            rel_size=rel_size,
            triple_size=triple_size,
            in_features=embedding_dim,
            out_features=embedding_dim,
            depth=depth,
            attn_heads=attn_heads,
            dropout=dropout,
        )

        # Initialize embeddings
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(
        self,
        adj_indices: torch.Tensor,
        sparse_indices: torch.Tensor,
        sparse_val: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            adj_indices: Adjacency list indices
            sparse_indices: Sparse relation indices
            sparse_val: Sparse relation values

        Returns:
            Entity embeddings
        """
        # Get embeddings
        entity_features = self.entity_emb.weight
        relation_features = self.relation_emb.weight

        # Apply GAT
        output = self.gat(
            features=entity_features,
            rel_emb=relation_features,
            adj_indices=adj_indices,
            sparse_indices=sparse_indices,
            sparse_val=sparse_val,
        )

        return output
