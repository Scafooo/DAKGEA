"""PyTorch port of the AttrE model.

Architecture mirrors the original TF 1.x implementation:
  - Two entity embedding spaces: *relation* view and *attribute* view
  - TransE-style scoring for both relation triples and attribute triples
  - Character n-gram encoding for literal values
  - Cosine similarity alignment loss between the two embedding spaces

References:
  Trisedya et al. (2019) "Entity Alignment between Knowledge Graphs Using
  Attribute Embeddings". AAAI 2019.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.alignment_models.methods.attrE.data_pipeline import AttrEDataBundle


class AttrEModel(nn.Module):
    """Dual-view entity alignment model with character-level literal encoding.

    Two embedding spaces are maintained:
    * **Relation view** – trained on (subject, relation, object) triples.
    * **Attribute view** – trained on (subject, attribute, literal) triples
      where the literal is encoded via character n-gram pooling.

    A cosine similarity loss aligns the two views for every entity.
    """

    def __init__(
        self,
        num_entities: int,
        num_predicates: int,
        num_chars: int,
        hidden_dim: int = 100,
        char_seq_len: int = 10,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.char_seq_len = char_seq_len

        # Relation-view embeddings
        self.rel_ent_emb = nn.Embedding(num_entities, hidden_dim)
        self.rel_pred_emb = nn.Embedding(num_predicates, hidden_dim)

        # Attribute-view embeddings
        self.atr_ent_emb = nn.Embedding(num_entities, hidden_dim)
        self.atr_pred_emb = nn.Embedding(num_predicates, hidden_dim)

        # Character embeddings (shared across both views)
        self.char_emb = nn.Embedding(num_chars, hidden_dim, padding_idx=0)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for emb in (
            self.rel_ent_emb, self.rel_pred_emb,
            self.atr_ent_emb, self.atr_pred_emb,
            self.char_emb,
        ):
            nn.init.xavier_uniform_(emb.weight)
        # Reset padding index for characters to zero
        with torch.no_grad():
            self.char_emb.weight[0].zero_()

    # ------------------------------------------------------------------
    # N-gram encoding (faithful port of the TF while_loop)
    # ------------------------------------------------------------------

    def _ngram_encode(self, char_seq: torch.Tensor) -> torch.Tensor:
        """Encode a character sequence via cumulative n-gram mean pooling.

        Reproduces the ``calculate_ngram_weight`` function from the original
        TF 1.x implementation:

        For a sequence of length *L*, the encoding is computed as::

            result = Σ_{k=1}^{L}  mean(char_emb[L-k : L])

        which gives characters closer to the start of the sequence more
        influence (they appear in more n-gram windows).

        Args:
            char_seq: ``[B, L]`` integer tensor of character IDs (0 = padding).

        Returns:
            ``[B, hidden_dim]`` literal embedding.
        """
        B, L = char_seq.shape
        # Character lookup + mask out padding
        emb = self.char_emb(char_seq)          # [B, L, H]
        mask = (char_seq != 0).float().unsqueeze(2)   # [B, L, 1]
        emb = emb * mask                       # zero-out padding positions

        # Reverse so that index-slicing implements prefix windows of the
        # *original* sequence (matching the tf.reverse in the TF code)
        emb_rev = emb.flip(1)                  # [B, L, H]

        result = torch.zeros(B, self.hidden_dim, device=char_seq.device, dtype=emb.dtype)
        for k in range(1, L + 1):
            # Take last k characters of reversed sequence = first k of original
            window = emb_rev[:, L - k:, :]    # [B, k, H]
            result = result + window.mean(1)   # [B, H]

        return result                          # [B, H]

    # ------------------------------------------------------------------
    # Scoring functions
    # ------------------------------------------------------------------

    def score_rel(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """TransE L1 score for relation triples: ||h + r - t||_1."""
        return (h + r - t).abs().sum(dim=-1)   # [B]

    def score_attr(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        char_seq: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted TransE score for attribute triples.

        Args:
            h: ``[B, H]`` entity embedding (attribute view).
            r: ``[B, H]`` predicate embedding (attribute view).
            char_seq: ``[B, L]`` character IDs of the literal value.
            weight: ``[B]`` predicate importance weights.

        Returns:
            ``[B]`` weighted L1 distances.
        """
        lit_emb = self._ngram_encode(char_seq)                    # [B, H]
        dist = (h + r - lit_emb).abs().sum(dim=-1)                # [B]
        return dist * weight                                       # [B]

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def rel_loss(
        self,
        pos_h: torch.Tensor,
        pos_r: torch.Tensor,
        pos_t: torch.Tensor,
        neg_h: torch.Tensor,
        neg_r: torch.Tensor,
        neg_t: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Margin ranking loss for relation triples + cosine alignment."""
        pos_score = self.score_rel(pos_h, pos_r, pos_t)
        neg_score = self.score_rel(neg_h, neg_r, neg_t)
        ranking_loss = F.relu(pos_score - neg_score + margin).mean()

        # Cosine similarity loss between relation and attribute views
        h_rel = F.normalize(self.rel_ent_emb(
            # We re-use indices from pos_h embedding lookup to get IDs.
            # Instead we compute the view alignment on the full batch.
            self._batch_ids_from_emb(pos_h, self.rel_ent_emb),
        ), dim=-1)
        h_atr = F.normalize(self.atr_ent_emb(
            self._batch_ids_from_emb(pos_h, self.rel_ent_emb),
        ), dim=-1)
        sim_loss = (1.0 - (h_rel * h_atr).sum(dim=-1)).mean()
        return ranking_loss + sim_loss

    def attr_loss(
        self,
        pos_h: torch.Tensor,
        pos_r: torch.Tensor,
        pos_char: torch.Tensor,
        pos_w: torch.Tensor,
        neg_h: torch.Tensor,
        neg_r: torch.Tensor,
        neg_char: torch.Tensor,
        neg_w: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Margin ranking loss for attribute triples."""
        pos_score = self.score_attr(pos_h, pos_r, pos_char, pos_w)
        neg_score = self.score_attr(neg_h, neg_r, neg_char, neg_w)
        return F.relu(pos_score - neg_score + margin).mean()

    def sim_alignment_loss(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Cosine loss to align relation-view and attribute-view embeddings."""
        h_rel = F.normalize(self.rel_ent_emb(entity_ids), dim=-1)
        h_atr = F.normalize(self.atr_ent_emb(entity_ids), dim=-1)
        return (1.0 - (h_rel * h_atr).sum(dim=-1)).mean()

    @staticmethod
    def _batch_ids_from_emb(
        embedded: torch.Tensor,
        emb_module: nn.Embedding,
    ) -> torch.Tensor:
        """Recover integer IDs from an already-looked-up embedding batch.

        This is used internally where we have the embedding tensor but need
        the IDs for a second lookup in a different table.  We locate each
        row in the weight matrix via nearest-neighbour.

        Note: for training this is only an approximation; the exact IDs are
        passed explicitly in :meth:`forward_rel` / :meth:`forward_attr`.
        """
        # Dot-product similarity: [B, num_entities]
        sim = embedded @ emb_module.weight.T
        return sim.argmax(dim=-1)

    # ------------------------------------------------------------------
    # Convenience forward methods
    # ------------------------------------------------------------------

    def forward_rel(
        self,
        pos_h_ids: torch.Tensor,
        pos_r_ids: torch.Tensor,
        pos_t_ids: torch.Tensor,
        neg_h_ids: torch.Tensor,
        neg_r_ids: torch.Tensor,
        neg_t_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Relation-view forward pass, returns scalar loss.

        Per the original AttrE paper, even epochs train *only* on relation
        triples (no similarity-alignment term).  The sim loss is applied
        exclusively in the dedicated sim epoch (odd epochs).
        """
        pos_h = self.rel_ent_emb(pos_h_ids)
        pos_r = self.rel_pred_emb(pos_r_ids)
        pos_t = self.rel_ent_emb(pos_t_ids)
        neg_h = self.rel_ent_emb(neg_h_ids)
        neg_r = self.rel_pred_emb(neg_r_ids)
        neg_t = self.rel_ent_emb(neg_t_ids)

        return F.relu(
            self.score_rel(pos_h, pos_r, pos_t) -
            self.score_rel(neg_h, neg_r, neg_t) + 1.0
        ).mean()

    def forward_attr(
        self,
        pos_h_ids: torch.Tensor,
        pos_r_ids: torch.Tensor,
        pos_char: torch.Tensor,
        pos_w: torch.Tensor,
        neg_h_ids: torch.Tensor,
        neg_r_ids: torch.Tensor,
        neg_char: torch.Tensor,
        neg_w: torch.Tensor,
    ) -> torch.Tensor:
        """Attribute-view forward pass, returns scalar loss."""
        pos_h = self.atr_ent_emb(pos_h_ids)
        pos_r = self.atr_pred_emb(pos_r_ids)
        neg_h = self.atr_ent_emb(neg_h_ids)
        neg_r = self.atr_pred_emb(neg_r_ids)
        return self.attr_loss(pos_h, pos_r, pos_char, pos_w, neg_h, neg_r, neg_char, neg_w)

    def forward_sim(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Similarity alignment loss for a batch of entity IDs."""
        return self.sim_alignment_loss(entity_ids)

    # ------------------------------------------------------------------
    # Embedding export (for evaluation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_relation_embeddings(self, entity_ids: Optional[List[int]] = None) -> torch.Tensor:
        """Return L2-normalised relation-view embeddings.

        Args:
            entity_ids: Subset of IDs to export; if None, returns all.

        Returns:
            ``[N, hidden_dim]`` float32 tensor on CPU.
        """
        if entity_ids is None:
            emb = self.rel_ent_emb.weight
        else:
            ids = torch.tensor(entity_ids, dtype=torch.long, device=self.rel_ent_emb.weight.device)
            emb = self.rel_ent_emb(ids)
        return F.normalize(emb, dim=-1).cpu()
