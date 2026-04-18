"""Training loop for AttrE.

Faithfully replicates the alternating-epoch training strategy of the
original KBA.py:

  * **Even epochs** — train on relation triples (URI mode).
  * **Odd epochs**  — run similarity-alignment optimisation on attribute triples.

Evaluation (hits@1, hits@10, MRR) is run every *eval_freq* epochs.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.alignment_models.methods.attrE.data_pipeline import (
    AttrEDataBundle,
    AttrTriple,
    RelTriple,
)
from src.alignment_models.methods.attrE.metrics import compute_metrics
from src.alignment_models.methods.attrE.model_core import AttrEModel
from src.logger import get_logger

logger = get_logger(__name__)

Pair = Tuple[int, int]
DeviceSpec = Union[int, str, torch.device, None]


def _resolve_device(spec: DeviceSpec) -> torch.device:
    if spec is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(spec, torch.device):
        return spec
    if isinstance(spec, int):
        return torch.device(f"cuda:{spec}" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


class AttrETrainer:
    """Train and evaluate an :class:`AttrEModel` on an :class:`AttrEDataBundle`."""

    def __init__(
        self,
        model: AttrEModel,
        data: AttrEDataBundle,
        config: Dict[str, Any],
        device_spec: DeviceSpec = None,
    ) -> None:
        self.model = model
        self.data = data
        self.cfg = config
        self.device = _resolve_device(device_spec)
        self.model.to(self.device)

        lr = float(config.get("learning_rate", 0.01))
        self.opt_rel = Adam(
            list(model.rel_ent_emb.parameters()) +
            list(model.rel_pred_emb.parameters()),
            lr=lr,
        )
        self.opt_atr = Adam(
            list(model.atr_ent_emb.parameters()) +
            list(model.atr_pred_emb.parameters()) +
            list(model.char_emb.parameters()),
            lr=lr,
        )
        # opt_sim must NOT update atr_ent_emb — doing so would pull attr-view
        # embeddings toward the (heterogeneous-schema) rel-view, destroying the
        # attribute alignment built by even-epoch training.  The TF original
        # (KBA.py opt_vars_sim) only updates relationship_ent_embedding.
        self.opt_sim = Adam(
            list(model.rel_ent_emb.parameters()),
            lr=lr,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> List[Dict[str, float]]:
        """Train for the configured number of epochs.

        Returns:
            List of per-evaluation-epoch metric dicts.
        """
        epochs = int(self.cfg.get("epochs", 50))
        eval_freq = int(self.cfg.get("eval_freq", 10))
        batch_size = int(self.cfg.get("batch_size", 100))

        history: List[Dict[str, float]] = []

        for epoch in range(epochs):
            if epoch % 2 == 0:
                # Even epoch: relation triples (ranking + sim) + attribute triples (ranking only)
                # Matches KBA.py even-epoch branch: URI data then literal data.
                loss_r = self._train_relation_epoch(batch_size)
                loss_a = self._train_attribute_epoch(batch_size)
                logger.debug(
                    "[AttrE] epoch=%d rel_loss=%.4f attr_loss=%.4f", epoch, loss_r, loss_a
                )
            else:
                # Odd epoch: sim alignment only (on attribute-triple head entities)
                # Matches KBA.py odd-epoch branch: sim_optimizer on literal data.
                loss_s = self._train_sim_epoch(batch_size)
                logger.debug("[AttrE] epoch=%d sim_loss=%.4f", epoch, loss_s)

            if (epoch + 1) % eval_freq == 0:
                metrics = self.evaluate()
                metrics["epoch"] = epoch + 1
                history.append(metrics)
                logger.info(
                    "[AttrE] epoch=%d  hits@1=%.4f  hits@10=%.4f  mrr=%.4f",
                    epoch + 1, metrics["hits@1"], metrics["hits@10"], metrics["mrr"],
                )

        return history

    def evaluate(self) -> Dict[str, float]:
        """Compute hits@1, hits@10, MRR on the test pairs."""
        self.model.eval()
        emb_kg1 = self.model.get_relation_embeddings(self.data.kg1_entity_ids)
        emb_kg2 = self.model.get_relation_embeddings(self.data.kg2_entity_ids)
        metrics = compute_metrics(
            emb_kg1, emb_kg2,
            self.data.test_pairs,
            self.data.kg1_entity_ids,
            self.data.kg2_entity_ids,
        )
        self.model.train()
        return metrics

    # ------------------------------------------------------------------
    # Per-epoch training helpers
    # ------------------------------------------------------------------

    def _train_relation_epoch(self, batch_size: int) -> float:
        """One pass over relation triples (URI mode)."""
        all_triples = self.data.rel_triples_1 + self.data.rel_triples_2
        random.shuffle(all_triples)

        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(all_triples), batch_size):
            batch = all_triples[start : start + batch_size]
            if not batch:
                continue

            pos_h, pos_r, pos_t = zip(*batch)
            neg_h, neg_r, neg_t = self._corrupt_rel_batch(pos_h, pos_r, pos_t)

            ph = torch.tensor(pos_h, dtype=torch.long, device=self.device)
            pr = torch.tensor(pos_r, dtype=torch.long, device=self.device)
            pt = torch.tensor(pos_t, dtype=torch.long, device=self.device)
            nh = torch.tensor(neg_h, dtype=torch.long, device=self.device)
            nr = torch.tensor(neg_r, dtype=torch.long, device=self.device)
            nt = torch.tensor(neg_t, dtype=torch.long, device=self.device)

            self.opt_rel.zero_grad()
            loss = self.model.forward_rel(ph, pr, pt, nh, nr, nt)
            loss.backward()
            self.opt_rel.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _train_attribute_epoch(self, batch_size: int) -> float:
        """One pass over attribute triples (literal mode)."""
        all_triples = self.data.attr_triples_1 + self.data.attr_triples_2
        random.shuffle(all_triples)

        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(all_triples), batch_size):
            batch = all_triples[start : start + batch_size]
            if not batch:
                continue

            pos_h, pos_r, pos_char, pos_w = zip(*batch)
            neg_h, neg_r, neg_char, neg_w = self._corrupt_attr_batch(
                pos_h, pos_r, pos_char, pos_w
            )

            ph = torch.tensor(pos_h, dtype=torch.long, device=self.device)
            pr = torch.tensor(pos_r, dtype=torch.long, device=self.device)
            pc = torch.tensor(pos_char, dtype=torch.long, device=self.device)
            pw = torch.tensor(pos_w, dtype=torch.float, device=self.device)
            nh = torch.tensor(neg_h, dtype=torch.long, device=self.device)
            nr = torch.tensor(neg_r, dtype=torch.long, device=self.device)
            nc = torch.tensor(neg_char, dtype=torch.long, device=self.device)
            nw = torch.tensor(neg_w, dtype=torch.float, device=self.device)

            self.opt_atr.zero_grad()
            loss = self.model.forward_attr(ph, pr, pc, pw, nh, nr, nc, nw)
            loss.backward()
            self.opt_atr.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _train_sim_epoch(self, batch_size: int) -> float:
        """Similarity alignment pass: pull rel-view and attr-view together.

        Matches KBA.py odd-epoch branch: sim_optimizer is run on attribute
        triple batches, using the *head* entity of each triple as the target.
        """
        all_triples = self.data.attr_triples_1 + self.data.attr_triples_2
        if not all_triples:
            # No attribute triples — fall back to all entities
            all_ids = list(range(self.data.num_entities))
            random.shuffle(all_ids)
            total_loss = 0.0
            num_batches = 0
            for start in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[start : start + batch_size]
                ids_t = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
                self.opt_sim.zero_grad()
                loss = self.model.forward_sim(ids_t)
                loss.backward()
                self.opt_sim.step()
                total_loss += loss.item()
                num_batches += 1
            return total_loss / max(num_batches, 1)

        random.shuffle(all_triples)

        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(all_triples), batch_size):
            batch = all_triples[start : start + batch_size]
            if not batch:
                continue

            # Use head entity IDs (subject of each attribute triple)
            head_ids = [t[0] for t in batch]
            ids_t = torch.tensor(head_ids, dtype=torch.long, device=self.device)

            self.opt_sim.zero_grad()
            loss = self.model.forward_sim(ids_t)
            loss.backward()
            self.opt_sim.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Negative sampling
    # ------------------------------------------------------------------

    def _corrupt_rel_batch(
        self,
        pos_h: Tuple[int, ...],
        pos_r: Tuple[int, ...],
        pos_t: Tuple[int, ...],
    ) -> Tuple[List[int], List[int], List[int]]:
        """Corrupt either head or tail of each positive triple."""
        neg_h, neg_r, neg_t = list(pos_h), list(pos_r), list(pos_t)
        neg_pool = self.data.neg_pool_1 + self.data.neg_pool_2
        for i in range(len(pos_h)):
            if random.random() < 0.5:
                # Corrupt tail
                neg_t[i] = random.choice(neg_pool)
            else:
                # Corrupt head
                neg_h[i] = random.choice(neg_pool)
        return neg_h, neg_r, neg_t

    def _corrupt_attr_batch(
        self,
        pos_h: Tuple[int, ...],
        pos_r: Tuple[int, ...],
        pos_char: Tuple[List[int], ...],
        pos_w: Tuple[float, ...],
    ) -> Tuple[List[int], List[int], List[List[int]], List[float]]:
        """Corrupt the head entity of each positive attribute triple."""
        neg_pool = self.data.neg_pool_1 + self.data.neg_pool_2
        neg_h = [random.choice(neg_pool) for _ in pos_h]
        return neg_h, list(pos_r), list(pos_char), list(pos_w)
