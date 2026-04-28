"""SDEA alignment model adapter for DAKGEA.

SDEA (ICDE 2022) is a two-phase entity alignment model:
  Phase 1 – PairwiseTrainer: BERT-based attribute matching
  Phase 2 – RelationTrainer: GRU-based relational refinement

This adapter bridges DAKGEA's Dataset/KnowledgeGraph interface with
SDEA's KBStore / file-based pipeline.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.alignment_models.registry import MODEL_REGISTRY
from src.logger import get_logger

logger = get_logger(__name__)


def _build_kb_store(kg) -> "KBStore":
    """Convert a DAKGEA KnowledgeGraph (rdflib Graph) to a SDEA KBStore."""
    from .preprocess.KBStore import KBStore
    from .preprocess.Parser import OEAFileType

    kb = KBStore()
    for s, p, o in kg:
        s_str = str(s)
        p_str = str(p)
        o_str = str(o)
        # Decide if this is an attribute or relation triple based on object type
        import rdflib
        if isinstance(o, rdflib.term.Literal):
            kb.add_tuple(s_str, p_str, f'"{o_str}"', OEAFileType.attr)
        else:
            kb.add_tuple(s_str, p_str, o_str, OEAFileType.rel)
    return kb


def _tokenize_kb(kb, pretrain_bert_path: str) -> Dict[int, List[int]]:
    """Build eid → token-ids dict from KBStore attribute triples."""
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(pretrain_bert_path)
    eid2tids: Dict[int, List[int]] = {}

    for eid, ename in enumerate(kb.entities):
        parts = [ename.split("/")[-1].replace("_", " ")]
        for prop_id, lit_id in kb.literal_facts.get(eid, []):
            parts.append(kb.literals[lit_id])
        text = " ".join(parts)
        tokens = tokenizer.tokenize(text)
        tids = tokenizer.convert_tokens_to_ids(tokens)
        if tids:
            eid2tids[eid] = tids

    logger.info("[SDEA] Tokenized %d entities", len(eid2tids))
    return eid2tids


def _write_links(pairs: List[Tuple[str, str]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e1, e2 in pairs:
            f.write(f"{e1}\t{e2}\n")


def _split_links(
    train_aligned: List[Tuple[str, str]],
    test_pool: Optional[List[Tuple[str, str]]],
    valid_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Split alignment pairs into train / valid / test.

    train  = all of aligned_entities (already the r% training supervision).
    valid  = valid_frac of test_pool (carved from the test pool, not from training).
    test   = rest of test_pool.

    Fallback when test_pool is None: carve valid+test from training as last resort.
    """
    import random as _random
    rng = _random.Random(seed)
    train = list(train_aligned)

    if test_pool:
        pool = list(test_pool)
        rng.shuffle(pool)
        n_valid = max(1, int(len(pool) * valid_frac))
        valid = pool[:n_valid]
        test = pool[n_valid:] or pool[-1:]
    else:
        # Fallback: carve from training (undesirable but avoids crash)
        rng.shuffle(train)
        n_valid = max(1, int(len(train) * valid_frac))
        n_test = max(1, int(len(train) * 0.2))
        valid = train[:n_valid]
        test = train[n_valid: n_valid + n_test]
        train = train[n_valid + n_test:]

    return train, valid, test


@MODEL_REGISTRY.register("sdea")
class SDEAAlignment:
    """Two-phase SDEA: PairwiseTrainer (BERT) + RelationTrainer (GRU)."""

    def __init__(self, stage_config: Optional[Dict[str, Any]] = None) -> None:
        self.stage_config = stage_config or {}

        lineage = self.stage_config.get("lineage", {})
        artifact_root = Path(lineage.get("artifact_root", "results/artifact"))
        artifact_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = artifact_root / "sdea"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        sdea_cfg = self.stage_config.get("sdea", {})
        self.pretrain_bert_path = sdea_cfg.get("pretrain_bert_path", "bert-base-multilingual-cased")
        self.pairwise_epochs = int(sdea_cfg.get("pairwise_epochs", 30))
        self.relation_epochs = int(sdea_cfg.get("relation_epochs", 30))
        self.use_relation = bool(sdea_cfg.get("use_relation", True))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            "[SDEA] Initialized (bert=%s, device=%s, pw_epochs=%d, rel_epochs=%d)",
            self.pretrain_bert_path,
            self.device,
            self.pairwise_epochs,
            self.relation_epochs,
        )

    def evaluate(self, dataset_reduced, dataset_augmented):
        """Run two-phase SDEA training and return hits@1/10 + MRR."""
        dataset = dataset_augmented

        logger.info("[SDEA] Building KBStores from dataset...")
        fs1 = _build_kb_store(dataset.knowledge_graph_source)
        fs2 = _build_kb_store(dataset.knowledge_graph_target)
        logger.info("[SDEA] KBStore 1: %d entities, %d relations", len(fs1.entities), len(fs1.relation_ids))
        logger.info("[SDEA] KBStore 2: %d entities, %d relations", len(fs2.entities), len(fs2.relation_ids))

        logger.info("[SDEA] Tokenizing entities...")
        eid2tids1 = _tokenize_kb(fs1, self.pretrain_bert_path)
        eid2tids2 = _tokenize_kb(fs2, self.pretrain_bert_path)

        # Train pairs = aligned_entities (already reduced by the framework)
        train_aligned_id = [
            (str(e1), str(e2))
            for e1, e2 in dataset.aligned_entities
            if str(e1) in fs1.entity_ids and str(e2) in fs2.entity_ids
        ]

        # Test pairs = fixed_test_pairs when available (set by the reducer)
        fixed_test = getattr(dataset, "fixed_test_pairs", None)
        test_aligned_id = None
        if fixed_test is not None:
            test_aligned_id = [
                (str(e1), str(e2))
                for e1, e2 in fixed_test
                if str(e1) in fs1.entity_ids and str(e2) in fs2.entity_ids
            ]

        if not train_aligned_id:
            logger.warning("[SDEA] No aligned entities found – returning zero metrics")
            return {"hits@1": 0.0, "hits@10": 0.0, "mrr": 0.0}

        train_pairs, valid_pairs, test_pairs = _split_links(train_aligned_id, test_aligned_id)
        logger.info("[SDEA] Links – train: %d, valid: %d, test: %d",
                    len(train_pairs), len(valid_pairs), len(test_pairs))

        # Write link files and configure module globals
        tmpdir = self.checkpoint_dir / "links"
        tmpdir.mkdir(exist_ok=True)
        train_path = str(tmpdir / "train_links")
        valid_path = str(tmpdir / "valid_links")
        test_path = str(tmpdir / "test_links")
        _write_links(train_pairs, train_path)
        _write_links(valid_pairs, valid_path)
        _write_links(test_pairs, test_path)

        sdea_links = SimpleNamespace(
            train=train_path,
            valid=valid_path,
            test=test_path,
            model_save=str(self.checkpoint_dir / "basic_bert_model.pkl"),
            rel_model_save=str(self.checkpoint_dir / "rel_model.pkl"),
            kb_prop_emb_1=str(self.checkpoint_dir / "kb_prop_emb_1.pt"),
            kb_prop_emb_2=str(self.checkpoint_dir / "kb_prop_emb_2.pt"),
        )

        # Update SDEA globals before running trainers
        from . import _globals as g
        g.args = SimpleNamespace(
            pretrain_bert_path=self.pretrain_bert_path,
            relation=self.use_relation,
            blocking=False,
            functionality=False,
        )
        g.links = sdea_links

        logger.info("=" * 70)
        logger.info("[SDEA] Phase 1: PairwiseTrainer (BERT attribute matching)")
        logger.info("=" * 70)
        phase1_results = self._run_pairwise(eid2tids1, eid2tids2, fs1, fs2)

        logger.info("=" * 70)
        logger.info("[SDEA] Phase 2: RelationTrainer (GRU relational refinement)")
        logger.info("=" * 70)
        phase2_results = self._run_relation(eid2tids1, eid2tids2, fs1, fs2, phase1_results["model_path"])

        final = {
            "model": "sdea",
            "phases": {
                "pairwise": phase1_results,
                "relation": phase2_results,
            },
            "hits@1": phase2_results.get("hits@1", phase1_results.get("hits@1", 0.0)),
            "hits@5": phase2_results.get("hits@5", phase1_results.get("hits@5", 0.0)),
            "hits@10": phase2_results.get("hits@10", phase1_results.get("hits@10", 0.0)),
            "mrr": phase2_results.get("mrr", phase1_results.get("mrr", 0.0)),
        }

        logger.info("[SDEA] Final: Hits@1=%.4f  Hits@10=%.4f  MRR=%.4f",
                    final["hits@1"], final["hits@10"], final["mrr"])
        return final

    def _run_pairwise(self, eid2tids1, eid2tids2, fs1, fs2):
        from .train.PairwiseTrainer import PairwiseTrainer
        trainer = PairwiseTrainer()
        trainer.data_prepare(eid2tids1, eid2tids2, fs1, fs2)
        results = trainer.train(epochs=self.pairwise_epochs, device=self.device)
        logger.info("[SDEA Phase 1] hits@1=%.4f  hits@10=%.4f  mrr=%.4f",
                    results.get("hits@1", 0), results.get("hits@10", 0), results.get("mrr", 0))
        return results

    def _run_relation(self, eid2tids1, eid2tids2, fs1, fs2, bert_model_path):
        from .train.RelationTrainer import RelationTrainer
        trainer = RelationTrainer()
        trainer.data_prepare(eid2tids1, eid2tids2, fs1, fs2)
        results = trainer.train(
            basic_bert_path=bert_model_path,
            epochs=self.relation_epochs,
            device=self.device,
        )
        logger.info("[SDEA Phase 2] hits@1=%.4f  hits@10=%.4f  mrr=%.4f",
                    results.get("hits@1", 0), results.get("hits@10", 0), results.get("mrr", 0))
        return results


__all__ = ("SDEAAlignment",)
