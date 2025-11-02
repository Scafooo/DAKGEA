"""Evaluation metrics mirroring the original BERT-INT reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class AlignmentMetrics:
    precision: float
    recall: float
    f1: float
    hits_at_1: float
    hits_at_10: float
    mrr: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "hits@1": self.hits_at_1,
            "hits@10": self.hits_at_10,
            "mrr": self.mrr,
        }


def evaluate_alignment(
    predictions: Sequence[Tuple[str, str, float]],
    truth_pairs: Iterable[Tuple[str, str]],
) -> AlignmentMetrics:
    truth_set = set(truth_pairs)
    pred_pairs = [(src, tgt) for src, tgt, _ in predictions]
    pred_set = set(pred_pairs)

    tp = len(pred_set & truth_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(truth_set) if truth_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    ranking: Dict[str, List[Tuple[str, float]]] = {}
    for src, tgt, score in predictions:
        ranking.setdefault(src, []).append((tgt, score))
    for src in ranking:
        ranking[src].sort(key=lambda item: item[1], reverse=True)

    hits1 = 0
    hits10 = 0
    reciprocal_ranks = 0.0
    total = len(ranking)
    for src, candidates in ranking.items():
        for position, (tgt, _) in enumerate(candidates, start=1):
            if (src, tgt) in truth_set:
                reciprocal_ranks += 1.0 / position
                if position == 1:
                    hits1 += 1
                if position <= 10:
                    hits10 += 1
                break
    hits_at_1 = hits1 / total if total else 0.0
    hits_at_10 = hits10 / total if total else 0.0
    mrr = reciprocal_ranks / total if total else 0.0

    return AlignmentMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        hits_at_1=hits_at_1,
        hits_at_10=hits_at_10,
        mrr=mrr,
    )
