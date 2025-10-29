"""Evaluation helpers for entity alignment results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


EntityPair = Tuple[str, str]


@dataclass
class AlignmentMetrics:
    """Container for common entity-alignment metrics."""

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


def compute_prf(predictions: Iterable[EntityPair], truth: Iterable[EntityPair]) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 from predicted alignments."""

    pred_set = set(predictions)
    truth_set = set(truth)
    if not pred_set:
        return 0.0, 0.0, 0.0

    true_pos = len(pred_set & truth_set)
    if true_pos == 0:
        return 0.0, 0.0, 0.0
    precision = true_pos / len(pred_set)
    recall = true_pos / len(truth_set) if truth_set else 0.0
    if precision + recall == 0.0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_hits_and_mrr(
    ranking: Dict[str, Sequence[Tuple[str, float]]],
    truth: Dict[str, str],
    *,
    k: int = 10,
) -> Tuple[float, float]:
    """Compute Hits@1/Hits@K and MRR given ranked candidates per source entity."""

    if not ranking:
        return 0.0, 0.0

    hits_at_1 = 0.0
    hits_at_k = 0.0
    mrr_total = 0.0
    evaluated = 0

    for source, candidates in ranking.items():
        if source not in truth:
            continue
        evaluated += 1
        target_truth = truth[source]
        ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
        for idx, (candidate, _) in enumerate(ordered):
            if candidate == target_truth:
                if idx == 0:
                    hits_at_1 += 1.0
                if idx < k:
                    hits_at_k += 1.0
                mrr_total += 1.0 / (idx + 1)
                break

    if evaluated == 0:
        return 0.0, 0.0
    hits_at_1 /= evaluated
    hits_at_k /= evaluated
    mrr = mrr_total / evaluated
    return hits_at_1, hits_at_k, mrr


def build_ranking(predictions: Iterable[Tuple[str, str, float]]) -> Dict[str, List[Tuple[str, float]]]:
    """Group scored predictions by source entity for ranking metrics."""

    grouped: Dict[str, List[Tuple[str, float]]] = {}
    for source, target, score in predictions:
        grouped.setdefault(source, []).append((target, score))
    return grouped


def evaluate_alignment(
    scored_predictions: Iterable[Tuple[str, str, float]],
    truth_pairs: Iterable[EntityPair],
) -> AlignmentMetrics:
    """Compute a suite of entity-alignment metrics from scored predictions."""

    truth_list = list(truth_pairs)
    truth_map = {src: tgt for src, tgt in truth_list}
    ranking = build_ranking(scored_predictions)
    hits1, hits10, mrr = compute_hits_and_mrr(ranking, truth_map, k=10)

    top_predictions = [(src, candidates[0][0]) for src, candidates in ranking.items() if candidates]
    precision, recall, f1 = compute_prf(top_predictions, truth_list)

    return AlignmentMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        hits_at_1=hits1,
        hits_at_10=hits10,
        mrr=mrr,
    )
