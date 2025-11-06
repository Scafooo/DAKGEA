"""Legacy access to predicate matching helpers."""

from src.augmentation.methods.predicate_matching import (
    get_predicate_label,
    match_relations,
    match_relations_with_attrnames,
    reduce_to_best_matches,
    string_similarity,
)

__all__ = [
    "get_predicate_label",
    "string_similarity",
    "match_relations_with_attrnames",
    "reduce_to_best_matches",
    "match_relations",
]
