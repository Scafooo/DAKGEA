"""Dataset wrapper around paired knowledge graphs and alignment metadata."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from src.core.knowledge_graph import KnowledgeGraph


class Dataset:
    """Hold a source/target knowledge graph pair together with aligned entities."""

    def __init__(
        self,
        knowledge_graph_source: KnowledgeGraph,
        knowledge_graph_target: KnowledgeGraph,
        aligned_entities: Iterable[Tuple[str, str]],
        attribute_matches: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Create a dataset for Entity Alignment experiments.

        Args:
            knowledge_graph_source: Knowledge graph representing the source domain.
            knowledge_graph_target: Knowledge graph representing the target domain.
            aligned_entities: Iterable of matched entity identifiers across graphs.
            attribute_matches: Optional dict mapping source_attr_uri -> [target_attr_uris]
                              Loaded from match_attr file if available.
        """
        self.knowledge_graph_source = knowledge_graph_source
        self.knowledge_graph_target = knowledge_graph_target
        self.aligned_entities = tuple(aligned_entities)
        self.attribute_matches = attribute_matches or {}

    def clone(self) -> "Dataset":
        """Return a deep copy of the dataset components."""

        source_copy = self.knowledge_graph_source.clone()
        target_copy = self.knowledge_graph_target.clone()
        aligned_copy = tuple(self.aligned_entities)
        # Deep copy of attribute matches
        attr_matches_copy = {k: list(v) for k, v in self.attribute_matches.items()}

        return Dataset(source_copy, target_copy, aligned_copy, attr_matches_copy)
