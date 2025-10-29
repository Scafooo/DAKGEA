"""Dataset wrapper around paired knowledge graphs and alignment metadata."""

from typing import Iterable, Tuple

from src.knowledge_graph.KnowledgeGraph import KnowledgeGraph

class Dataset:
    """Hold a source/target knowledge graph pair together with aligned entities."""

    def __init__(
        self,
        knowledge_graph_source: KnowledgeGraph,
        knowledge_graph_target: KnowledgeGraph,
        aligned_entities: Iterable[Tuple[str, str]],
    ):
        """
        Create a dataset for Entity Alignment experiments.

        Args:
            knowledge_graph_source: Knowledge graph representing the source domain.
            knowledge_graph_target: Knowledge graph representing the target domain.
            aligned_entities: Iterable of matched entity identifiers across graphs.
        """
        self.knowledge_graph_source = knowledge_graph_source
        self.knowledge_graph_target = knowledge_graph_target
        self.aligned_entities = aligned_entities

    def clone(self) -> "Dataset":
        """Return a deep copy of the dataset components."""

        source_copy = self.knowledge_graph_source.clone()
        target_copy = self.knowledge_graph_target.clone()
        aligned_copy = {(left, right) for left, right in self.aligned_entities}

        return Dataset(source_copy, target_copy, aligned_copy)
