"""Data structures for PLM augmentation BFS traversal."""

from dataclasses import dataclass
from typing import Optional

from rdflib import URIRef


@dataclass
class ExpansionNode:
    """Represents a node in the BFS expansion queue.

    Attributes:
        uri: The URI of the node to expand
        depth: Current depth in the BFS tree
        node_type: Type of node ("set" or "non-set")
        parent: Optional parent node URI for tracking expansion path
    """
    uri: URIRef
    depth: int
    node_type: str  # "set" or "non-set"
    parent: Optional[URIRef] = None

    @property
    def is_set_node(self) -> bool:
        """Check if this is a set node."""
        return self.node_type == "set"

    def __repr__(self) -> str:
        parent_str = f", parent={self.parent}" if self.parent else ""
        return f"ExpansionNode({self.node_type}, depth={self.depth}{parent_str})"
