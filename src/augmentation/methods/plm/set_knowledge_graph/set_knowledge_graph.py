"""RDF-compatible Set-based Knowledge Graph preserving literals on fused nodes."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

from rdflib import Literal, URIRef

from src.core.dataset import Dataset
from src.core.knowledge_graph import KnowledgeGraph


class SetKnowledgeGraph(KnowledgeGraph):
    """Fuse aligned entities into RDF-compatible URI-based sets with 'set:' prefix."""

    PREFIX = "set:"

    def __init__(self):
        super().__init__()
        self.set_nodes: dict[URIRef, list[URIRef]] = {}  # mapping merged_id → [src, tgt]
        self._relation_origin: dict[
            URIRef, dict[str, set[Tuple[URIRef, URIRef | Literal, str]]]
        ] = {}

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> "SetKnowledgeGraph":
        merged = cls()

        # --- Build mappings for aligned entities
        map_src: dict[URIRef, URIRef] = {}
        map_tgt: dict[URIRef, URIRef] = {}

        for src, tgt in dataset.aligned_entities:
            src_uri = URIRef(src)
            tgt_uri = URIRef(tgt)
            merged_id = URIRef(f"{cls.PREFIX}{src}|{tgt}")

            map_src[src_uri] = merged_id
            map_tgt[tgt_uri] = merged_id
            merged.set_nodes[merged_id] = [src_uri, tgt_uri]

        # --- Merge SOURCE triples
        for s, p, o in dataset.knowledge_graph_source.triples((None, None, None)):
            s_new = map_src.get(s, s)
            o_new = o if isinstance(o, Literal) else map_src.get(o, o)
            merged.add((s_new, p, o_new))
            merged._record_relation_origin(s_new, p, o_new, "out", "source")
            if isinstance(o_new, URIRef):
                merged._record_relation_origin(o_new, p, s_new, "in", "source")

        # --- Merge TARGET triples
        for s, p, o in dataset.knowledge_graph_target.triples((None, None, None)):
            s_new = map_tgt.get(s, s)
            o_new = o if isinstance(o, Literal) else map_tgt.get(o, o)
            merged.add((s_new, p, o_new))
            merged._record_relation_origin(s_new, p, o_new, "out", "target")
            if isinstance(o_new, URIRef):
                merged._record_relation_origin(o_new, p, s_new, "in", "target")

        # --- Attach literals from aligned entities to fused nodes
        for (src_uri, merged_id) in map_src.items():
            for _, p, o in dataset.knowledge_graph_source.triples((src_uri, None, None)):
                if isinstance(o, Literal):
                    merged.add((merged_id, p, o))
                    merged._record_relation_origin(merged_id, p, o, "out", "source")
        for (tgt_uri, merged_id) in map_tgt.items():
            for _, p, o in dataset.knowledge_graph_target.triples((tgt_uri, None, None)):
                if isinstance(o, Literal):
                    merged.add((merged_id, p, o))
                    merged._record_relation_origin(merged_id, p, o, "out", "target")

        return merged

    # ------------------------------------------------------------------
    # 🔹 Utility methods
    # ------------------------------------------------------------------
    @staticmethod
    def decode(set_uri: str) -> list[str]:
        """Convert 'set:http://dbpedia.org/...|http://www.wikidata.org/...' → list."""
        if set_uri.startswith("set:"):
            return set_uri.removeprefix("set:").split("|")
        return [set_uri]

    def is_set_node(self, node: URIRef) -> bool:
        """Return True if this node is a fused set node."""
        return isinstance(node, URIRef) and str(node).startswith(self.PREFIX)

    def get_components(self, node: URIRef) -> list[URIRef]:
        """Return component URIs (src/tgt) of a set node."""
        if not self.is_set_node(node):
            return [node]
        return self.set_nodes.get(node, [])

    def iter_set_nodes(self):
        """Iterate over all fused set nodes."""
        yield from self.set_nodes.keys()

    def summary(self) -> str:
        """Simple graph summary."""
        ents = {s for s, _, _ in self.triples((None, None, None)) if isinstance(s, URIRef)}
        lits = {o for _, _, o in self.triples((None, None, None)) if isinstance(o, Literal)}
        return (
            f"SetKnowledgeGraph with {len(self)} triples | "
            f"{len(ents)} entity nodes | {len(lits)} literal values | "
            f"{len(self.set_nodes)} fused set nodes"
        )

    # ------------------------------------------------------------------
    # Provenance helpers
    # ------------------------------------------------------------------
    def _relation_entry(self, node: URIRef) -> dict[str, set[Tuple[URIRef, URIRef | Literal, str]]]:
        return self._relation_origin.setdefault(node, {"source": set(), "target": set()})

    def _record_relation_origin(
        self,
        node: URIRef,
        predicate: URIRef,
        other: URIRef | Literal,
        direction: str,
        origin: str,
    ) -> None:
        if not self.is_set_node(node):
            return
        entry = self._relation_entry(node)
        entry[origin].add((predicate, other, direction))

    def get_relation_origins(
        self, node: URIRef
    ) -> dict[str, set[Tuple[URIRef, URIRef | Literal, str]]]:
        """Return relation provenance for a fused node, grouped by source/target."""
        if not self.is_set_node(node):
            return {"source": set(), "target": set()}
        entry = self._relation_origin.get(node)
        if not entry:
            return {"source": set(), "target": set()}
        return {
            "source": set(entry["source"]),
            "target": set(entry["target"]),
        }
