"""Context serialisation, prompts and response parsing for Luffy-L.

The same ``ContextBuilder`` output is used by every Luffy-L variant and by the
``all_llm`` ablation, so the only thing that differs between them is what the
model is asked to produce and what the code does with it.

Identifiers inside a prompt are *local*: the visited set-entity is ``S`` and
its immediate set-neighbours are ``N1``, ``N2``, ... ; attribute and relation
predicates are referred to by readable labels that are unique per KG. The
``PromptContext`` keeps the mapping back to URIs, so the model never has to
reproduce a URI and every key it returns can be checked against the input.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rdflib import Literal, URIRef

SIDES = ("KG1", "KG2")
_AUG_RE = re.compile(r"_aug\d+$")

DOMAINS = {
    "BBC_DB": "music (artists, bands, albums)",
    "D_W_15K_V1": "general knowledge (DBpedia / Wikidata)",
    "D_W_15K_V2": "general knowledge (DBpedia / Wikidata)",
    "ICEW_WIKI": "political events (actors, organisations, countries)",
    "ICEW_YAGO": "political events (actors, organisations, countries)",
}


def domain_for(dataset_name: str) -> str:
    name = str(dataset_name or "").split("/")[-1]
    return DOMAINS.get(name, "general knowledge")


def _local_name(uri: str) -> str:
    text = str(uri).rstrip("/")
    for sep in ("#", "/", ":"):
        if sep in text:
            text = text.rsplit(sep, 1)[-1]
    return text or str(uri)


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class EntityView:
    """What the prompt shows of one node on one KG side."""

    entity: URIRef                                   # original URI in that KG
    attributes: Dict[str, List[str]] = field(default_factory=dict)   # label -> values
    relations: List[Tuple[str, str, str]] = field(default_factory=list)  # (label, dir, local id)
    n_truncated: int = 0                             # attribute values not shown (cap)


@dataclass
class PromptContext:
    node: URIRef                                     # Set-KG node of the visited entity
    is_pair: bool
    sides: Dict[str, EntityView]                     # side -> view of S
    neighbors: Dict[str, URIRef]                     # local id -> Set-KG node
    neighbor_is_pair: Dict[str, bool]
    neighbor_sides: Dict[str, Dict[str, EntityView]]
    attr_label_to_uri: Dict[str, Dict[str, URIRef]]  # side -> label -> predicate URI
    rel_label_to_uri: Dict[str, Dict[str, URIRef]]   # side -> label -> relation URI
    n_neighbors_total: int = 0
    n_literals_truncated: int = 0
    # Continuation calls (neighbour chunks after the first): the values already
    # generated for S, by KG and attribute label. None in the first call.
    s_generated: Optional[Dict[str, Dict[str, List[str]]]] = None


# ----------------------------------------------------------------------
# Context builder
# ----------------------------------------------------------------------
class ContextBuilder:
    """Builds the local context of a Set-KG node from the *original* KGs."""

    def __init__(self, dataset, set_graph, *, max_neighbors: int = 40,
                 max_neighbor_values: int = 120,
                 max_literals_target: int = 100, max_literals_neighbor: int = 30,
                 max_relations: int = 60, max_value_chars: int = 300,
                 max_value_chars_neighbor: int = 150):
        self.kgs = {"KG1": dataset.knowledge_graph_source, "KG2": dataset.knowledge_graph_target}
        self.set_graph = set_graph
        self.max_neighbors = max_neighbors              # neighbours per call
        self.max_neighbor_values = max_neighbor_values  # neighbour attribute values per call
        self.max_literals_target = max_literals_target
        self.max_literals_neighbor = max_literals_neighbor
        self.max_relations = max_relations
        self.max_value_chars = max_value_chars
        self.max_value_chars_neighbor = max_value_chars_neighbor

        # entity URI -> Set-KG node, per side (matched entities collapse into set nodes)
        self.to_node: Dict[str, Dict[URIRef, URIRef]] = {"KG1": {}, "KG2": {}}
        for node, comps in set_graph.set_nodes.items():
            if len(comps) >= 1:
                self.to_node["KG1"][comps[0]] = node
            if len(comps) >= 2:
                self.to_node["KG2"][comps[1]] = node

        self._lit_memo: Dict[Tuple[str, URIRef], List[Tuple[URIRef, str]]] = {}
        self._nb_memo: Dict[URIRef, List[URIRef]] = {}

        # Labels are computed once, before any augmented triple is written.
        self.attr_labels = {s: self._make_labels(self.kgs[s], attr=True) for s in SIDES}
        self.rel_labels = {s: self._make_labels(self.kgs[s], attr=False) for s in SIDES}

    # -- access to ORIGINAL triples only ---------------------------------
    # The KGs are augmented in place while the BFS runs, so every lookup
    # skips augmented entities (URIs minted as "<original>_aug<N>").
    @staticmethod
    def is_augmented(uri) -> bool:
        return bool(_AUG_RE.search(str(uri)))

    def _literals_of(self, side: str, ent: URIRef) -> List[Tuple[URIRef, str]]:
        # Original entities never gain literals during augmentation: safe to memoise.
        key = (side, ent)
        hit = self._lit_memo.get(key)
        if hit is None:
            hit = [(p, str(o)) for _, p, o in self.kgs[side].triples((ent, None, None))
                   if isinstance(o, Literal)]
            self._lit_memo[key] = hit
        return hit

    def _edges_of(self, side: str, ent: URIRef) -> List[Tuple[URIRef, str, URIRef]]:
        kg = self.kgs[side]
        out = [(p, "->", o) for _, p, o in kg.triples((ent, None, None))
               if isinstance(o, URIRef) and not self.is_augmented(o)]
        out += [(p, "<-", s) for s, p, _ in kg.triples((None, None, ent))
                if isinstance(s, URIRef) and not self.is_augmented(s)]
        return out

    def _in_kg(self, side: str, ent: URIRef) -> bool:
        kg = self.kgs[side]
        for _ in kg.triples((ent, None, None)):
            return True
        for _ in kg.triples((None, None, ent)):
            return True
        return False

    # -- labels ---------------------------------------------------------
    def _make_labels(self, kg, attr: bool) -> Dict[URIRef, str]:
        preds = set()
        for _, p, o in kg.triples((None, None, None)):
            if isinstance(o, Literal) == attr:
                preds.add(p)
        names = getattr(kg, "attr_to_name", {}) or {}
        base = {p: str(names.get(str(p)) or names.get(p) or _local_name(p)).strip() for p in preds}
        counts = Counter(base.values())
        labels, seen = {}, Counter()
        for p in sorted(preds, key=str):
            label = base[p] or _local_name(p)
            if counts[label] > 1:
                seen[label] += 1
                label = f"{label}#{seen[label]}"
            labels[p] = label
        return labels

    # -- helpers --------------------------------------------------------
    def components(self, node: URIRef) -> Dict[str, URIRef]:
        """Original entity URI of ``node`` on each side where it exists."""
        if self.set_graph.is_set_node(node):
            comps = self.set_graph.get_components(node)
            return {"KG1": comps[0], "KG2": comps[1]}
        return {side: node for side in SIDES if self._in_kg(side, node)}

    def _attributes(self, side: str, entity: URIRef, cap: int, max_chars: int
                    ) -> Tuple[Dict[str, List[str]], int]:
        items = sorted(self._literals_of(side, entity), key=lambda pv: (str(pv[0]), pv[1]))
        truncated = max(0, len(items) - cap)
        attrs: Dict[str, List[str]] = defaultdict(list)
        for p, v in items[:cap]:
            if len(v) > max_chars:
                v = v[:max_chars] + "..."
            attrs[self.attr_labels[side][p]].append(v)
        return dict(attrs), truncated

    def neighbor_nodes(self, node: URIRef) -> List[URIRef]:
        """All immediate Set-KG neighbours of ``node``: matched pairs first, then by
        number of connecting edges, then by URI (deterministic)."""
        hit = self._nb_memo.get(node)
        if hit is not None:
            return hit
        freq: Counter = Counter()
        for side, ent in self.components(node).items():
            for _, _, other in self._edges_of(side, ent):
                nb = self.to_node[side].get(other, other)
                if nb != node:
                    freq[nb] += 1
        hit = sorted(
            freq,
            key=lambda n: (0 if self.set_graph.is_set_node(n) else 1, -freq[n], str(n)),
        )
        self._nb_memo[node] = hit
        return hit

    def _n_values(self, nb: URIRef) -> int:
        return sum(min(len(self._literals_of(side, ent)), self.max_literals_neighbor)
                   for side, ent in self.components(nb).items())

    def chunk_neighbors(self, neighbors: List[URIRef]) -> List[List[URIRef]]:
        """Split neighbours into calls of at most ``max_neighbors`` entities and
        ``max_neighbor_values`` attribute values (a single large neighbour still gets a call)."""
        chunks: List[List[URIRef]] = []
        current: List[URIRef] = []
        values = 0
        for nb in neighbors:
            n = self._n_values(nb)
            if current and (len(current) >= self.max_neighbors or values + n > self.max_neighbor_values):
                chunks.append(current)
                current, values = [], 0
            current.append(nb)
            values += n
        if current:
            chunks.append(current)
        return chunks

    # -- main entry point -----------------------------------------------
    def build(self, node: URIRef, neighbors: Optional[List[URIRef]] = None,
              s_generated: Optional[Dict[str, Dict[URIRef, List[str]]]] = None) -> PromptContext:
        """Context of ``node`` with the given neighbours (default: the first chunk).

        ``s_generated`` (predicate URIs) marks a continuation call: S is shown with the
        values already generated for it and only the neighbours are to be generated.
        """
        comps = self.components(node)
        all_nbs = self.neighbor_nodes(node)
        n_total = len(all_nbs)
        if neighbors is None:
            chunks = self.chunk_neighbors(all_nbs)
            neighbors = chunks[0] if chunks else []
        nbs = list(neighbors)
        local_ids = {nb: f"N{i + 1}" for i, nb in enumerate(nbs)}
        truncated = 0

        sides: Dict[str, EntityView] = {}
        for side, ent in comps.items():
            attrs, t = self._attributes(side, ent, self.max_literals_target, self.max_value_chars)
            truncated += t
            rels, seen = [], set()
            for p, direction, other in self._edges_of(side, ent):
                nb = self.to_node[side].get(other, other)
                if nb not in local_ids:
                    continue
                item = (self.rel_labels[side][p], direction, local_ids[nb])
                if item not in seen:
                    seen.add(item)
                    rels.append(item)
            rels.sort()
            sides[side] = EntityView(ent, attrs, rels[: self.max_relations], t)

        nb_sides: Dict[str, Dict[str, EntityView]] = {}
        for nb, lid in local_ids.items():
            views = {}
            for side, ent in self.components(nb).items():
                attrs, t = self._attributes(side, ent, self.max_literals_neighbor,
                                            self.max_value_chars_neighbor)
                views[side] = EntityView(ent, attrs, [], t)
            nb_sides[lid] = views

        return PromptContext(
            node=node,
            is_pair=self.set_graph.is_set_node(node),
            sides=sides,
            neighbors={lid: nb for nb, lid in local_ids.items()},
            neighbor_is_pair={lid: self.set_graph.is_set_node(nb) for nb, lid in local_ids.items()},
            neighbor_sides=nb_sides,
            attr_label_to_uri={s: {l: p for p, l in self.attr_labels[s].items()} for s in SIDES},
            rel_label_to_uri={s: {l: p for p, l in self.rel_labels[s].items()} for s in SIDES},
            n_neighbors_total=n_total,
            n_literals_truncated=truncated,
            s_generated=None if s_generated is None else {
                side: {self.attr_labels[side][p]: vals for p, vals in preds.items()}
                for side, preds in s_generated.items()
            },
        )


# ----------------------------------------------------------------------
# Serialisation (identical for Luffy-L and all_llm)
# ----------------------------------------------------------------------
def serialize_context(ctx: PromptContext, domain: str) -> str:
    def view(v: EntityView, with_relations: bool) -> dict:
        out = {"attributes": v.attributes}
        if with_relations:
            out["relations"] = [list(r) for r in v.relations]
        return out

    s_block = {
        "kind": "matched pair (KG1 and KG2 describe the same real entity)" if ctx.is_pair
        else "single entity",
        **{side: view(v, True) for side, v in ctx.sides.items()},
    }
    if ctx.s_generated is not None:
        s_block["already_generated_replacement"] = ctx.s_generated
    payload = {
        "domain": domain,
        "S": s_block,
        "neighbors": {
            lid: {
                "kind": "matched pair" if ctx.neighbor_is_pair[lid] else "single entity",
                **{side: view(v, False) for side, v in ctx.neighbor_sides[lid].items()},
            }
            for lid in ctx.neighbors
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(", ", ": "))


SYSTEM_LUFFY_L = """You generate synthetic training data for entity resolution across two knowledge graphs, KG1 and KG2.

You receive the local context of an entity S: its attribute values in each KG where it exists, its relations to neighbouring entities N1, N2, ..., and the attribute values of those neighbours.

Invent a NEW, FICTIONAL entity that replaces S, and fictional replacements for each neighbour, following these rules:
1. For every attribute listed in the input, produce a new value for the same entity, KG and attribute name. Produce one new value for each original value.
2. Never copy an original value and never use the name of a real-world entity. The new values must be plausible for the attribute and the domain.
3. When an entity is a matched pair, its KG1 and KG2 values must describe the SAME fictional entity, each written in the surface style and format of its own KG (e.g. "Crowe, Daniel" in one KG and "Daniel Crowe" in the other).
4. Keep the format of each original value: dates stay dates in the same format, numbers stay numbers, short names stay short, long texts stay of similar length.
5. The fictional neighbours must be consistent with the fictional S and with the relations shown (e.g. the band of a fictional musician is a fictional band).
6. Use exactly the identifiers (S, N1, N2, ...), KG names and attribute names given in the input. Do not add attributes, entities or KGs.

Answer with JSON only, no explanation, in this format:
{"S": {"KG1": {"<attribute>": ["<new value>", ...]}, "KG2": {...}}, "N1": {"KG1": {...}, "KG2": {...}}, ...}
Include only the KGs in which each entity appears in the input."""


SYSTEM_LUFFY_L_CONTINUATION = """You generate synthetic training data for entity resolution across two knowledge graphs, KG1 and KG2.

You receive the local context of an entity S: its original attribute values, its relations to some neighbouring entities N1, N2, ..., and the attribute values of those neighbours. S has ALREADY been replaced by a fictional entity, given in S.already_generated_replacement.

Invent fictional replacements for the neighbours only, following these rules:
1. For every attribute of every neighbour listed in the input, produce a new value for the same neighbour, KG and attribute name. Produce one new value for each original value.
2. Never copy an original value and never use the name of a real-world entity. The new values must be plausible for the attribute and the domain.
3. When a neighbour is a matched pair, its KG1 and KG2 values must describe the SAME fictional entity, each written in the surface style and format of its own KG.
4. Keep the format of each original value: dates stay dates in the same format, numbers stay numbers, short names stay short, long texts stay of similar length.
5. The fictional neighbours must be consistent with the already generated replacement of S and with the relations shown.
6. Use exactly the identifiers (N1, N2, ...), KG names and attribute names given in the input. Do not add attributes, entities or KGs. Do not output S.

Answer with JSON only, no explanation, in this format:
{"N1": {"KG1": {"<attribute>": ["<new value>", ...]}, "KG2": {...}}, "N2": {...}, ...}
Include only the KGs in which each neighbour appears in the input."""


SYSTEM_ALL_LLM = """You generate synthetic training data for entity resolution across two knowledge graphs, KG1 and KG2.

You receive the local context of a matched pair S (the KG1 and KG2 descriptions of the same real entity): its attribute values, its relations to neighbouring entities N1, N2, ..., and the attribute values of those neighbours.

Create ONE new, FICTIONAL matched pair that can be added to the training data, following these rules:
1. Give the new entity attribute values in KG1 and in KG2, using only attribute names that appear in the input for that KG.
2. Never copy an original value and never use the name of a real-world entity. The new values must be plausible for the attribute and the domain.
3. The KG1 and KG2 values must describe the SAME fictional entity, each written in the surface style and format of its own KG.
4. Give the new entity relations in KG1 and in KG2. Each relation must use a relation name that appears in the input for that KG and must point to one of the given neighbours N1, N2, ... that appears in that KG. Do not invent new entities or new relation names.

Answer with JSON only, no explanation, in this format:
{"KG1": {"attributes": {"<attribute>": ["<value>", ...]}, "relations": [["<relation>", "->" or "<-", "<neighbour id>"], ...]}, "KG2": {...}}"""


def build_user_prompt(ctx: PromptContext, domain: str) -> str:
    return "Input:\n" + serialize_context(ctx, domain)


# ----------------------------------------------------------------------
# Parsing and validation
# ----------------------------------------------------------------------
def extract_json(text: str) -> Optional[dict]:
    """Return the outermost JSON object in ``text`` or None."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    blob = cleaned[start:end + 1]
    for candidate in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _as_values(raw) -> List[str]:
    if isinstance(raw, list):
        vals = raw
    else:
        vals = [raw]
    out = []
    for v in vals:
        if isinstance(v, (str, int, float)) and str(v).strip():
            out.append(str(v).strip())
    return out


@dataclass
class ParsedLiterals:
    """Generated literals per local id and side: id -> side -> predicate URI -> values."""

    values: Dict[str, Dict[str, Dict[URIRef, List[str]]]]
    n_unknown_keys: int = 0


def parse_luffy_l(text: str, ctx: PromptContext) -> Optional[ParsedLiterals]:
    obj = extract_json(text)
    if obj is None:
        return None
    expected = {"S": ctx.sides}
    expected.update(ctx.neighbor_sides)
    unknown = 0
    out: Dict[str, Dict[str, Dict[URIRef, List[str]]]] = {}
    for lid, side_obj in obj.items():
        if lid not in expected or not isinstance(side_obj, dict):
            unknown += 1
            continue
        for side, attrs in side_obj.items():
            if side not in expected[lid] or not isinstance(attrs, dict):
                unknown += 1
                continue
            allowed = expected[lid][side].attributes
            for label, raw in attrs.items():
                if label not in allowed:
                    unknown += 1
                    continue
                vals = _as_values(raw)
                if vals:
                    pred = ctx.attr_label_to_uri[side][label]
                    out.setdefault(lid, {}).setdefault(side, {})[pred] = vals
    return ParsedLiterals(out, unknown)


@dataclass
class ParsedPair:
    attributes: Dict[str, Dict[URIRef, List[str]]]          # side -> pred -> values
    relations: Dict[str, List[Tuple[URIRef, str, str]]]      # side -> (rel URI, dir, local id)
    n_out_of_alphabet: int = 0


def parse_all_llm(text: str, ctx: PromptContext) -> Optional[ParsedPair]:
    obj = extract_json(text)
    if obj is None:
        return None
    attrs: Dict[str, Dict[URIRef, List[str]]] = {}
    rels: Dict[str, List[Tuple[URIRef, str, str]]] = {}
    dropped = 0
    for side, side_obj in obj.items():
        if side not in ctx.sides or not isinstance(side_obj, dict):
            dropped += 1
            continue
        allowed_attrs = set(ctx.sides[side].attributes)
        for label, raw in (side_obj.get("attributes") or {}).items():
            if label not in allowed_attrs:
                dropped += 1
                continue
            vals = _as_values(raw)
            if vals:
                attrs.setdefault(side, {})[ctx.attr_label_to_uri[side][label]] = vals
        allowed_rel_labels = {r[0] for r in ctx.sides[side].relations}
        for item in side_obj.get("relations") or []:
            if not (isinstance(item, (list, tuple)) and len(item) == 3):
                dropped += 1
                continue
            label, direction, lid = (str(x) for x in item)
            ok = (
                label in allowed_rel_labels
                and direction in ("->", "<-")
                and lid in ctx.neighbor_sides
                and side in ctx.neighbor_sides[lid]
            )
            if not ok:
                dropped += 1
                continue
            rels.setdefault(side, []).append((ctx.rel_label_to_uri[side][label], direction, lid))
    return ParsedPair(attrs, rels, dropped)
