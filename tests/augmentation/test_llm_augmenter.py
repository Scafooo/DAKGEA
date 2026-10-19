"""Offline tests for Luffy-L (no Bedrock): run with `pytest tests/augmentation/test_llm_augmenter.py -q`.

Toy graph (the paper's running example):

    KG1: bbc:bowie --member_of--> bbc:tm      bbc:bowie --label--> bbc:emi (singleton)
    KG2: db:bowie  --member_of--> db:tm
    training matches: (bbc:bowie, db:bowie), (bbc:tm, db:tm)
"""

import json
import re

import pytest
from rdflib import Literal, URIRef

from src.augmentation.methods.llm.augmenter import LLMAugmenter
from src.augmentation.methods.llm.llm_client import CacheMiss
from src.core.dataset import Dataset
from src.core.knowledge_graph import KnowledgeGraph

B, D = "http://bbc/", "http://db/"
AUG = re.compile(r"_aug\d+$")


def toy_dataset() -> Dataset:
    k1, k2 = KnowledgeGraph(), KnowledgeGraph()
    k1.add((URIRef(B + "bowie"), URIRef(B + "has_name"), Literal("Bowie, David")))
    k1.add((URIRef(B + "bowie"), URIRef(B + "member_of"), URIRef(B + "tm")))
    k1.add((URIRef(B + "bowie"), URIRef(B + "label"), URIRef(B + "emi")))
    k1.add((URIRef(B + "tm"), URIRef(B + "has_name"), Literal("Tin Machine")))
    k1.add((URIRef(B + "emi"), URIRef(B + "has_name"), Literal("EMI")))
    k2.add((URIRef(D + "bowie"), URIRef(D + "name"), Literal("David Bowie")))
    k2.add((URIRef(D + "bowie"), URIRef(D + "active_from"), Literal("1962")))
    k2.add((URIRef(D + "bowie"), URIRef(D + "member_of"), URIRef(D + "tm")))
    k2.add((URIRef(D + "tm"), URIRef(D + "name"), Literal("Tin Machine (band)")))
    return Dataset(k1, k2, [(B + "bowie", D + "bowie"), (B + "tm", D + "tm")])


class MockLLM:
    """Deterministic stand-in for the LLM. Values encode the call number: FAKE-c<call>-..."""

    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.calls = 0

    def __call__(self, system: str, user: str) -> str:
        self.calls += 1
        if self.mode == "garbage":
            return "Sorry, I cannot help with that."
        data = json.loads(user.split("Input:\n", 1)[1])
        c = self.calls

        def fake(lid, side, attrs):
            return {a: [f"FAKE-c{c}-{lid}-{side}-{a}-{i}" for i in range(len(v))] for a, v in attrs.items()}

        if "Create ONE new" in system:                         # all_llm prompt
            out = {}
            for side in ("KG1", "KG2"):
                if side in data["S"]:
                    rels = [list(r) for r in data["S"][side]["relations"]]
                    rels += [["invented_relation", "->", "N1"], [rels[0][0] if rels else "x", "->", "N99"]]
                    out[side] = {"attributes": fake("S", side, data["S"][side]["attributes"]),
                                 "relations": rels}
            return json.dumps(out)
        out = {}
        if "ALREADY been replaced" not in system:              # continuation calls do not regenerate S
            out["S"] = {s: fake("S", s, v["attributes"]) for s, v in data["S"].items() if s in ("KG1", "KG2")}
        for lid, nb in data["neighbors"].items():
            out[lid] = {s: fake(lid, s, v["attributes"]) for s, v in nb.items() if s in ("KG1", "KG2")}
        return json.dumps(out)


def run(tmp_path, variant="full", copy=False, ratio=1.0, provider="mock", mock=None, seed=11037,
        **llm_extra):
    cfg = {
        "seed": seed,
        "experiment": {"dataset": "openea/BBC_DB"},
        "augmentation": {
            "method": "llm", "ratio": ratio, "variant": variant, "copy_original_literals": copy,
            "stage_root": str(tmp_path / f"stage_{variant}_{ratio}_{provider}"),
            "llm": {"provider": provider, "cache_path": str(tmp_path / "cache.sqlite"), **llm_extra},
        },
    }
    aug = LLMAugmenter(cfg)
    aug.mock_fn = mock or MockLLM()
    out = aug.augment(toy_dataset())
    return out, aug


def aug_pairs(ds):
    return [(a, b) for a, b in ds.aligned_entities if AUG.search(a)]


def new_triples(ds):
    orig = toy_dataset()
    return (set(ds.knowledge_graph_source) - set(orig.knowledge_graph_source),
            set(ds.knowledge_graph_target) - set(orig.knowledge_graph_target))


def literals(kg, ent):
    return {(str(p), str(o)) for _, p, o in kg.triples((URIRef(ent), None, None)) if isinstance(o, Literal)}


# ----------------------------------------------------------------------
def test_full_structure_cache_and_same_call(tmp_path):
    out, aug = run(tmp_path)
    pairs = aug_pairs(out)
    assert len(pairs) == 2                                         # budget = ceil(2 * 1.0)
    by_orig = {AUG.sub("", a): (a, b) for a, b in pairs}
    bowie, tm = by_orig[B + "bowie"], by_orig[B + "tm"]
    # P1: relational edges point to the augmented neighbour pair (O_aug), in both KGs
    assert (URIRef(bowie[0]), URIRef(B + "member_of"), URIRef(tm[0])) in out.knowledge_graph_source
    assert (URIRef(bowie[1]), URIRef(D + "member_of"), URIRef(tm[1])) in out.knowledge_graph_target
    # Algorithm 3: each node generated once; at least one neighbour served by LLMCache
    s = aug.stats
    assert s["llm_requests"] + s["llm_cache_hits"] == 3           # bowie, tm, emi
    assert s["llm_cache_hits"] >= 1
    # both sides of every augmented pair come from the same LLM call
    for a, b in pairs:
        calls = {v.split("-")[1] for _, v in literals(out.knowledge_graph_source, a) | literals(out.knowledge_graph_target, b)}
        assert len(calls) == 1, calls
    # no original value is copied when copy_original_literals = false
    for kg_new in new_triples(out):
        for _, _, o in kg_new:
            if isinstance(o, Literal):
                assert str(o).startswith("FAKE-"), o
    stats = json.loads((tmp_path / "stage_full_1.0_mock" / "luffy_l_stats.json").read_text())
    assert stats["variant"] == "full" and stats["api_calls"] == s["llm_requests"]


def test_copy_original_literals_reproduces_luffy_s_scaffold(tmp_path):
    out, _ = run(tmp_path, copy=True)
    bowie = [a for a, _ in aug_pairs(out) if a.startswith(B + "bowie")][0]
    lits = literals(out.knowledge_graph_source, bowie)
    assert (B + "has_name", "Bowie, David") in lits               # original KG1 value copied
    assert (D + "name", "David Bowie") in lits                    # ... and KG2 value too (as in Luffy-S)
    assert any(v.startswith("FAKE-") for _, v in lits)


def test_no_cache_makes_one_call_per_node(tmp_path):
    _, aug = run(tmp_path, variant="no_cache")
    assert aug.stats["llm_cache_hits"] == 0
    assert aug.stats["llm_requests"] == 3


def test_no_bfs_has_no_relational_edges(tmp_path):
    out, aug = run(tmp_path, variant="no_bfs")
    assert aug.max_depth == 0
    assert len(aug_pairs(out)) == 2
    for kg_new in new_triples(out):
        assert not [t for t in kg_new if isinstance(t[2], URIRef)]


def test_parse_failure_falls_back_to_original_values(tmp_path):
    out, aug = run(tmp_path, mock=MockLLM("garbage"))
    assert aug.stats["parse_failures"] >= 1
    bowie = [a for a, _ in aug_pairs(out) if a.startswith(B + "bowie")][0]
    assert (B + "has_name", "Bowie, David") in literals(out.knowledge_graph_source, bowie)


def test_smaller_af_replays_prefix_without_api_calls(tmp_path):
    full, _ = run(tmp_path, ratio=1.0)
    half, aug = run(tmp_path, ratio=0.5, provider="replay")        # raises CacheMiss on any new prompt
    assert len(aug_pairs(half)) == 1
    assert aug_pairs(half)[0] in aug_pairs(full)
    # deterministic replay of the full run gives exactly the same graphs
    again, _ = run(tmp_path, ratio=1.0, provider="replay")
    assert set(again.knowledge_graph_source) == set(full.knowledge_graph_source)
    assert set(again.knowledge_graph_target) == set(full.knowledge_graph_target)


def test_replay_with_other_seed_misses(tmp_path):
    run(tmp_path, seed=11037)
    with pytest.raises(CacheMiss):
        run(tmp_path, seed=11038, provider="replay")


def test_all_llm_stays_in_the_input_alphabet(tmp_path):
    out, aug = run(tmp_path, variant="all_llm")
    pairs = aug_pairs(out)
    assert len(pairs) == 2 and aug.stats["llm_requests"] == 2
    assert aug.stats["out_of_alphabet_dropped"] >= 2               # invented relation + unknown N99
    orig = toy_dataset()
    for kg_new, kg_orig, base in ((new_triples(out)[0], orig.knowledge_graph_source, B),
                                  (new_triples(out)[1], orig.knowledge_graph_target, D)):
        orig_rels = {p for _, p, o in kg_orig if isinstance(o, URIRef)}
        orig_ents = {s for s, _, _ in kg_orig} | {o for _, _, o in kg_orig if isinstance(o, URIRef)}
        for s, p, o in kg_new:
            if isinstance(o, URIRef):
                assert p in orig_rels                                # no new relation names
                endpoint = o if AUG.search(str(s)) else s
                assert endpoint in orig_ents                         # edges only to original entities
        # every new triple touches a synthetic anchor: no new neighbour entities are created
        for s, p, o in kg_new:
            assert AUG.search(str(s)) or AUG.search(str(o))
        new_aug = {x for t in kg_new for x in (t[0], t[2]) if isinstance(x, URIRef) and AUG.search(str(x))}
        assert new_aug <= {URIRef(a) for pair in pairs for a in pair}


def test_required_flags(tmp_path):
    cfg = {"seed": 1, "augmentation": {"method": "llm", "ratio": 1.0, "variant": "full",
                                       "llm": {"provider": "mock", "cache_path": str(tmp_path / "c.sqlite")}}}
    with pytest.raises(ValueError, match="copy_original_literals"):
        LLMAugmenter(cfg)
    cfg["augmentation"]["copy_original_literals"] = False
    cfg["augmentation"]["variant"] = "bogus"
    with pytest.raises(ValueError, match="variant"):
        LLMAugmenter(cfg)


def test_fake_provider_never_feeds_real_runs(tmp_path):
    fake_out, _ = run(tmp_path, provider="fake")
    again, _ = run(tmp_path, provider="replay", replay_fake=True)     # dry-run replay works
    assert set(again.knowledge_graph_source) == set(fake_out.knowledge_graph_source)
    with pytest.raises(CacheMiss):                                     # a real replay does not see it
        run(tmp_path, provider="replay")


def star_dataset(n=100) -> Dataset:
    """One matched hub with n singleton neighbours (like an ICEWS country), one value each."""
    k1, k2 = KnowledgeGraph(), KnowledgeGraph()
    k1.add((URIRef(B + "hub"), URIRef(B + "name"), Literal("Hub")))
    k2.add((URIRef(D + "hub"), URIRef(D + "name"), Literal("Hub (KG2)")))
    for i in range(n):
        k1.add((URIRef(B + f"x{i:03d}"), URIRef(B + "related_to"), URIRef(B + "hub")))
        k1.add((URIRef(B + f"x{i:03d}"), URIRef(B + "name"), Literal(f"Actor {i}")))
    return Dataset(k1, k2, [(B + "hub", D + "hub")])


def test_hub_neighbours_are_chunked_and_all_primed(tmp_path):
    cfg = {"seed": 11037, "experiment": {"dataset": "openea/ICEW_WIKI"},
           "augmentation": {"method": "llm", "ratio": 1.0, "variant": "full", "copy_original_literals": False,
                            "stage_root": str(tmp_path / "stage"),
                            "llm": {"provider": "mock", "cache_path": str(tmp_path / "c.sqlite")}}}
    aug = LLMAugmenter(cfg)
    systems = []
    mock = MockLLM()
    aug.mock_fn = lambda system, user: (systems.append(system), mock(system, user))[1]
    out = aug.augment(star_dataset(100))
    s = aug.stats
    assert s["llm_requests"] == 3                        # ceil(100 / 40) calls, not 1 + 100
    assert s["llm_cache_hits"] == 100                    # every singleton served by LLMCache
    assert sum("ALREADY been replaced" in x for x in systems) == 2
    new = set(out.knowledge_graph_source) - set(star_dataset(100).knowledge_graph_source)
    assert all(str(o).startswith("FAKE-") for _, _, o in new if isinstance(o, Literal))
    assert len({sub for sub, _, o in new if isinstance(o, Literal)}) == 101   # hub_aug + 100 x_aug


def test_traversal_is_independent_of_triple_insertion_order(tmp_path):
    """Same seed -> same prompts even if the graphs were filled in another order
    (DAKGEA loaders fill graphs from Python sets, so order changes with PYTHONHASHSEED)."""
    def shuffled(ds, seed):
        import random as _r
        out = []
        for kg in (ds.knowledge_graph_source, ds.knowledge_graph_target):
            triples = list(kg)
            _r.Random(seed).shuffle(triples)
            g = KnowledgeGraph()
            for t in triples:
                g.add(t)
            out.append(g)
        return Dataset(out[0], out[1], list(ds.aligned_entities)[::-1])

    prompts = []
    for order_seed in (1, 2):
        cfg = {"seed": 11037, "experiment": {"dataset": "openea/BBC_DB"},
               "augmentation": {"method": "llm", "ratio": 1.0, "variant": "full", "copy_original_literals": False,
                                "llm": {"provider": "mock", "cache_path": str(tmp_path / f"c{order_seed}.sqlite")}}}
        aug = LLMAugmenter(cfg)
        seq = []
        mock = MockLLM()
        aug.mock_fn = lambda system, user, seq=seq, mock=mock: (seq.append(user), mock(system, user))[1]
        aug.augment(shuffled(toy_dataset(), order_seed))
        prompts.append(seq)
    assert prompts[0] == prompts[1]


def test_cached_pair_primes_its_own_pending_neighbours(tmp_path):
    """bowie's call primes tm; when tm is served from LLMCache, tm's own singleton
    neighbour (EMI-2) is generated by one continuation call that sees tm's values."""
    ds = toy_dataset()
    ds.knowledge_graph_source.add((URIRef(B + "tm"), URIRef(B + "label"), URIRef(B + "emi2")))
    ds.knowledge_graph_source.add((URIRef(B + "emi2"), URIRef(B + "has_name"), Literal("EMI-2")))
    cfg = {"seed": 11037, "experiment": {"dataset": "openea/BBC_DB"},
           "augmentation": {"method": "llm", "ratio": 1.0, "variant": "full", "copy_original_literals": False,
                            "llm": {"provider": "mock", "cache_path": str(tmp_path / "c.sqlite")}}}
    aug = LLMAugmenter(cfg)
    systems, mock = [], MockLLM()
    aug.mock_fn = lambda system, user: (systems.append(user), mock(system, user))[1]
    aug.augment(ds)
    s = aug.stats
    assert s["llm_requests"] == 2 and s["continuation_requests"] == 1
    assert s["llm_cache_hits"] == 3                      # tm, emi, emi2
    cont = [u for u in systems if "already_generated_replacement" in u]
    assert len(cont) == 1 and "EMI-2" in cont[0]
