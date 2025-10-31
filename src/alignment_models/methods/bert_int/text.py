"""Text extraction helpers mirroring the original BERT-INT heuristics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set

from rdflib import Graph, Literal, URIRef

from src.alignment_models.methods.bert_int.tokenization import normalise_uri


def friendly_name(uri: str, dataset_name: str) -> str:
    """Replicate the dataset-specific URI cleaning used by the reference code."""

    dataset_key = (dataset_name or "").lower()

    def _extract(segment: str, prefix: str) -> str:
        return segment.split(prefix)[-1]

    if "en_ja" in dataset_key:
        if "http://dbpedia.org/resource/" in uri:
            text = _extract(uri, "http://dbpedia.org/resource/")
        else:
            text = _extract(uri, "http://ja.dbpedia.org/resource/")
    elif "en_de" in dataset_key:
        if "http://dbpedia.org/resource/" in uri:
            text = _extract(uri, "http://dbpedia.org/resource/")
        else:
            text = _extract(uri, "http://de.dbpedia.org/resource/")
    elif "en_fr" in dataset_key:
        if "http://dbpedia.org/resource/" in uri:
            text = _extract(uri, "http://dbpedia.org/resource/")
        else:
            text = _extract(uri, "http://fr.dbpedia.org/resource/")
    elif "dbp_en_yg_en" in dataset_key or "d_y" in dataset_key:
        if "http://dbpedia.org/resource/" in uri:
            text = _extract(uri, "http://dbpedia.org/resource/")
        else:
            text = uri
    elif "dbp_en_wd_en" in dataset_key or "d_w" in dataset_key:
        if "http://dbpedia.org/resource/" in uri:
            text = _extract(uri, "http://dbpedia.org/resource/")
        else:
            text = _extract(uri, "http://www.wikidata.org/entity/")
    elif "bbc" in dataset_key:
        if "dbp:" in uri:
            text = _extract(uri, "dbp:")
        else:
            text = uri.split("/")[-1]
    elif "imdb" in dataset_key or "tmdb" in dataset_key or "tvdb" in dataset_key:
        text = uri.split("/")[-1]
    elif "rest" in dataset_key:
        text = uri.split("/")[-1]
    else:
        text = uri.split("/")[-1]

    text = text.replace("_", " ").strip()
    return text or normalise_uri(uri)


def _priority_for_dataset(dataset_name: str, kg_index: int) -> Sequence[str]:
    """Return ordered attribute priorities for the given dataset/graph."""

    key = (dataset_name or "").lower()
    attrs: Sequence[str] = ()

    def one_of(*needles: str) -> bool:
        return any(n in key for n in needles)

    if one_of("en_ja"):
        attrs = (
            (
                "http://xmlns.com/foaf/0.1/name",
                "http://dbpedia.org/ontology/birthName",
                "http://xmlns.com/foaf/0.1/nick",
                "http://dbpedia.org/ontology/synonym",
                "http://dbpedia.org/ontology/alias",
                "http://dbpedia.org/ontology/title",
                "http://dbpedia.org/ontology/longName",
                "http://dbpedia.org/ontology/givenName",
            )
            if kg_index == 1
            else (
                "http://xmlns.com/foaf/0.1/name",
                "http://dbpedia.org/ontology/title",
                "http://dbpedia.org/ontology/commonName",
                "http://dbpedia.org/ontology/givenName",
                "http://dbpedia.org/ontology/alias",
                "http://dbpedia.org/ontology/background",
                "http://dbpedia.org/ontology/purpose",
            )
        )
    elif one_of("en_de"):
        attrs = (
            (
                "http://xmlns.com/foaf/0.1/name",
                "http://dbpedia.org/ontology/title",
                "http://dbpedia.org/ontology/birthName",
                "http://xmlns.com/foaf/0.1/nick",
                "http://dbpedia.org/ontology/office",
                "http://dbpedia.org/ontology/leaderTitle",
            )
            if kg_index == 1
            else (
                "http://xmlns.com/foaf/0.1/name",
                "http://dbpedia.org/ontology/originalTitle",
                "http://xmlns.com/foaf/0.1/nick",
                "http://dbpedia.org/ontology/motto",
                "http://dbpedia.org/ontology/leaderTitle",
            )
        )
    elif one_of("en_fr"):
        attrs = (
            (
                "http://xmlns.com/foaf/0.1/name",
                "http://dbpedia.org/ontology/title",
                "http://dbpedia.org/ontology/birthName",
                "http://xmlns.com/foaf/0.1/nick",
                "http://dbpedia.org/ontology/office",
                "http://dbpedia.org/ontology/motto",
                "http://dbpedia.org/ontology/combatant",
            )
            if kg_index == 1
            else (
                "http://xmlns.com/foaf/0.1/name",
                "http://dbpedia.org/ontology/birthName",
                "http://xmlns.com/foaf/0.1/nick",
                "http://dbpedia.org/ontology/peopleName",
                "http://dbpedia.org/ontology/thumbnailCaption",
                "http://dbpedia.org/ontology/flag",
                "http://dbpedia.org/ontology/motto",
                "http://dbpedia.org/ontology/title",
            )
        )
    elif one_of("dbp_en_yg_en", "d_y"):
        attrs = (
            (
                "http://xmlns.com/foaf/0.1/name",
                "http://dbpedia.org/ontology/birthName",
                "http://xmlns.com/foaf/0.1/nick",
                "http://dbpedia.org/ontology/alias",
                "http://dbpedia.org/ontology/office",
                "http://dbpedia.org/ontology/leaderTitle",
                "http://dbpedia.org/ontology/motto",
            )
            if kg_index == 1
            else (
                "skos:prefLabel",
                "rdfs:label",
                "redirectedFrom",
                "hasFamilyName",
                "hasGivenName",
                "hasMotto",
            )
        )
    elif one_of("dbp_en_wd_en", "d_w"):
        attrs = (
            (
                "http://xmlns.com/foaf/0.1/name",
                "http://dbpedia.org/ontology/birthName",
                "http://purl.org/dc/elements/1.1/description",
                "http://xmlns.com/foaf/0.1/nick",
                "http://xmlns.com/foaf/0.1/givenName",
                "http://dbpedia.org/ontology/leaderTitle",
                "http://dbpedia.org/ontology/alias",
                "http://dbpedia.org/ontology/motto",
                "http://dbpedia.org/ontology/office",
            )
            if kg_index == 1
            else (
                "http://www.wikidata.org/entity/P373",
                "http://schema.org/description",
                "http://www.wikidata.org/entity/P1476",
                "http://www.wikidata.org/entity/P935",
                "http://www.w3.org/2004/02/skos/core#altLabel",
            )
        )
    elif one_of("srprs_d_w_15k_v1"):
        attrs = (
            (
                "http://dbpedia.org/ontology/title",
                "http://dbpedia.org/ontology/birthName",
                "http://dbpedia.org/ontology/longName",
            )
            if kg_index == 1
            else (
                "http://www.wikidata.org/entity/P373",
                "http://www.wikidata.org/entity/P1476",
            )
        )
    elif one_of("srprs_d_w_15k_v2"):
        attrs = (
            (
                "http://dbpedia.org/ontology/title",
                "http://dbpedia.org/ontology/birthName",
                "http://dbpedia.org/ontology/longName",
            )
            if kg_index == 1
            else (
                "http://www.wikidata.org/entity/P373",
                "http://www.wikidata.org/entity/P1476",
            )
        )
    elif one_of("d_w_15k_v1"):
        attrs = (
            (
                "http://xmlns.com/foaf/0.1/name",
                "http://dbpedia.org/ontology/birthName",
                "http://dbpedia.org/ontology/name",
                "http://dbpedia.org/ontology/longName",
                "http://dbpedia.org/ontology/otherName",
                "http://dbpedia.org/ontology/teamName",
            )
            if kg_index == 1
            else (
                "http://www.wikidata.org/entity/P373",
                "http://www.wikidata.org/entity/P1476",
                "http://www.w3.org/2004/02/skos/core#altLabel",
                "http://schema.org/description",
            )
        )
    elif one_of("d_w_15k_v2"):
        attrs = (
            (
                "http://xmlns.com/foaf/0.1/name",
                "http://xmlns.com/foaf/0.1/givenName",
                "http://dbpedia.org/ontology/birthName",
                "http://dbpedia.org/ontology/longName",
            )
            if kg_index == 1
            else (
                "http://www.wikidata.org/entity/P373",
                "http://www.wikidata.org/entity/P1476",
                "http://www.w3.org/2004/02/skos/core#altLabel",
                "http://schema.org/description",
            )
        )
    elif one_of("fr_en"):
        attrs = (
            (
                "http://fr.dbpedia.org/property/titre",
                "http://xmlns.com/foaf/0.1/name",
                "http://fr.dbpedia.org/property/name",
                "http://fr.dbpedia.org/property/label",
            )
            if kg_index == 1
            else (
                "http://dbpedia.org/property/title",
                "http://xmlns.com/foaf/0.1/name",
                "http://dbpedia.org/property/name",
                "http://xmlns.com/foaf/0.1/givenName",
                "http://dbpedia.org/ontology/birthName",
                "http://dbpedia.org/property/label",
            )
        )
    elif one_of("ja_en"):
        attrs = (
            (
                "http://ja.dbpedia.org/property/title",
                "http://xmlns.com/foaf/0.1/name",
                "http://ja.dbpedia.org/property/name",
                "http://xmlns.com/foaf/0.1/givenName",
                "http://ja.dbpedia.org/property/label",
            )
            if kg_index == 1
            else (
                "http://dbpedia.org/property/title",
                "http://xmlns.com/foaf/0.1/name",
                "http://dbpedia.org/property/name",
                "http://xmlns.com/foaf/0.1/givenName",
                "http://dbpedia.org/property/label",
            )
        )
    elif one_of("zh_en"):
        attrs = (
            (
                "http://zh.dbpedia.org/property/title",
                "http://xmlns.com/foaf/0.1/name",
                "http://ja.dbpedia.org/property/name",
                "http://xmlns.com/foaf/0.1/givenName",
                "http://ja.dbpedia.org/property/label",
            )
            if kg_index == 1
            else (
                "http://dbpedia.org/property/title",
                "http://xmlns.com/foaf/0.1/name",
                "http://dbpedia.org/property/name",
                "http://xmlns.com/foaf/0.1/givenName",
                "http://dbpedia.org/property/label",
            )
        )
    elif one_of("bbc"):
        attrs = (
            (
                "http://purl.org/dc/elements/1.1/title",
                "http://xmlns.com/foaf/0.1/name",
                "http://open.vocab.org/terms/sortlabel",
            )
            if kg_index == 1
            else ("prop:title",)
        )
    elif one_of("imdb-tmdb", "imdb-tvdb", "tmdb-tvdb"):
        attrs = (
            (
                "https://www.scads.de/movieBenchmark/ontology/name",
                "https://www.scads.de/movieBenchmark/ontology/title",
                "https://www.scads.de/movieBenchmark/ontology/originalTitle",
            )
        )
    elif one_of("rest"):
        attrs = (
            ("http://www.okkam.org/ontology_restaurant1.owl#name",)
            if kg_index == 1
            else ("http://www.okkam.org/ontology_restaurant2.owl#name",)
        )

    return attrs


def _collect_literals(graph: Graph) -> Dict[str, Dict[str, List[str]]]:
    literal_map: Dict[str, Dict[str, List[str]]] = {}
    for subj, pred, obj in graph.triples((None, None, None)):
        if isinstance(subj, URIRef) and isinstance(obj, Literal):
            literal_map.setdefault(str(subj), {}).setdefault(str(pred), []).append(str(obj))
    return literal_map


def _collect_entities(graph: Graph) -> Set[str]:
    entities: Set[str] = set()
    for subj, _, obj in graph.triples((None, None, None)):
        if isinstance(subj, URIRef):
            entities.add(str(subj))
        if isinstance(obj, URIRef):
            entities.add(str(obj))
    return entities


def _choose_attribute(
    attr_values: Mapping[str, Sequence[str]],
    priorities: Sequence[str],
) -> Optional[str]:
    for attr in priorities:
        values = attr_values.get(attr)
        if values:
            return values[0]
    for values in attr_values.values():
        if values:
            return values[0]
    return None


def build_graph_entity_texts(
    graph: Graph,
    dataset_name: str,
    kg_index: int,
) -> Dict[str, str]:
    """Return representative texts for every entity in the graph."""

    priorities = _priority_for_dataset(dataset_name, kg_index)
    literals = _collect_literals(graph)
    entities = _collect_entities(graph)

    texts: Dict[str, str] = {}
    for entity in entities:
        attr_values = literals.get(entity, {})
        best_attr = _choose_attribute(attr_values, priorities)
        if best_attr:
            texts[entity] = best_attr
        else:
            texts[entity] = friendly_name(entity, dataset_name)
    return texts
