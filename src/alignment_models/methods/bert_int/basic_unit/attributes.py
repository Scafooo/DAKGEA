"""Attribute extraction utilities for BERT-INT Basic Unit.

This module implements the attribute priority logic from the reference implementation,
which selects the most informative attribute for each entity based on predicate priority.
"""

from typing import Dict, Tuple


# Predicate priorities for D_W (DBPedia-Wikidata) datasets
# Lower number = higher priority
PRIORITIES_D_W_KG1 = {
    "http://xmlns.com/foaf/0.1/name": 0,
    "http://dbpedia.org/ontology/birthName": 1,
    "http://purl.org/dc/elements/1.1/description": 2,
    "http://xmlns.com/foaf/0.1/nick": 3,
    "http://xmlns.com/foaf/0.1/givenName": 4,
    "http://dbpedia.org/ontology/leaderTitle": 5,
    "http://dbpedia.org/ontology/alias": 6,
    "http://dbpedia.org/ontology/motto": 7,
    "http://dbpedia.org/ontology/office": 7,
}

PRIORITIES_D_W_KG2 = {
    "http://www.wikidata.org/entity/P373": 0,
    "http://schema.org/description": 1,
    "http://www.wikidata.org/entity/P1476": 2,
    "http://www.wikidata.org/entity/P935": 3,
    "http://www.w3.org/2004/02/skos/core#altLabel": 4,
}

# Default priorities for unknown datasets (fallback to common predicates)
PRIORITIES_DEFAULT_KG1 = {
    "http://xmlns.com/foaf/0.1/name": 0,
    "http://www.w3.org/2000/01/rdf-schema#label": 1,
    "http://dbpedia.org/ontology/birthName": 2,
    "http://purl.org/dc/elements/1.1/description": 3,
}

PRIORITIES_DEFAULT_KG2 = {
    "http://www.w3.org/2000/01/rdf-schema#label": 0,
    "http://schema.org/name": 1,
    "http://schema.org/description": 2,
}


def get_priority_map(attr_path: str) -> Dict[str, int]:
    """Get predicate priority map based on dataset and KG.

    Args:
        attr_path: Path to attribute triples file

    Returns:
        Dictionary mapping predicate URIs to priority values (lower = higher priority)
    """
    # Check for D_W datasets (DBPedia-Wikidata)
    if "D_W" in attr_path:
        if "attr_triples1" in attr_path:
            return PRIORITIES_D_W_KG1
        else:
            return PRIORITIES_D_W_KG2

    # Check for D_Y datasets (DBPedia-YAGO)
    elif "D_Y" in attr_path:
        if "attr_triples1" in attr_path:
            return PRIORITIES_D_W_KG1  # Same as D_W KG1
        else:
            return {
                "skos:prefLabel": 0,
                "redirectedFrom": 1,
                "hasFamilyName": 2,
                "hasGivenName": 3,
                "hasMotto": 4,
            }

    # Fallback to default priorities
    else:
        if "attr_triples1" in attr_path or "1" in attr_path:
            return PRIORITIES_DEFAULT_KG1
        else:
            return PRIORITIES_DEFAULT_KG2


def extract_priority_attributes(attr_file: str) -> Dict[str, str]:
    """Extract the highest-priority attribute value for each entity.

    For each entity, selects the attribute with the highest priority predicate
    (lowest priority number). If an entity has multiple attributes, only the
    one with the most informative predicate is kept.

    Args:
        attr_file: Path to attribute triples file (entity, predicate, value format)

    Returns:
        Dictionary mapping entity URIs to their best attribute values

    Example:
        >>> attrs = extract_priority_attributes("path/to/attr_triples1")
        >>> attrs["http://dbpedia.org/resource/Jake_Bugg"]
        'English singer-songwriter'
    """
    priority = get_priority_map(attr_file)

    # entity -> (predicate, value)
    ents_attr: Dict[str, Tuple[str, str]] = {}

    try:
        with open(attr_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) != 3:
                    continue

                entity, predicate, value = parts

                # Only consider predicates in our priority list
                if predicate not in priority:
                    continue

                # Clean value (remove datatype annotations like ^^<http://...>)
                if "^^<" in value:
                    value = value.split("^^<")[0].strip('"')

                # Keep this attribute if:
                # 1. Entity has no attribute yet, OR
                # 2. This predicate has higher priority (lower number)
                if entity not in ents_attr:
                    ents_attr[entity] = (predicate, value)
                else:
                    current_pred, _ = ents_attr[entity]
                    if priority[predicate] < priority[current_pred]:
                        ents_attr[entity] = (predicate, value)

    except FileNotFoundError:
        # If file doesn't exist, return empty dict
        return {}

    # Return just entity -> value mapping
    return {entity: value for entity, (_, value) in ents_attr.items()}
