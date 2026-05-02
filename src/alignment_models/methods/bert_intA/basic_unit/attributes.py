"""Attribute extraction utilities for BERT-INT Basic Unit.

This module implements the attribute priority logic from the reference implementation,
which selects the most informative attribute for each entity based on predicate priority.
"""

from typing import Dict, Tuple


# Predicate priorities by dataset
# Lower number = higher priority
# Based on BERT-INT reference implementation

# BBC dataset
PRIORITIES_BBC_KG1 = {
    "http://purl.org/dc/elements/1.1/title": 0,
    "http://xmlns.com/foaf/0.1/name": 1,
    "http://open.vocab.org/terms/sortlabel": 2,
}

PRIORITIES_BBC_KG2 = {
    "prop:title": 0,
}

# EN_JA (English-Japanese)
PRIORITIES_EN_JA_KG1 = {
    "http://xmlns.com/foaf/0.1/name": 0,
    "http://dbpedia.org/ontology/birthName": 1,
    "http://xmlns.com/foaf/0.1/nick": 2,
    "http://dbpedia.org/ontology/synonym": 3,
    "http://dbpedia.org/ontology/alias": 4,
    "http://dbpedia.org/ontology/office": 5,
    "http://dbpedia.org/ontology/background": 5,
    "http://dbpedia.org/ontology/leaderTitle": 5,
    "http://dbpedia.org/ontology/orderInOffice": 5,
}

PRIORITIES_EN_JA_KG2 = {
    "http://xmlns.com/foaf/0.1/name": 0,
    "http://dbpedia.org/ontology/title": 1,
    "http://dbpedia.org/ontology/commonName": 2,
    "http://xmlns.com/foaf/0.1/nick": 3,
    "http://dbpedia.org/ontology/givenName": 4,
    "http://dbpedia.org/ontology/alias": 5,
    "http://dbpedia.org/ontology/background": 6,
    "http://dbpedia.org/ontology/purpose": 6,
}

# EN_DE (English-German)
PRIORITIES_EN_DE_KG1 = {
    "http://xmlns.com/foaf/0.1/name": 0,
    "http://dbpedia.org/ontology/title": 1,
    "http://dbpedia.org/ontology/birthName": 2,
    "http://xmlns.com/foaf/0.1/nick": 3,
    "http://dbpedia.org/ontology/office": 4,
    "http://dbpedia.org/ontology/leaderTitle": 5,
    "http://dbpedia.org/ontology/orderInOffice": 5,
}

PRIORITIES_EN_DE_KG2 = {
    "http://xmlns.com/foaf/0.1/name": 0,
    "http://dbpedia.org/ontology/originalTitle": 1,
    "http://xmlns.com/foaf/0.1/nick": 2,
    "http://dbpedia.org/ontology/motto": 3,
    "http://dbpedia.org/ontology/leaderTitle": 4,
}

# EN_FR (English-French)
PRIORITIES_EN_FR_KG1 = {
    "http://xmlns.com/foaf/0.1/name": 0,
    "http://dbpedia.org/ontology/title": 1,
    "http://dbpedia.org/ontology/birthName": 2,
    "http://xmlns.com/foaf/0.1/nick": 3,
    "http://dbpedia.org/ontology/office": 4,
    "http://dbpedia.org/ontology/leaderTitle": 5,
    "http://dbpedia.org/ontology/orderInOffice": 5,
}

PRIORITIES_EN_FR_KG2 = {
    "http://xmlns.com/foaf/0.1/name": 0,
    "http://dbpedia.org/ontology/originalTitle": 1,
    "http://xmlns.com/foaf/0.1/nick": 2,
    "http://dbpedia.org/ontology/motto": 3,
    "http://dbpedia.org/ontology/leaderTitle": 4,
}

# DBP_en_WD_en / D_W (DBPedia-Wikidata)
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

# DBP_en_YG_en / D_Y (DBPedia-YAGO)
PRIORITIES_D_Y_KG2 = {
    "skos:prefLabel": 0,
    "redirectedFrom": 1,
    "hasFamilyName": 2,
    "hasGivenName": 3,
    "hasMotto": 4,
}

# Movie benchmark datasets (IMDB-TMDB, IMDB-TVDB, TMDB-TVDB)
PRIORITIES_MOVIE_KG1 = {
    "https://www.scads.de/movieBenchmark/ontology/name": 0,
    "https://www.scads.de/movieBenchmark/ontology/title": 1,
    "https://www.scads.de/movieBenchmark/ontology/originalTitle": 2,
}

PRIORITIES_MOVIE_KG2 = {
    "https://www.scads.de/movieBenchmark/ontology/name": 0,
    "https://www.scads.de/movieBenchmark/ontology/title": 1,
    "https://www.scads.de/movieBenchmark/ontology/originalTitle": 2,
}

# Restaurant dataset (REST)
PRIORITIES_REST_KG1 = {
    "http://www.okkam.org/ontology_restaurant1.owl#name": 0,
}

PRIORITIES_REST_KG2 = {
    "http://www.okkam.org/ontology_restaurant2.owl#name": 0,
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
    # Normalize path for case-insensitive matching
    attr_path_lower = attr_path.lower()
    is_kg1 = "attr_triples1" in attr_path_lower or "/1" in attr_path

    # Check for BBC dataset
    if "bbc" in attr_path_lower:
        return PRIORITIES_BBC_KG1 if is_kg1 else PRIORITIES_BBC_KG2

    # Check for EN_JA (English-Japanese)
    elif "en_ja" in attr_path_lower:
        return PRIORITIES_EN_JA_KG1 if is_kg1 else PRIORITIES_EN_JA_KG2

    # Check for EN_DE (English-German)
    elif "en_de" in attr_path_lower:
        return PRIORITIES_EN_DE_KG1 if is_kg1 else PRIORITIES_EN_DE_KG2

    # Check for EN_FR (English-French)
    elif "en_fr" in attr_path_lower:
        return PRIORITIES_EN_FR_KG1 if is_kg1 else PRIORITIES_EN_FR_KG2

    # Check for DBP_en_WD_en (alternative name for D_W)
    elif "dbp_en_wd_en" in attr_path_lower:
        return PRIORITIES_D_W_KG1 if is_kg1 else PRIORITIES_D_W_KG2

    # Check for D_W datasets (DBPedia-Wikidata)
    elif "d_w" in attr_path_lower:
        return PRIORITIES_D_W_KG1 if is_kg1 else PRIORITIES_D_W_KG2

    # Check for DBP_en_YG_en (alternative name for D_Y)
    elif "dbp_en_yg_en" in attr_path_lower:
        return PRIORITIES_D_W_KG1 if is_kg1 else PRIORITIES_D_Y_KG2

    # Check for D_Y datasets (DBPedia-YAGO)
    elif "d_y" in attr_path_lower:
        return PRIORITIES_D_W_KG1 if is_kg1 else PRIORITIES_D_Y_KG2

    # Check for movie benchmark datasets (IMDB-TMDB, IMDB-TVDB, TMDB-TVDB)
    elif any(x in attr_path_lower for x in ["imdb-tmdb", "imdb-tvdb", "tmdb-tvdb"]):
        return PRIORITIES_MOVIE_KG1 if is_kg1 else PRIORITIES_MOVIE_KG2

    # Check for restaurant dataset
    elif "rest" in attr_path_lower:
        return PRIORITIES_REST_KG1 if is_kg1 else PRIORITIES_REST_KG2

    # Fallback to default priorities
    else:
        return PRIORITIES_DEFAULT_KG1 if is_kg1 else PRIORITIES_DEFAULT_KG2


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
