import io
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from langdetect import detect as _detect_language
except ModuleNotFoundError:  # pragma: no cover - dependency might be unavailable
    _detect_language = None

from src.alignment_models.methods.hybea import runtime as cfg
from src.logger import get_logger

logger = get_logger(__name__)

_LEGACY_LANG_CACHE = None


def _load_legacy_language_cache():
    """Load language annotations from legacy HyBEA analysis files for fallback."""
    global _LEGACY_LANG_CACHE
    if _LEGACY_LANG_CACHE is not None:
        return _LEGACY_LANG_CACHE

    cache = {}
    try:
        current_file = Path(__file__).resolve()
    except OSError:  # pragma: no cover - defensive guard
        _LEGACY_LANG_CACHE = cache
        return cache

    legacy_root = None
    for parent in current_file.parents:
        candidate = parent / "hybea" / "data" / "entity_names"
        if candidate.exists():
            legacy_root = candidate
            break

    if legacy_root is None:
        _LEGACY_LANG_CACHE = cache
        return cache

    for workbook in legacy_root.glob("**/*_analysis.xlsx"):
        try:
            df = pd.read_excel(workbook)
        except Exception:  # pragma: no cover - IO failures or missing engine
            logger.debug("[HyBEA][names] Unable to read legacy workbook %s", workbook)
            continue

        for _, row in df.iterrows():
            key = str(row.get("replaced_puncs", "")).strip()
            lang = str(row.get("Lang", "")).strip()
            if key and lang and key not in cache:
                cache[key] = lang

    _LEGACY_LANG_CACHE = cache
    logger.debug("[HyBEA][names] Loaded %d legacy language entries", len(cache))
    return cache


"""
    Language detection
"""
def lang_detect(s):
    text = "" if s is None else str(s)

    if _detect_language is not None:
        try:
            return _detect_language(text)
        except Exception:  # pragma: no cover - keep behaviour identical
            logger.debug("[HyBEA][names] langdetect failed for '%s'", text, exc_info=True)

    legacy_cache = _load_legacy_language_cache()
    lang = legacy_cache.get(text.strip())
    if lang:
        return lang

    return "Other"


def to_delete(lang):
    return lang in ["ne", "ml", "ja", "hi", "pa", "ko", "ar", "fa", "zh-tw", "zh-cn", "ta", "ru", "bg", "pt", "te", "Other"]



def clean_text(text):

    text = text.replace("@eng", "")
    text = text.replace("@en", "")
    text = text.replace("@la", "")
    text = text.replace("<http://www.w3.org/2001/XMLSchema#date>", "")
    text = text.replace("<http://www.w3.org/2001/XMLSchema#string>", "")

    if text == "None":
        text = "None"

    clean_text = re.sub(r'[^\w\s\']', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    return clean_text


"""
    Find the name of the entity by the postfix of its url
"""    
def find_url_name(url):
    # Ensure url is a string to avoid attribute errors when IDs are integers or None
    if url is None:
        return ""
    url = str(url)

    url = url.replace("dbp:", "")
    cleaned_text = clean_text(url.split("/")[-1])
    cleaned_text = cleaned_text.replace("_", " ")

    return cleaned_text


"""
    Map ids to uris
"""
def get_ids_to_uris(DATASET, num):
    data_path = cfg.DATA_TARGET + "/knowformer_data/" + cfg.DATASET
    ids_to_uris = {}
    with open(data_path + "/ent_ids_" + num) as fp:
        for line in fp:
            ids_to_uris[int(line.split("\t")[0])] = line.split("\t")[1].rstrip()
    return ids_to_uris


"""
    dataframe with attribute triples
    the following dictionary with the attribute types for each entity:
        {
            "http://dbpedia.org/resource/Captain_Pirate": [ "http://dbpedia.org/ontology/imdbId", ...]
        }
"""
def load_attr_graph(kg_id):
    data_path = cfg.DATA_TARGET + "/knowformer_data/" + cfg.DATASET + "/attr_triples_" + kg_id
    attr_df = pd.read_csv(data_path,  sep='\t', names=["e1", "attr", "val"])
    attr_dict = {}
    with open(data_path, "r") as fp:
        for line in fp:
            ent = line.split("\t")[0]
            attr = line.split("\t")[1]

            if ent not in attr_dict.keys():
                attr_dict[ent] = list()

            attr_dict[ent].append(attr)

    return attr_df, attr_dict

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logger.debug("[HyBEA][names] Created folder %s", folder_path)
    else:
        logger.debug("[HyBEA][names] Folder %s already exists", folder_path)
