from .utils import *
from src.alignment_models.methods.hybea import runtime as cfg
from src.logger import get_logger

logger = get_logger(__name__)

"""
    Find the url of an entity by mapping from id --> url (if need)
"""


def find_url(e1):
    ids_to_uris_1 = get_ids_to_uris(cfg.DATASET, "1")
    ids_to_uris_2 = get_ids_to_uris(cfg.DATASET, "2")

    if e1 in ids_to_uris_1:
        return ids_to_uris_1[e1]
    elif e1 in ids_to_uris_2:
        return ids_to_uris_2[e1]
    else:
        return e1


def export_names(df, kg_dest_path, attrs):
    ids_to_uris_1 = get_ids_to_uris(cfg.DATASET, "1")
    ids_to_uris_2 = get_ids_to_uris(cfg.DATASET, "2")

    # Keep only the manually selected attributes for name's information
    filtered_df_1 = df[df['attr'].isin(attrs)]
    # sort by entity
    sorted_df = filtered_df_1.sort_values(by='e1', ascending=True)
    # find the url of an entity
    sorted_df['URLs'] = sorted_df.apply(lambda x: find_url(x.e1), axis=1)

    # the following steps are for cleaning the text
    sorted_df['replaced_puncs'] = sorted_df.apply(lambda x: clean_text(x.val), axis=1)
    sorted_df["Lang"] = sorted_df.apply(lambda x: lang_detect(x.replaced_puncs), axis=1)
    sorted_df["Delete"] = sorted_df.apply(lambda x: to_delete(x.Lang), axis=1)

    # find name by url
    sorted_df['URL Names'] = sorted_df.apply(lambda x: find_url_name(x.URLs), axis=1)
    # export to excel
    sorted_df.to_excel(kg_dest_path)
    logger.info("[HyBEA][names] Exported name analysis to %s", kg_dest_path)


# Global mappings are no longer created at import time. Initialize them lazily.
ids_to_uris_1 = None
ids_to_uris_2 = None


def _ensure_ids_loaded():
    global ids_to_uris_1, ids_to_uris_2
    if ids_to_uris_1 is None or ids_to_uris_2 is None:
        ids_to_uris_1 = get_ids_to_uris(cfg.DATASET, "1")
        ids_to_uris_2 = get_ids_to_uris(cfg.DATASET, "2")


def run_name_analysis():
    _ensure_ids_loaded()
    ids_to_uris_1 = get_ids_to_uris(cfg.DATASET, "1")
    ids_to_uris_2 = get_ids_to_uris(cfg.DATASET, "2")

    attr_df_1, attr_dict_kg1 = load_attr_graph("1")
    attr_df_2, attr_dict_kg2 = load_attr_graph("2")

    if cfg.DATASET == "D_W_15K_V1":

        attrs_1 = ["http://xmlns.com/foaf/0.1/name", "http://xmlns.com/foaf/0.1/givenName",
                   "http://dbpedia.org/ontology/birthName",
                   "http://dbpedia.org/ontology/name", "http://dbpedia.org/ontology/longName",
                   "http://dbpedia.org/ontology/otherName", "http://dbpedia.org/ontology/teamName"]

        attrs_2 = ['http://www.wikidata.org/entity/P373', 'http://www.wikidata.org/entity/P1476',
                   'http://www.w3.org/2004/02/skos/core#altLabel', 'http://schema.org/description']

        kg1_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" +  cfg.DATASET + "/DBpedia_analysis.xlsx"
        kg2_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/Wikidata_analysis.xlsx"

    elif cfg.DATASET == "BBC_DB":

        attrs_1 = ['http://purl.org/dc/elements/1.1/title', 'http://xmlns.com/foaf/0.1/name',
                   'http://open.vocab.org/terms/sortlabel']

        attrs_2 = ['http://xmlns.com/foaf/0.1/name', 'prop:birthname', 'rdfs:label', 'prop:name',
                   'http://xmlns.com/foaf/0.1/givenname', 'http://xmlns.com/foaf/0.1/surname', 'prop:label',

                   'http://purl.org/dc/elements/1.1/description', 'prop:description', 'prop:shortdescription']

        kg1_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/BBC_analysis.xlsx"
        kg2_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/DBpedia_analysis.xlsx"

    elif cfg.DATASET == "D_W_15K_V2":

        attrs_1 = ["http://xmlns.com/foaf/0.1/name", "http://xmlns.com/foaf/0.1/givenName",
                   "http://dbpedia.org/ontology/birthName",
                   "http://dbpedia.org/ontology/longName"]

        attrs_2 = ['http://www.wikidata.org/entity/P373', 'http://www.wikidata.org/entity/P1476',
                   'http://www.w3.org/2004/02/skos/core#altLabel', 'http://schema.org/description']

        kg1_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/DBpedia_analysis.xlsx"
        kg2_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/Wikidata_analysis.xlsx"

    elif cfg.DATASET == "SRPRS_D_W_15K_V1":

        attrs_1 = ["http://dbpedia.org/ontology/title", "http://dbpedia.org/ontology/birthName",
                   "http://dbpedia.org/ontology/longName"]

        attrs_2 = ['http://www.wikidata.org/entity/P373', 'http://www.wikidata.org/entity/P1476']

        kg1_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/DBpedia_analysis.xlsx"
        kg2_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/Wikidata_analysis.xlsx"

    elif cfg.DATASET == "SRPRS_D_W_15K_V2":

        attrs_1 = ["http://dbpedia.org/ontology/title", "http://dbpedia.org/ontology/birthName",
                   "http://dbpedia.org/ontology/longName"]

        attrs_2 = ['http://www.wikidata.org/entity/P373', 'http://www.wikidata.org/entity/P1476']

        kg1_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/DBpedia_analysis.xlsx"
        kg2_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/Wikidata_analysis.xlsx"

    elif cfg.DATASET == "fr_en":

        attrs_1 = ["http://fr.dbpedia.org/property/titre", "http://xmlns.com/foaf/0.1/name",
                   "http://fr.dbpedia.org/property/name", "http://fr.dbpedia.org/property/label"]

        attrs_2 = ["http://dbpedia.org/property/title", "http://xmlns.com/foaf/0.1/name",
                   "http://dbpedia.org/property/name", "http://xmlns.com/foaf/0.1/givenName",
                   "http://dbpedia.org/ontology/birthName", "http://dbpedia.org/property/label"]

        kg1_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/fr_analysis.xlsx"
        kg2_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/en_analysis.xlsx"

    elif cfg.DATASET == "ja_en":

        attrs_1 = ["http://ja.dbpedia.org/property/title", "http://xmlns.com/foaf/0.1/name",
                   "http://ja.dbpedia.org/property/name", "http://xmlns.com/foaf/0.1/givenName",
                   "http://ja.dbpedia.org/property/label"]

        attrs_2 = ["http://dbpedia.org/property/title", "http://xmlns.com/foaf/0.1/name",
                   "http://dbpedia.org/property/name", "http://xmlns.com/foaf/0.1/givenName",
                   "http://dbpedia.org/property/label"]

        kg1_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/ja_analysis.xlsx"
        kg2_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/en_analysis.xlsx"

    elif cfg.DATASET == "zh_en":

        attrs_1 = ["http://zh.dbpedia.org/property/title", "http://xmlns.com/foaf/0.1/name",
                   "http://ja.dbpedia.org/property/name", "http://xmlns.com/foaf/0.1/givenName",
                   "http://ja.dbpedia.org/property/label"]

        attrs_2 = ["http://dbpedia.org/property/title", "http://xmlns.com/foaf/0.1/name",
                   "http://dbpedia.org/property/name", "http://xmlns.com/foaf/0.1/givenName",
                   "http://dbpedia.org/property/label"]

        kg1_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/zh_analysis.xlsx"
        kg2_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/en_analysis.xlsx"
    elif cfg.DATASET == "ICEW_WIKI":
        attrs_1 = ["has_name"]
        attrs_2 = ["has_name"]
        kg1_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/icew_analysis.xlsx"
        kg2_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/wiki_analysis.xlsx"
    elif cfg.DATASET == "ICEW_YAGO":
        attrs_1 = ["has_name"]
        attrs_2 = ["has_name"]
        kg1_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/icew_analysis.xlsx"
        kg2_dest_path = cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET + "/yago_analysis.xlsx"

    create_folder_if_not_exists(cfg.BASE_DIR + "/data/entity_names/" + str(round(cfg.SIZE_AFTER_REDUCTION_IN_PERCENTAGE,1)) + "/" + cfg.DATASET)

    logger.info("[HyBEA][names] Preparing name analysis for KG1 (%s)", kg1_dest_path)
    logger.debug("[HyBEA][names] KG1 tracked attributes: %s", attrs_1)
    export_names(attr_df_1, kg1_dest_path, attrs_1)

    logger.info("[HyBEA][names] Preparing name analysis for KG2 (%s)", kg2_dest_path)
    logger.debug("[HyBEA][names] KG2 tracked attributes: %s", attrs_2)
    export_names(attr_df_2, kg2_dest_path, attrs_2)
