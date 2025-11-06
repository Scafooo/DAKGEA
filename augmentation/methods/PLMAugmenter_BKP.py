import random
from rdflib import Literal
from rdflib import URIRef

from augmentation.util.predicate_matching import match_relations_with_attrnames, match_relations
from src.core.dataset import Dataset
from src.core.knowledge_graph import KnowledgeGraph
from src.logger import get_logger

logger = get_logger(__name__)

def augment(dataset: Dataset, aug_percentage: float, seed: int = 42):
    logger.info("Augmenting dataset")

    aligned_list = list(dataset.aligned_entities)
    aligned_list.sort()

    # Select random candidates
    rng = random.Random(seed)
    total = len(dataset.aligned_entities)
    sample_size = int(total * aug_percentage)
    candidates = rng.sample(aligned_list, sample_size)

    matches = match_relations(dataset)

    for (p1, p2), score in matches:
        print(f"{p1:25s} ↔ {p2:25s}   score = {score:.3f}")


    # for s,t in candidates:
    #     print("Start ")
    info_e1 = _get_entity_info(candidates[10][0], dataset.knowledge_graph_source)
    info_e2 = _get_entity_info(candidates[10][1], dataset.knowledge_graph_target)

    common_predicates = set()
    for s1,p1,o1 in info_e1:
        for s2,p2,o2 in info_e2:
            if any((p1, p2) == pair for pair, score in matches):
                print(f"match: {p1:<25s} {p2:<25s} {s1:<25s} = {s2:<25s} or {o1:<25s} = {o2:<25s}")
                common_predicates.add((p1,p2))

    logger.info("Dataset augmented")


def _get_entity_info(entity: URIRef, kg: KnowledgeGraph):
    ret = set()



    for s, p, o in kg.triples((entity, None, None)):
        ret.add((s, p, o))
    for s, p, o in kg.triples((None, None, entity)):
        ret.add((s, p, o))

    for s, p, o in ret:
        print(str(s), str(p), str(o))

    return ret


