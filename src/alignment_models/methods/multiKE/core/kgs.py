"""KGs class: builds the combined two-KG structure with ID assignment."""

from __future__ import annotations

from .kg import KG
from .read import (
    generate_mapping_id,
    generate_sharing_id,
    uris_attribute_triple_2ids,
    uris_pair_2ids,
    uris_relation_triple_2ids,
    generate_sup_relation_triples,
    generate_sup_attribute_triples,
)


class KGs:
    def __init__(self, kg1: KG, kg2: KG, train_links, valid_links, test_links=None,
                 mode: str = "swapping", ordered: bool = True):
        if mode == "sharing":
            ent_ids1, ent_ids2 = generate_sharing_id(
                train_links, kg1.relation_triples_set, kg1.entities_set,
                kg2.relation_triples_set, kg2.entities_set, ordered=ordered)
            rel_ids1, rel_ids2 = generate_sharing_id(
                [], kg1.relation_triples_set, kg1.relations_set,
                kg2.relation_triples_set, kg2.relations_set, ordered=ordered)
            attr_ids1, attr_ids2 = generate_sharing_id(
                [], kg1.attribute_triples_set, kg1.attributes_set,
                kg2.attribute_triples_set, kg2.attributes_set, ordered=ordered)
        else:
            ent_ids1, ent_ids2 = generate_mapping_id(
                kg1.relation_triples_set, kg1.entities_set,
                kg2.relation_triples_set, kg2.entities_set, ordered=ordered)
            rel_ids1, rel_ids2 = generate_mapping_id(
                kg1.relation_triples_set, kg1.relations_set,
                kg2.relation_triples_set, kg2.relations_set, ordered=ordered)
            attr_ids1, attr_ids2 = generate_mapping_id(
                kg1.attribute_triples_set, kg1.attributes_set,
                kg2.attribute_triples_set, kg2.attributes_set, ordered=ordered)

        id_rel_triples1 = uris_relation_triple_2ids(kg1.relation_triples_set, ent_ids1, rel_ids1)
        id_rel_triples2 = uris_relation_triple_2ids(kg2.relation_triples_set, ent_ids2, rel_ids2)
        id_attr_triples1 = uris_attribute_triple_2ids(kg1.attribute_triples_set, ent_ids1, attr_ids1)
        id_attr_triples2 = uris_attribute_triple_2ids(kg2.attribute_triples_set, ent_ids2, attr_ids2)

        self.uri_kg1 = kg1
        self.uri_kg2 = kg2

        kg1 = KG(id_rel_triples1, id_attr_triples1)
        kg2 = KG(id_rel_triples2, id_attr_triples2)
        kg1.set_id_dict(ent_ids1, rel_ids1, attr_ids1)
        kg2.set_id_dict(ent_ids2, rel_ids2, attr_ids2)

        self.uri_train_links = train_links
        self.uri_valid_links = valid_links
        self.train_links = uris_pair_2ids(train_links, ent_ids1, ent_ids2)
        self.valid_links = uris_pair_2ids(valid_links, ent_ids1, ent_ids2)
        self.train_entities1 = [l[0] for l in self.train_links]
        self.train_entities2 = [l[1] for l in self.train_links]
        self.valid_entities1 = [l[0] for l in self.valid_links]
        self.valid_entities2 = [l[1] for l in self.valid_links]

        if mode == "swapping":
            sup1, sup2 = generate_sup_relation_triples(
                self.train_links, kg1.rt_dict, kg1.hr_dict, kg2.rt_dict, kg2.hr_dict)
            kg1.add_sup_relation_triples(sup1)
            kg2.add_sup_relation_triples(sup2)
            sup1, sup2 = generate_sup_attribute_triples(self.train_links, kg1.av_dict, kg2.av_dict)
            kg1.add_sup_attribute_triples(sup1)
            kg2.add_sup_attribute_triples(sup2)

        self.kg1 = kg1
        self.kg2 = kg2

        self.test_links = []
        self.test_entities1 = []
        self.test_entities2 = []
        if test_links is not None:
            self.uri_test_links = test_links
            self.test_links = uris_pair_2ids(test_links, ent_ids1, ent_ids2)
            self.test_entities1 = [l[0] for l in self.test_links]
            self.test_entities2 = [l[1] for l in self.test_links]

        self.useful_entities_list1 = self.train_entities1 + self.valid_entities1 + self.test_entities1
        self.useful_entities_list2 = self.train_entities2 + self.valid_entities2 + self.test_entities2

        self.entities_num = len(self.kg1.entities_set | self.kg2.entities_set)
        self.relations_num = len(self.kg1.relations_set | self.kg2.relations_set)
        self.attributes_num = len(self.kg1.attributes_set | self.kg2.attributes_set)
