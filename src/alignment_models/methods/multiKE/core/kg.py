"""KG data structure for MultiKE."""

from __future__ import annotations


def parse_triples(triples):
    subjects, predicates, objects = set(), set(), set()
    for s, p, o in triples:
        subjects.add(s)
        predicates.add(p)
        objects.add(o)
    return subjects, predicates, objects


class KG:
    def __init__(self, relation_triples, attribute_triples):
        self.entities_set = self.entities_list = None
        self.relations_set = self.relations_list = None
        self.attributes_set = self.attributes_list = None
        self.entities_num = self.relations_num = self.attributes_num = None
        self.relation_triples_num = self.attribute_triples_num = None
        self.local_relation_triples_num = self.local_attribute_triples_num = None

        self.entities_id_dict = None
        self.relations_id_dict = None
        self.attributes_id_dict = None

        self.rt_dict = self.hr_dict = None
        self.entity_relations_dict = None
        self.entity_attributes_dict = None
        self.av_dict = None

        self.sup_relation_triples_set = self.sup_relation_triples_list = None
        self.sup_attribute_triples_set = self.sup_attribute_triples_list = None

        self.relation_triples_set = self.relation_triples_list = None
        self.attribute_triples_set = self.attribute_triples_list = None
        self.local_relation_triples_set = self.local_relation_triples_list = None
        self.local_attribute_triples_set = self.local_attribute_triples_list = None

        self.set_relations(relation_triples)
        self.set_attributes(attribute_triples)

    def set_relations(self, relation_triples):
        self.relation_triples_set = set(relation_triples)
        self.relation_triples_list = list(self.relation_triples_set)
        self.local_relation_triples_set = self.relation_triples_set
        self.local_relation_triples_list = self.relation_triples_list

        heads, relations, tails = parse_triples(self.relation_triples_set)
        self.entities_set = heads | tails
        self.relations_set = relations
        self.entities_list = list(self.entities_set)
        self.relations_list = list(self.relations_set)
        self.entities_num = len(self.entities_set)
        self.relations_num = len(self.relations_set)
        self.relation_triples_num = len(self.relation_triples_set)
        self.local_relation_triples_num = len(self.local_relation_triples_set)
        self._generate_relation_triple_dict()
        self._parse_relations()

    def set_attributes(self, attribute_triples):
        self.attribute_triples_set = set(attribute_triples)
        self.attribute_triples_list = list(self.attribute_triples_set)
        self.local_attribute_triples_set = self.attribute_triples_set
        self.local_attribute_triples_list = self.attribute_triples_list

        entities, attributes, values = parse_triples(self.attribute_triples_set)
        self.attributes_set = attributes
        self.attributes_list = list(self.attributes_set)
        self.attributes_num = len(self.attributes_set)
        self.attribute_triples_num = len(self.attribute_triples_set)
        self.local_attribute_triples_num = len(self.local_attribute_triples_set)
        self._generate_attribute_triple_dict()
        self._parse_attributes()

    def _generate_relation_triple_dict(self):
        self.rt_dict, self.hr_dict = {}, {}
        for h, r, t in self.local_relation_triples_list:
            self.rt_dict.setdefault(h, set()).add((r, t))
            self.hr_dict.setdefault(t, set()).add((h, r))

    def _generate_attribute_triple_dict(self):
        self.av_dict = {}
        for h, a, v in self.local_attribute_triples_list:
            self.av_dict.setdefault(h, set()).add((a, v))

    def _parse_relations(self):
        self.entity_relations_dict = {}
        for ent, attr, _ in self.local_relation_triples_set:
            self.entity_relations_dict.setdefault(ent, set()).add(attr)

    def _parse_attributes(self):
        self.entity_attributes_dict = {}
        for ent, attr, _ in self.local_attribute_triples_set:
            self.entity_attributes_dict.setdefault(ent, set()).add(attr)

    def set_id_dict(self, entities_id_dict, relations_id_dict, attributes_id_dict):
        self.entities_id_dict = entities_id_dict
        self.relations_id_dict = relations_id_dict
        self.attributes_id_dict = attributes_id_dict

    def add_sup_relation_triples(self, sup_triples):
        self.sup_relation_triples_set = set(sup_triples)
        self.sup_relation_triples_list = list(self.sup_relation_triples_set)
        self.relation_triples_set |= self.sup_relation_triples_set
        self.relation_triples_list = list(self.relation_triples_set)
        self.relation_triples_num = len(self.relation_triples_list)

    def add_sup_attribute_triples(self, sup_triples):
        self.sup_attribute_triples_set = set(sup_triples)
        self.sup_attribute_triples_list = list(self.sup_attribute_triples_set)
        self.attribute_triples_set |= self.sup_attribute_triples_set
        self.attribute_triples_list = list(self.attribute_triples_set)
        self.attribute_triples_num = len(self.attribute_triples_list)
