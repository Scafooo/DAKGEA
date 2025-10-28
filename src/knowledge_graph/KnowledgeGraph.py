from rdflib import *


class KnowledgeGraph(Graph):

    def __init__(self):
        super().__init__()
        self.attr_to_name = dict()

    def add_attribute_triples(self, triple):
        self.add((URIRef(triple[0]), URIRef(triple[1]), Literal(triple[2])))

    def add_relation_triples(self, triple):
        self.add((URIRef(triple[0]), URIRef(triple[1]), URIRef(triple[2])))