from src.knowledge_graph.KnowledgeGraph import KnowledgeGraph

class Dataset:

    def __init__(self, knowledge_graph_source: KnowledgeGraph, knowledge_graph_target: KnowledgeGraph, aligned_entities):
        self.knowledge_graph_source = knowledge_graph_source
        self.knowledge_graph_target = knowledge_graph_target
        self.aligned_entities = aligned_entities