from abc import ABC, abstractmethod
from src.core.knowledge_graph.reader.knowledge_graph_reader_factory import KnowledgeGraphReaderFactory


class KnowledgeGraphReader(ABC):
    file_type = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.file_type:
            KnowledgeGraphReaderFactory.register_reader(cls.file_type, cls)

    @abstractmethod
    def read(self, *args, **kwargs):
        pass
