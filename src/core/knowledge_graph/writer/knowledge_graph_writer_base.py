from abc import ABC, abstractmethod
from src.core.knowledge_graph.writer.knowledge_graph_writer_factory import KnowledgeGraphWriterFactory


class KnowledgeGraphWriter(ABC):
    file_type = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.file_type:
            KnowledgeGraphWriterFactory.register_writer(cls.file_type, cls)
    @abstractmethod
    def write(self, *args, **kwargs):
        pass
