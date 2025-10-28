from abc import ABC, abstractmethod
from src.knowledge_graph.writer.WriterFactory import WriterFactory


class Writer(ABC):
    file_type = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.file_type:
            WriterFactory.register_writer(cls.file_type, cls)

    @abstractmethod
    def write(self, *args, **kwargs):
        pass