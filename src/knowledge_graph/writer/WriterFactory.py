import importlib
import pkgutil

class WriterFactory:
    _writers = {}

    @classmethod
    def register_writer(cls, file_type: str, writer_cls):
        cls._writers[file_type.lower()] = writer_cls

    @classmethod
    def create_writer(cls, file_type: str, *args, **kwargs):
        if not cls._writers:
            cls._autoload("src.knowledge_graph.writer")
        writer_cls = cls._writers.get(file_type.lower())
        if writer_cls is None:
            raise ValueError(f"Unknown file type: {file_type}")
        return writer_cls(*args, **kwargs)

    @classmethod
    def _autoload(cls, package_name: str):
        package = importlib.import_module(package_name)
        if not hasattr(package, "__path__"):
            return
        for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            importlib.import_module(module_name)
