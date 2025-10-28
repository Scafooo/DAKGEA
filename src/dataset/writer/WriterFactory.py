class WriterFactory:

    _writers = {}

    @classmethod
    def register_writer(cls, file_type: str, writer_cls):
        cls._writers[file_type.lower()] = writer_cls


    @classmethod
    def create_writer(cls, file_type: str, *args, **kwargs):
        writer_cls = cls._writers.get(file_type.lower())
        if writer_cls is None:
            raise ValueError(f"Unknown file type: {file_type}")
        return writer_cls(*args, **kwargs)