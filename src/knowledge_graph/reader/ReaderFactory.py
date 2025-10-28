class ReaderFactory:

    _readers = {}

    @classmethod
    def register_reader(cls, file_type: str, reader_cls):
        cls._readers[file_type.lower()] = reader_cls


    @classmethod
    def create_reader(cls, file_type: str, *args, **kwargs):
        reader_cls = cls._readers.get(file_type.lower())
        if reader_cls is None:
            raise ValueError(f"Unknown file type: {file_type}")
        return reader_cls(*args, **kwargs)