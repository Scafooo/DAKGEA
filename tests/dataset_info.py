from src.core.dataset.reader.dataset_reader_factory import DatasetReaderFactory

import logging
logging.getLogger("rdflib.term").setLevel(logging.ERROR)

# ----------------------------------------------------------------------------
# 2. Dataset loading
# ----------------------------------------------------------------------------
reader = DatasetReaderFactory.create_reader("bert_int")
reader.read("/home/federico/Programming/Python/DAKGEA/data/raw/openea/BBC_DB/attribute_data")
reader.read("/home/federico/Programming/Python/DAKGEA/data/raw/openea/D_W_15K_V1/attribute_data")
reader.read("/home/federico/Programming/Python/DAKGEA/data/raw/openea/D_W_15K_V2/attribute_data")
reader.read("/home/federico/Programming/Python/DAKGEA/data/raw/openea/ICEW_WIKI/attribute_data")
reader.read("/home/federico/Programming/Python/DAKGEA/data/raw/openea/ICEW_YAGO/attribute_data")