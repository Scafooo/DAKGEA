from src.augmentation.set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph
from src.core import DatasetReaderFactory

reader = DatasetReaderFactory.create_reader("bert_int")
dataset = reader.read("/home/federico/Programming/Python/DAKGEA/data/test/attribute_data")

skg = SetKnowledgeGraph()
skg = skg.from_dataset(dataset)

for triple in skg.triples((None, None, None)):
    print(triple)