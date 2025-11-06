from augmentation.methods.plm.set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph
from src.core import DatasetReaderFactory

reader = DatasetReaderFactory.create_reader("bert_int")
dataset = reader.read("/home/federico/Programming/Python/DAKGEA/Bert_int_reference/D_W_15K_V1/10/attribute_date")

skg = SetKnowledgeGraph()
skg = skg.from_dataset(dataset)

for triple in skg.triples((None, None, None)):
    print(triple)