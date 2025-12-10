"""Test PLM augmentation with reduction on BBC-DB dataset.

This test uses the configuration from config/augmentation/plm.yaml
to avoid parameter duplication.
"""
from pathlib import Path
import yaml

from src.augmentation.methods.plm import PLMAugmenter
from src.core.dataset.reader.dataset_reader_factory import DatasetReaderFactory
from src.augmentation.methods.plm.set_knowledge_graph.set_knowledge_graph import SetKnowledgeGraph
from src.reduction.registry import load_builtin_reducers, REDUCTION_REGISTRY

import logging
logging.getLogger("rdflib.term").setLevel(logging.ERROR)

# ----------------------------------------------------------------------------
# 1. Load configuration from plm.yaml
# ----------------------------------------------------------------------------
config_path = Path(__file__).parent.parent / "config" / "augmentation" / "plm.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Add experiment seed
config["experiment"] = {"seed": 11037}

# ----------------------------------------------------------------------------
# 2. Dataset loading
# ----------------------------------------------------------------------------
reader = DatasetReaderFactory.create_reader("bert_int")
dataset = reader.read("/home/federico/Programming/Python/DAKGEA/data/raw/openea/D_W_15K_V1/attribute_data")

# ----------------------------------------------------------------------------
# 3. Reduction (optional)
# ----------------------------------------------------------------------------
load_builtin_reducers()
reducer = REDUCTION_REGISTRY.get("random_entities")(
    {"reduction": {"target_entities": 400}, "experiment": {"seed": 11037}}
)
reducer.reduce(dataset)

# ----------------------------------------------------------------------------
# 4. SetKnowledgeGraph creation
# ----------------------------------------------------------------------------
skg = SetKnowledgeGraph.from_dataset(dataset)

# ----------------------------------------------------------------------------
# 5. PLM Augmentation (using config from plm.yaml)
# ----------------------------------------------------------------------------
augmenter = PLMAugmenter(config)
dataset_augmented = augmenter.augment(dataset)



