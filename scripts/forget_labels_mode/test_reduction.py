#!/usr/bin/env python3
"""Test the ForgetLabelsReducer logic."""

import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rdflib import URIRef, Literal

from src.core.dataset import Dataset
from src.core.knowledge_graph import KnowledgeGraph
from scripts.forget_labels_mode.reducer import ForgetLabelsReducer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_dummy_dataset():
    """Create a small dummy dataset for testing."""
    kg_src = KnowledgeGraph()
    kg_tgt = KnowledgeGraph()
    
    # 10 aligned pairs
    aligned_entities = []
    for i in range(10):
        src_uri = URIRef(f"http://example.org/src/{i}")
        tgt_uri = URIRef(f"http://example.org/tgt/{i}")
        aligned_entities.append((src_uri, tgt_uri))
        
        # Add some triples for these entities
        kg_src.add((src_uri, URIRef("http://example.org/p"), Literal(f"val_src_{i}")))
        kg_tgt.add((tgt_uri, URIRef("http://example.org/p"), Literal(f"val_tgt_{i}")))

    return Dataset(kg_src, kg_tgt, aligned_entities)

def test_reduction():
    logger.info("Creating dummy dataset...")
    dataset = create_dummy_dataset()
    
    original_pairs = len(dataset.aligned_entities)
    original_src_triples = len(dataset.knowledge_graph_source)
    original_tgt_triples = len(dataset.knowledge_graph_target)
    
    logger.info(f"Original pairs: {original_pairs}")
    logger.info(f"Original src triples: {original_src_triples}")
    logger.info(f"Original tgt triples: {original_tgt_triples}")
    
    # Configure reducer with ratio 0.5
    config = {
        "reduction": {
            "method": "forget_labels",
            "ratio": 0.5,
            "random_seed": 42
        }
    }
    
    logger.info("Running ForgetLabelsReducer (ratio=0.5)...")
    reducer = ForgetLabelsReducer(config)
    reduced_dataset = reducer.reduce(dataset)
    
    final_pairs = len(reduced_dataset.aligned_entities)
    final_src_triples = len(reduced_dataset.knowledge_graph_source)
    final_tgt_triples = len(reduced_dataset.knowledge_graph_target)
    
    logger.info(f"Final pairs: {final_pairs}")
    logger.info(f"Final src triples: {final_src_triples}")
    logger.info(f"Final tgt triples: {final_tgt_triples}")
    
    # Assertions
    assert final_pairs == 5, f"Expected 5 pairs, got {final_pairs}"
    assert final_src_triples == original_src_triples, f"Source graph modified! {final_src_triples} != {original_src_triples}"
    assert final_tgt_triples == original_tgt_triples, f"Target graph modified! {final_tgt_triples} != {original_tgt_triples}"
    
    logger.info("✅ TEST PASSED: Pairs reduced correctly, graphs preserved.")

if __name__ == "__main__":
    try:
        test_reduction()
    except AssertionError as e:
        logger.error(f"❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        sys.exit(1)
