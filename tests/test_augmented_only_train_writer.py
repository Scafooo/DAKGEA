#!/usr/bin/env python3
"""Test the augmented_only_train option in BERT-INT writer."""

import tempfile
from pathlib import Path

from rdflib import URIRef

from src.core.dataset import Dataset
from src.core.knowledge_graph import KnowledgeGraph
from src.core.dataset.writer.bert_int_writer import BertIntWriter
from src.utils.reader import read_tsv


def test_augmented_only_train():
    """Test that augmented entities go to training set when option is enabled."""

    # Create simple knowledge graphs
    kg1 = KnowledgeGraph()
    kg2 = KnowledgeGraph()

    # Add some original entities
    e1 = URIRef("http://example.org/e1")
    e2 = URIRef("http://example.org/e2")
    e3 = URIRef("http://example.org/e3")
    e4 = URIRef("http://example.org/e4")

    # Add some augmented entities (with _aug suffix)
    e1_aug = URIRef("http://example.org/e1_aug1")
    e2_aug = URIRef("http://example.org/e2_aug1")
    e3_aug = URIRef("http://example.org/e3_aug1")
    e4_aug = URIRef("http://example.org/e4_aug1")

    # Add triples to KGs
    rel = URIRef("http://example.org/rel")

    kg1.add((e1, rel, e2))
    kg1.add((e3, rel, e4))
    kg1.add((e1_aug, rel, e2))  # Augmented entities also in KG1
    kg1.add((e3_aug, rel, e4))

    kg2.add((e2, rel, e3))
    kg2.add((e4, rel, e1))
    kg2.add((e2_aug, rel, e3))  # Augmented entities also in KG2
    kg2.add((e4_aug, rel, e1))

    # Create dataset with original + augmented aligned pairs
    dataset = Dataset(
        knowledge_graph_source=kg1,
        knowledge_graph_target=kg2,
        aligned_entities=[
            (e1, e2),  # Original pair
            (e3, e4),  # Original pair
            (e1_aug, e2_aug),  # Augmented pair
            (e3_aug, e4_aug),  # Augmented pair
        ]
    )

    # Test with augmented_only_train=False (default behavior)
    with tempfile.TemporaryDirectory() as tmpdir:
        writer_default = BertIntWriter(augmented_only_train=False)
        writer_default.write(dataset, tmpdir)

        sup_pairs_default = read_tsv(Path(tmpdir) / "sup_pairs")
        ref_pairs_default = read_tsv(Path(tmpdir) / "ref_pairs")
        valid_pairs_default = read_tsv(Path(tmpdir) / "valid_pairs")

        total_default = len(sup_pairs_default) + len(ref_pairs_default) + len(valid_pairs_default)

        print(f"\n=== Default mode (augmented_only_train=False) ===")
        print(f"Training (sup): {len(sup_pairs_default)} pairs")
        print(f"Test (ref): {len(ref_pairs_default)} pairs")
        print(f"Valid: {len(valid_pairs_default)} pairs")
        print(f"Total: {total_default} pairs")

        # With 4 pairs, default split is:
        # 20% train = 0 pairs (int(4 * 0.2) = 0)
        # 70% test = 3 pairs (int(4 * 0.9) = 3, so 3-0 = 3)
        # 10% valid = 1 pair (4 - 3 = 1)
        # This is not ideal! Some augmented might be in test/valid

    # Test with augmented_only_train=True (new behavior)
    with tempfile.TemporaryDirectory() as tmpdir:
        writer_aug_train = BertIntWriter(augmented_only_train=True)
        writer_aug_train.write(dataset, tmpdir)

        sup_pairs_aug = read_tsv(Path(tmpdir) / "sup_pairs")
        ref_pairs_aug = read_tsv(Path(tmpdir) / "ref_pairs")
        valid_pairs_aug = read_tsv(Path(tmpdir) / "valid_pairs")

        total_aug = len(sup_pairs_aug) + len(ref_pairs_aug) + len(valid_pairs_aug)

        print(f"\n=== Augmented-only-train mode (augmented_only_train=True) ===")
        print(f"Training (sup): {len(sup_pairs_aug)} pairs")
        print(f"Test (ref): {len(ref_pairs_aug)} pairs")
        print(f"Valid: {len(valid_pairs_aug)} pairs")
        print(f"Total: {total_aug} pairs")

        # With 2 original + 2 augmented:
        # Original split: 20% train = 0, 70% test = 1, 10% valid = 1
        # Augmented: all 2 in train
        # Result: train = 0 + 2 = 2, test = 1, valid = 1

        assert len(sup_pairs_aug) >= 2, f"Expected at least 2 training pairs (augmented), got {len(sup_pairs_aug)}"
        print("\n✓ Test passed: Augmented entities are in training set!")


if __name__ == "__main__":
    test_augmented_only_train()
