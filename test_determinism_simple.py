#!/usr/bin/env python3
"""Test determinism in reduction by running it twice with same seed."""
from src.core.dataset.readers.reader_openea import ReaderOpenEA
from src.reduction.methods.random_entities.reducer_random_entities import RandomEntitiesReducer
import copy

print("=== Test 1: Caricare dataset due volte ===")
reader = ReaderOpenEA()
dataset1 = reader.read("openea/BBC_DB")
print(f"Dataset 1: {len(dataset1.aligned_entities)} aligned pairs")

dataset2 = reader.read("openea/BBC_DB")
print(f"Dataset 2: {len(dataset2.aligned_entities)} aligned pairs")

if dataset1.aligned_entities == dataset2.aligned_entities:
    print("✓ Dataset loading is deterministic")
else:
    print("✗ Dataset loading is NOT deterministic!")
    exit(1)

print("\n=== Test 2: Reduction con stesso seed (con filter) ===")
config = {
    "method": "random_entities",
    "ratio": 0.1,
    "filter_alignment": True,
}

# Clone datasets for independent reduction
ds1 = copy.deepcopy(dataset1)
ds2 = copy.deepcopy(dataset1)  # Start from same data!

reducer1 = RandomEntitiesReducer(config, seed=11037)
result1 = reducer1.reduce(ds1)

reducer2 = RandomEntitiesReducer(config, seed=11037)
result2 = reducer2.reduce(ds2)

print(f"Result 1: {len(result1.aligned_entities)} aligned pairs")
print(f"Result 2: {len(result2.aligned_entities)} aligned pairs")

if result1.aligned_entities == result2.aligned_entities:
    print("✓ Reduction WITH filter is deterministic")
else:
    print("✗ Reduction WITH filter is NOT deterministic!")
    diff = result1.aligned_entities.symmetric_difference(result2.aligned_entities)
    print(f"  Difference: {len(diff)} pairs")
    print(f"  In result1 but not result2: {len(result1.aligned_entities - result2.aligned_entities)}")
    print(f"  In result2 but not result1: {len(result2.aligned_entities - result1.aligned_entities)}")
    exit(1)

print("\n=== Test 3: Reduction senza filter ===")
config_no_filter = {
    "method": "random_entities",
    "ratio": 0.1,
    "filter_alignment": False,
}

ds3 = copy.deepcopy(dataset1)
ds4 = copy.deepcopy(dataset1)

reducer3 = RandomEntitiesReducer(config_no_filter, seed=11037)
result3 = reducer3.reduce(ds3)

reducer4 = RandomEntitiesReducer(config_no_filter, seed=11037)
result4 = reducer4.reduce(ds4)

print(f"Result 3 (no filter): {len(result3.aligned_entities)} aligned pairs")
print(f"Result 4 (no filter): {len(result4.aligned_entities)} aligned pairs")

if result3.aligned_entities == result4.aligned_entities:
    print("✓ Reduction WITHOUT filter is deterministic")
else:
    print("✗ Reduction WITHOUT filter is NOT deterministic!")
    exit(1)

print("\n=== SUCCESS: Tutti i test passati! ===")
