# DAKGEA Dataset Guide

This guide explains how datasets work in DAKGEA, including supported formats, readers, writers, and how to add custom datasets.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Formats](#dataset-formats)
3. [Directory Structure](#directory-structure)
4. [Readers](#readers)
5. [Writers](#writers)
6. [Working with Datasets](#working-with-datasets)
7. [Adding Custom Datasets](#adding-custom-datasets)

---

## Overview

DAKGEA uses a **reader/writer architecture** to support multiple dataset formats. This design allows:

- **Format independence**: Work with HybEA, RDF, BERT-INT, or custom formats
- **Automatic conversion**: Transform datasets between formats as needed
- **Pipeline flexibility**: Use the same experiment config with different formats

### Key Concepts

- **Reader**: Loads raw data into DAKGEA's internal `Dataset` representation
- **Writer**: Saves `Dataset` objects to disk in specific formats
- **Variant**: Format-specific subdirectory (e.g., `attribute_data` vs `knowformer_data`)

---

## Dataset Formats

DAKGEA supports three primary formats out of the box:

### 1. HybEA Format

Original format from the [HybEA project](https://github.com/fanourakis/HybEA).

**Structure:**
```
data/raw/hybea/BBC_DB/
├── attribute_data/          # Attribute-based representation
│   ├── ent_ids_1            # Source KG entity IDs
│   ├── ent_ids_2            # Target KG entity IDs
│   ├── attr_names1          # Source KG attribute names
│   ├── attr_names2          # Target KG attribute names
│   ├── attr_triples1        # Source KG attribute triples
│   ├── attr_triples2        # Target KG attribute triples
│   ├── triples_1            # Source KG relation triples
│   ├── triples_2            # Target KG relation triples
│   ├── rel_ids_1            # Source KG relation IDs
│   ├── rel_ids_2            # Target KG relation IDs
│   ├── ent_links            # All entity alignments
│   ├── sup_pairs            # Training pairs
│   ├── ref_pairs            # Test pairs
│   └── valid_pairs          # Validation pairs (optional)
└── knowformer_data/         # Alternative KnowFormer representation
    └── ... (similar structure)
```

**Features:**
- Supports both attribute and relation triples
- Multiple variants (attribute_data, knowformer_data)
- Compatible with HybEA model

**Configuration:**
```yaml
dataset:
  name: "hybea/BBC_DB"
  subtype: "attribute_data"  # Optional: specify variant
```

### 2. BERT-INT Format

Native format for [BERT-INT model](https://github.com/kosugi11037/bert-int).

**Structure:**
```
data/raw/bert_int/D_W_15K_V1/
├── ent_ids_1                # Source entity IDs
├── ent_ids_2                # Target entity IDs
├── triples1                 # Source KG triples (relation)
├── triples2                 # Target KG triples (relation)
├── attr_triples1            # Source KG triples (attribute)
├── attr_triples2            # Target KG triples (attribute)
├── rel2id                   # Relation vocabulary
├── ent_links                # All alignments
├── training_ent_links       # Training pairs
└── testing_ent_links        # Test pairs
```

**Features:**
- Optimized for BERT-INT two-phase training
- Separates relation and attribute triples
- Includes pre-tokenized entity descriptions

**Configuration:**
```yaml
dataset:
  name: "bert_int/D_W_15K_V1"
```

### 3. RDF Format

Standard RDF/Turtle format for semantic web applications.

**Structure:**
```
data/raw/rdf/DW_15/
├── source.ttl               # Source KG in Turtle format
├── target.ttl               # Target KG in Turtle format
└── alignments.txt           # Entity alignment pairs
```

**Features:**
- Standard semantic web format
- Easy integration with existing RDF tools
- Supports full RDF triple semantics

**Configuration:**
```yaml
dataset:
  name: "rdf/DW_15"
```

---

## Directory Structure

### Raw Data

Raw datasets are stored under `data/raw/`:

```
data/raw/
├── hybea/
│   ├── BBC_DB/
│   ├── D_W_15K_V1/
│   └── zh_en/
├── bert_int/
│   └── D_W_15K_V1/
└── rdf/
    └── DW_15/
```

### Processed Data

Processed datasets are stored under workspace directories:

```
results/<experiment>/
└── <dataset>/
    └── <reduction_ratio>/
        ├── reduction/
        │   └── artefacts/
        │       ├── bert_int/      # BERT-INT format output
        │       └── hybea/         # HybEA format output
        ├── augmentation/
        │   └── <method>/
        │       └── artefacts/
        │           ├── bert_int/
        │           └── hybea/
        └── evaluation/
            ├── reduced/           # Baseline (no augmentation)
            └── <augmentation>/    # Augmented variant
```

---

## Readers

Readers load raw datasets into DAKGEA's internal representation.

### Available Readers

#### HybEA Reader

**Class:** `HybEAKnowledgeGraphReader`

**Supports:**
- Multiple variants (attribute_data, knowformer_data)
- Automatic variant selection (chooses most complete)
- Attribute and relation triples
- Entity alignment splits

**Usage:**
```python
from src.core.dataset.reader import DatasetReaderFactory

reader = DatasetReaderFactory.create("hybea")
dataset = reader.read(dataset_path)
```

**Configuration:**
```yaml
dataset:
  name: "hybea/BBC_DB"
  reader: "hybea"
  subtype: "attribute_data"  # Force specific variant
```

#### BERT-INT Reader

**Class:** `BertIntKnowledgeGraphReader`

**Supports:**
- Relation and attribute triples
- Entity descriptions
- Training/test splits

**Usage:**
```python
reader = DatasetReaderFactory.create("bert_int")
dataset = reader.read(dataset_path)
```

**Configuration:**
```yaml
dataset:
  name: "bert_int/D_W_15K_V1"
```

#### RDF Reader

**Class:** `RDFKnowledgeGraphReader`

**Supports:**
- Turtle (.ttl) format
- N-Triples format
- RDF/XML format

**Usage:**
```python
reader = DatasetReaderFactory.create("rdf")
dataset = reader.read(dataset_path)
```

---

## Writers

Writers save datasets to disk in specific formats.

### Available Writers

#### BERT-INT Writer

**Class:** `BertIntKnowledgeGraphWriter`

**Writes:**
- Entity IDs with offset support (required for BERT-INT)
- Relation triples (indexed)
- Attribute triples (separate files)
- Relation vocabulary
- Entity alignment splits

**Key Features:**
- **Offset handling**: Correctly offsets entity IDs between source and target KGs
- **Attribute preservation**: Saves attribute triples separately
- **Format compliance**: Output matches reference BERT-INT format exactly

**Configuration:**
```yaml
dataset:
  writers:
    - type: "bert_int"
```

**Output Structure:**
```
ent_ids_1                # Source entities: 0, 1, 2, ...
ent_ids_2                # Target entities: N, N+1, N+2, ...
triples1                 # Relation triples (source)
triples2                 # Relation triples (target)
attr_triples1            # Attribute triples (source)
attr_triples2            # Attribute triples (target)
rel2id                   # Relation vocabulary
training_ent_links       # Training pairs
testing_ent_links        # Test pairs
```

#### HybEA Writer

**Class:** `HybEAKnowledgeGraphWriter`

**Writes:**
- Entity and relation IDs
- Attribute names and triples
- Relation triples
- Alignment pairs (sup_pairs, ref_pairs)

**Configuration:**
```yaml
dataset:
  writers:
    - type: "hybea"
```

#### Multi-Writer Support

Use multiple writers to save in multiple formats:

```yaml
dataset:
  name: "hybea/BBC_DB"
  writers:
    - type: "bert_int"
    - type: "hybea"
    - type: "rdf"
```

---

## Working with Datasets

### Loading a Dataset

**Via Configuration:**
```yaml
dataset:
  name: "hybea/BBC_DB"
```

**Programmatically:**
```python
from src.core.dataset.reader import DatasetReaderFactory
from pathlib import Path

# Create reader
reader = DatasetReaderFactory.create("hybea")

# Read dataset
dataset = reader.read(Path("data/raw/hybea/BBC_DB/attribute_data"))

# Access data
print(f"Source KG: {len(dataset.knowledge_graph_source)} triples")
print(f"Target KG: {len(dataset.knowledge_graph_target)} triples")
print(f"Alignments: {len(dataset.entity_alignment)} pairs")
```

### Saving a Dataset

**Via Configuration:**
```yaml
dataset:
  name: "hybea/BBC_DB"
  writers:
    - type: "bert_int"
```

**Programmatically:**
```python
from src.core.dataset.writer import DatasetWriterFactory

# Create writer
writer = DatasetWriterFactory.create("bert_int")

# Write dataset
output_path = Path("output/my_dataset")
writer.write(dataset, output_path)
```

### Converting Between Formats

Read in one format, write in another:

```python
from src.core.dataset.reader import DatasetReaderFactory
from src.core.dataset.writer import DatasetWriterFactory
from pathlib import Path

# Read HybEA format
hybea_reader = DatasetReaderFactory.create("hybea")
dataset = hybea_reader.read(Path("data/raw/hybea/BBC_DB/attribute_data"))

# Write BERT-INT format
bert_int_writer = DatasetWriterFactory.create("bert_int")
bert_int_writer.write(dataset, Path("output/BBC_DB_bert_int"))
```

---

## Adding Custom Datasets

### Option 1: Use Existing Format

If your dataset matches an existing format (HybEA, BERT-INT, RDF), simply place it in the appropriate directory:

```bash
# For HybEA format
data/raw/hybea/MY_DATASET/attribute_data/
  ├── ent_ids_1
  ├── ent_ids_2
  # ... etc

# For BERT-INT format
data/raw/bert_int/MY_DATASET/
  ├── ent_ids_1
  ├── ent_ids_2
  # ... etc
```

Then reference it in your config:

```yaml
dataset:
  name: "hybea/MY_DATASET"
```

### Option 2: Create Custom Reader

For custom formats, implement a reader:

```python
from src.core.dataset.reader.base import KnowledgeGraphReader
from src.core.dataset import Dataset

class MyCustomReader(KnowledgeGraphReader):
    """Reader for my custom format."""

    def read(self, dataset_path: Path) -> Dataset:
        """Read dataset from custom format."""
        # Load your files
        source_kg = self._load_source_kg(dataset_path)
        target_kg = self._load_target_kg(dataset_path)
        alignments = self._load_alignments(dataset_path)

        # Create Dataset object
        return Dataset(
            name="my_dataset",
            knowledge_graph_source=source_kg,
            knowledge_graph_target=target_kg,
            entity_alignment=alignments,
            # ... other fields
        )
```

Register it:

```python
from src.core.dataset.reader import READER_REGISTRY

READER_REGISTRY.register("my_format")(MyCustomReader)
```

Use it:

```yaml
dataset:
  name: "MY_DATASET"
  reader: "my_format"
```

### Option 3: Convert to Supported Format

Use external tools to convert your dataset to HybEA or BERT-INT format, then use existing readers.

---

## Dataset Statistics

Get dataset statistics:

```python
dataset = reader.read(dataset_path)

print(f"Name: {dataset.name}")
print(f"Source KG: {len(dataset.knowledge_graph_source)} triples")
print(f"Target KG: {len(dataset.knowledge_graph_target)} triples")
print(f"Total alignments: {len(dataset.entity_alignment)}")
print(f"Training pairs: {len(dataset.training_alignment)}")
print(f"Test pairs: {len(dataset.test_alignment)}")

# Count attribute vs relation triples
from rdflib import Literal

source_attrs = sum(1 for _, _, o in dataset.knowledge_graph_source if isinstance(o, Literal))
target_attrs = sum(1 for _, _, o in dataset.knowledge_graph_target if isinstance(o, Literal))

print(f"Source attributes: {source_attrs}")
print(f"Target attributes: {target_attrs}")
```

---

## Troubleshooting

### "Dataset not found"

**Check:**
1. Dataset exists at `data/raw/<reader>/<dataset>/`
2. Using correct reader name in config
3. File permissions are correct

### "Unable to infer reader"

**Solution:** Use explicit reader/dataset format:
```yaml
dataset:
  name: "hybea/BBC_DB"  # Explicit reader
```

### "Missing required files"

**Check:**
- All required files for the format are present
- File names match expected format exactly
- No typos in filenames

**HybEA Required Files:**
- `ent_ids_1`, `ent_ids_2`
- `ent_links` or (`sup_pairs` + `ref_pairs`)
- At least one triple file (`triples_1`, `attr_triples1`, etc.)

**BERT-INT Required Files:**
- `ent_ids_1`, `ent_ids_2`
- `ent_links` or (`training_ent_links` + `testing_ent_links`)
- `triples1`, `triples2` (or `attr_triples1`, `attr_triples2`)

### Attribute triples missing after write

**Cause:** Writer not handling Literals correctly.

**Solution:** Verify writer implementation:
```python
# Good: Check for Literals
if isinstance(obj, Literal):
    attr_triples.append((subj, pred, obj))
else:
    relation_triples.append((subj, pred, obj))
```

---

## Best Practices

1. **Keep raw data immutable**: Never modify files in `data/raw/`
2. **Use consistent naming**: Follow `<source>_<target>_<size>` convention
3. **Document variants**: Add README in dataset directories explaining variants
4. **Version datasets**: Use git-lfs or DVC for large datasets
5. **Validate after write**: Read back written datasets to verify correctness
6. **Test with small samples**: Create small test datasets for CI/CD

---

## See Also

- [Configuration Guide](configuration-guide.md) - Dataset configuration options
- [BERT-INT Guide](bert-int-guide.md) - BERT-INT specific formats
- [Developer Guide](developer-guide.md) - Implementing custom readers/writers
