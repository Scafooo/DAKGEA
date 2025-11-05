# BERT-INT Model Guide

Complete guide to using the BERT-INT alignment model in DAKGEA, including architecture, configuration, and troubleshooting.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [Dataset Requirements](#dataset-requirements)
5. [Training Pipeline](#training-pipeline)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)
8. [Reference Comparison](#reference-comparison)

---

## Overview

BERT-INT is a **two-phase** entity alignment model that combines:
1. **Basic Unit** (Phase 1): BERT-based entity encoder
2. **Interaction Model** (Phase 2): MLP classifier on multi-view features

**Key Features:**
- Multilingual BERT for entity encoding
- Multi-view interaction features (neighbor, attribute, description)
- Two-stage training for better performance
- Supports attribute-rich knowledge graphs

**Reference:** [BERT-INT GitHub](https://github.com/kosugi11037/bert-int)

---

## Architecture

### Two-Phase Design

```
┌─────────────────────────────────────────────────────────────┐
│                        BERT-INT Model                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Phase 1: Basic Unit (Entity Encoder)                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Input: Entity descriptions (text)                   │    │
│  │         ↓                                            │    │
│  │  BERT Tokenizer                                      │    │
│  │         ↓                                            │    │
│  │  BERT Encoder (bert-base-multilingual-cased)        │    │
│  │         ↓                                            │    │
│  │  Linear Projection (768 → 300)                      │    │
│  │         ↓                                            │    │
│  │  Entity Embeddings                                   │    │
│  └─────────────────────────────────────────────────────┘    │
│         ↓                                                     │
│  Phase 2: Interaction Model (Alignment Classifier)           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  1. Candidate Generation (top-K nearest)            │    │
│  │         ↓                                            │    │
│  │  2. Feature Extraction                               │    │
│  │     - Neighbor View (42 features)                    │    │
│  │     - Attribute View (42 features)                   │    │
│  │     - Description View (1 feature)                   │    │
│  │         ↓                                            │    │
│  │  3. MLP Classifier (85 → 11 → 1)                    │    │
│  │         ↓                                            │    │
│  │  Alignment Scores                                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Phase 1: Basic Unit

**Purpose:** Learn entity representations from textual descriptions.

**Architecture:**
```python
Input: Entity text (name + attributes)
    ↓
Tokenization (max_length=128)
    ↓
BERT Encoder (bert-base-multilingual-cased)
    ↓
[CLS] token output (768-dim)
    ↓
Linear projection (768 → 300)
    ↓
Entity embedding (300-dim)
```

**Training:**
- Loss: Margin-based contrastive loss
- Optimizer: AdamW
- Learning rate: 5e-5
- Batch size: 256
- Epochs: 20

**Output:**
- Entity embeddings for all entities (shape: `[num_entities, 300]`)
- Trained BERT encoder checkpoint

### Phase 2: Interaction Model

**Purpose:** Refine alignments using multi-view graph features.

**Feature Extraction:**

1. **Neighbor View (42 features)**
   - Kernel functions over neighbor embeddings
   - Captures structural similarity
   - Kernel count: 21 × 2 (source + target)

2. **Attribute View (42 features)**
   - Kernel functions over attribute values
   - Captures attribute similarity
   - Kernel count: 21 × 2

3. **Description View (1 feature)**
   - Cosine similarity of entity embeddings
   - Captures textual similarity

**MLP Architecture:**
```python
Input: 85 features (42 + 42 + 1)
    ↓
Linear(85 → 11) + ReLU
    ↓
Linear(11 → 1) + Tanh
    ↓
Alignment score ∈ [-1, 1]
```

**Training:**
- Loss: Margin ranking loss
- Optimizer: Adam
- Learning rate: 5e-4
- Batch size: 128
- Epochs: 100
- Negative samples: 5 per positive

---

## Configuration

### Basic Configuration

Minimal config for BERT-INT:

```yaml
experiment:
  name: "bert_int_basic"
  dataset:
    name: "hybea/BBC_DB"
    writer: "bert_int"  # Required: BERT-INT format
  augmentation:
    method: "stub"
    reduction: 0.1
  model: bert_int
  seed: 11037
```

### Full Configuration

Complete config with all parameters:

```yaml
experiment:
  name: "bert_int_full"

  dataset:
    name: "hybea/BBC_DB"
    writers:
      - type: "bert_int"  # Must write in BERT-INT format

  augmentation:
    method: "stub"
    reduction: 0.2

  model: bert_int
  seed: 11037

  parameters:
    models:
      bert_int:
        # Global settings
        device: "cuda:0"

        # Phase 1: Basic Unit
        basic_unit:
          # Model architecture
          encoder_name: "bert-base-multilingual-cased"
          max_seq_length: 128
          output_dim: 300
          dropout: 0.1

          # Training parameters
          epochs: 20
          batch_size: 256
          learning_rate: 5.0e-5
          weight_decay: 0.0
          warmup_steps: 100
          max_grad_norm: 1.0

          # Loss parameters
          margin: 3.0
          negatives_per_positive: 2

          # Evaluation
          eval_frequency: 1
          eval_batch_size: 128
          eval_top_k: 1000

        # Phase 2: Interaction Model
        interaction_model:
          # Feature extraction
          kernel_num: 21
          entity_neigh_max_num: 50
          entity_attvalue_max_num: 50
          candidate_topk: 50

          # Model architecture
          mlp_hidden_dim: 11

          # Training parameters
          epochs: 100
          batch_size: 128
          learning_rate: 5e-4
          margin: 1.0
          neg_num: 5
          eval_every: 2

          # Device
          device: "cuda:0"
```

### Configuration Reference

#### Basic Unit Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoder_name` | `bert-base-multilingual-cased` | Pretrained BERT model |
| `max_seq_length` | 128 | Maximum token length |
| `output_dim` | 300 | Entity embedding dimension |
| `dropout` | 0.1 | Dropout rate |
| `epochs` | 20 | Training epochs |
| `batch_size` | 256 | Batch size |
| `learning_rate` | 5e-5 | Learning rate |
| `margin` | 3.0 | Contrastive loss margin |
| `negatives_per_positive` | 2 | Negative sampling ratio |

#### Interaction Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel_num` | 21 | Number of kernel functions |
| `entity_neigh_max_num` | 50 | Max neighbors per entity |
| `entity_attvalue_max_num` | 50 | Max attributes per entity |
| `candidate_topk` | 50 | Candidates for reranking |
| `mlp_hidden_dim` | 11 | Hidden layer size |
| `epochs` | 100 | Training epochs |
| `batch_size` | 128 | Batch size |
| `learning_rate` | 5e-4 | Learning rate |
| `margin` | 1.0 | Ranking loss margin |
| `neg_num` | 5 | Negative samples per positive |

---

## Dataset Requirements

### Required Format

BERT-INT requires datasets in **BERT-INT format**. Use the `bert_int` writer:

```yaml
dataset:
  name: "hybea/BBC_DB"
  writers:
    - type: "bert_int"
```

### File Structure

BERT-INT expects these files:

```
<dataset_path>/
├── ent_ids_1              # Source entity IDs (0, 1, 2, ...)
├── ent_ids_2              # Target entity IDs (N, N+1, N+2, ...)
├── triples1               # Source relation triples (id1 rel_id id2)
├── triples2               # Target relation triples
├── attr_triples1          # Source attribute triples (id pred value)
├── attr_triples2          # Target attribute triples
├── rel2id                 # Relation vocabulary
├── training_ent_links     # Training alignment pairs
└── testing_ent_links      # Test alignment pairs
```

### Entity ID Offset

**Important:** Target entity IDs must be offset by source entity count.

**Example:**
```
Source entities: 0, 1, 2, 3, 4       (5 entities)
Target entities: 5, 6, 7, 8, 9       (5 entities, offset by 5)
```

**Alignment pairs:**
```
training_ent_links:
0    5    # Source entity 0 aligns with target entity 5
1    7    # Source entity 1 aligns with target entity 7
```

The BERT-INT writer handles this automatically.

### Attribute Triples

BERT-INT supports attribute triples (entity-predicate-literal):

```
attr_triples1:
0    name    "London"
0    population    "8900000"
1    name    "Paris"
```

**Format:** `entity_id \t predicate \t literal_value`

---

## Training Pipeline

### End-to-End Example

```bash
# 1. Create configuration
cat > config/experiments/my_bert_int.yaml << EOF
experiment:
  name: "my_bert_int_experiment"
  dataset:
    name: "hybea/BBC_DB"
    writer: "bert_int"
  augmentation:
    method: "stub"
    reduction: 0.2
  model: bert_int
  seed: 42
EOF

# 2. Run experiment
./run.sh config/experiments/my_bert_int.yaml

# 3. Check results
cat results/my_bert_int_experiment/BBC_DB/0.2/evaluation/reduced/bert_int.json
```

### Training Stages

The pipeline executes these stages automatically:

1. **Dataset Loading**
   ```
   Read raw dataset → DAKGEA Dataset object
   ```

2. **Reduction**
   ```
   Apply reduction (e.g., 0.2 = 20% of training pairs)
   ```

3. **Format Conversion**
   ```
   Write to BERT-INT format (with entity ID offset)
   ```

4. **Phase 1: Basic Unit Training**
   ```
   Tokenize entities → Train BERT encoder → Generate embeddings
   ```

5. **Phase 2: Interaction Model Training**
   ```
   Generate candidates → Extract features → Train MLP → Evaluate
   ```

6. **Results**
   ```
   Save metrics (Hits@K, MRR) to JSON
   ```

### Output Files

After training, find:

**Checkpoints:**
```
results/<experiment>/evaluation/bert_int/<variant>/
├── run_1.pth              # Basic Unit checkpoint (epoch 1)
├── run_2.pth              # Basic Unit checkpoint (epoch 2)
├── ...
├── run_20.pth             # Basic Unit final checkpoint
└── interaction_model.pt   # Interaction Model checkpoint
```

**Metrics:**
```
results/<experiment>/<dataset>/<ratio>/evaluation/<variant>/bert_int.json
{
  "model": "bert_int",
  "phases": {
    "basic_unit": {
      "hits@1": 0.3456,
      "hits@10": 0.7823,
      "mrr": 0.5234
    },
    "interaction_model": {
      "hits@1": 0.4521,
      "hits@10": 0.8345,
      "mrr": 0.6123
    }
  },
  "hits@1": 0.4521,    // Final metrics (from Phase 2)
  "hits@10": 0.8345,
  "mrr": 0.6123
}
```

### Resuming Training

Skip retraining if checkpoints exist:

```yaml
experiment:
  name: "bert_int_resume"
  dataset:
    path: "/path/to/checkpoint/dir"  # Must contain .pth files
  model: bert_int
  skip_training: true  # Load existing checkpoints
```

---

## Performance Tuning

### GPU Memory Optimization

If running out of memory:

```yaml
parameters:
  models:
    bert_int:
      basic_unit:
        batch_size: 128        # Reduce from 256
        eval_batch_size: 64    # Reduce from 128

      interaction_model:
        batch_size: 64         # Reduce from 128
```

### CPU-Only Mode

For machines without GPU:

```yaml
parameters:
  models:
    bert_int:
      device: "cpu"
      basic_unit:
        device: "cpu"
        batch_size: 32         # Much smaller batches on CPU
      interaction_model:
        device: "cpu"
        batch_size: 32
```

### Speed vs. Accuracy Tradeoffs

**Faster training (lower accuracy):**
```yaml
parameters:
  models:
    bert_int:
      basic_unit:
        epochs: 5              # Reduce from 20
        batch_size: 512        # Larger batches
      interaction_model:
        epochs: 20             # Reduce from 100
        candidate_topk: 20     # Reduce from 50
```

**Better accuracy (slower training):**
```yaml
parameters:
  models:
    bert_int:
      basic_unit:
        epochs: 30             # Increase from 20
        learning_rate: 3e-5    # Lower learning rate
      interaction_model:
        epochs: 200            # Increase from 100
        candidate_topk: 100    # More candidates
        entity_neigh_max_num: 100  # More neighbors
```

---

## Troubleshooting

### Common Issues

#### Issue 1: "Unable to infer reader for dataset"

**Cause:** Dataset path not found or ambiguous.

**Solution:** Use explicit writer in config:
```yaml
dataset:
  name: "hybea/BBC_DB"
  writer: "bert_int"
```

#### Issue 2: Metrics showing as percentages (3938.22%)

**Cause:** Old version with percentage bug (fixed).

**Solution:** Update to latest version. Metrics should be fractions (0-1):
```json
{
  "hits@1": 0.3938,  // ✓ Correct: fraction
  "hits@10": 0.6332
}
```

#### Issue 3: Attribute triples missing (0 attributes loaded)

**Cause:** Writer not saving attribute triples correctly (fixed).

**Solution:**
1. Update to latest version
2. Verify writer saved `attr_triples1` and `attr_triples2`
3. Check file contains Literal values:
   ```
   0    name    "London"
   0    population    "8900000"
   ```

#### Issue 4: Entity ID offset wrong

**Symptom:** Alignment pairs reference non-existent entities.

**Cause:** Target entities not offset correctly.

**Solution:** The BERT-INT writer handles this automatically. If using custom writer, ensure:
```python
# Source entities: 0, 1, 2, ..., N-1
source_offset = 0

# Target entities: N, N+1, N+2, ..., N+M-1
target_offset = len(source_entities)
```

#### Issue 5: CUDA out of memory

**Solutions:**
1. Reduce batch size (see [GPU Memory Optimization](#gpu-memory-optimization))
2. Use CPU mode
3. Use gradient accumulation:
   ```yaml
   basic_unit:
     batch_size: 64
     gradient_accumulation: 4  # Effective batch size = 256
   ```

#### Issue 6: Training too slow

**Solutions:**
1. Use GPU (up to 50x faster than CPU)
2. Reduce sequence length:
   ```yaml
   basic_unit:
     max_seq_length: 64  # Reduce from 128
   ```
3. Use smaller BERT model:
   ```yaml
   basic_unit:
     encoder_name: "bert-base-uncased"  # Smaller than multilingual
   ```

---

## Reference Comparison

### Comparing with Reference Implementation

To verify implementation correctness, compare with the reference:

**1. Prepare identical dataset:**
```bash
# Use same random seed
# Use same reduction ratio
# Use same train/test split
```

**2. Run both implementations:**
```bash
# Reference
cd Bert_int_reference/basic_bert_unit
python main.py

# DAKGEA
./run.sh config/experiments/my_config.yaml
```

**3. Compare results:**

Expect ~1-3% variance due to:
- PyTorch version differences
- CUDA randomness
- Hardware variations

**Exact match** indicates implementation is correct.

### Known Differences from Reference

Our implementation has these improvements:

1. **Attribute triple support**: Reference skips attributes, ours preserves them
2. **Modular design**: Separate reader/writer components
3. **Multi-format support**: Works with HybEA, RDF, BERT-INT formats
4. **Better logging**: Detailed phase-by-phase metrics
5. **Checkpoint management**: Automatic checkpoint saving/loading

### Validation Results

Tested on D_W_15K_V1 (fold=1, reduction=0.1):

| Metric | Reference | DAKGEA | Difference |
|--------|-----------|--------|------------|
| Phase 1 Hits@1 | 0.3436 | 0.3436 | 0.00% |
| Phase 1 Hits@10 | 0.7456 | 0.7456 | 0.00% |
| Phase 2 Hits@1 | 0.4015 | 0.3997 | -0.45% |
| Phase 2 Hits@10 | 0.6305 | 0.6289 | -0.25% |

✅ **Validation successful**: Differences < 1% (within normal variance)

---

## Advanced Topics

### Custom Entity Descriptions

Override default entity representation:

```python
from src.alignment_models.methods.bert_int import load_basic_unit_data

# Custom description function
def custom_description(entity_uri, attributes):
    """Generate entity description from URI and attributes."""
    name = entity_uri.split('/')[-1]
    attr_texts = [f"{k}: {v}" for k, v in attributes.items()]
    return f"{name} | {' | '.join(attr_texts)}"

# Use in configuration
# (requires extending the code)
```

### Multi-GPU Training

For large datasets, use multiple GPUs:

```yaml
parameters:
  models:
    bert_int:
      basic_unit:
        device: "cuda:0,1,2,3"  # Use 4 GPUs
        batch_size: 1024        # Larger total batch
```

### Checkpoint Inspection

Load and inspect checkpoints:

```python
import torch

# Load Basic Unit checkpoint
checkpoint = torch.load("results/.../run_20.pth")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Loss: {checkpoint['loss']}")

# Load model
from src.alignment_models.methods.bert_int.basic_unit import BasicBertUnit

config = {...}  # Your config
model = BasicBertUnit(config)
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## See Also

- [Configuration Guide](configuration-guide.md) - Full config options
- [Dataset Guide](dataset-guide.md) - Dataset formats and readers
- [User Guide](user-guide.md) - Installation and basic usage
- [Developer Guide](developer-guide.md) - Extending BERT-INT

---

## References

- [BERT-INT Paper](https://arxiv.org/abs/2104.08095) - Original paper
- [BERT-INT GitHub](https://github.com/kosugi11037/bert-int) - Reference implementation
- [BERT Documentation](https://huggingface.co/docs/transformers/model_doc/bert) - Hugging Face BERT docs
