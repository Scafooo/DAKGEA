# Entity Alignment Models

This section contains documentation for entity alignment models supported by DAKGEA.

## 📋 Contents

### [BERT-INT](bert-int.md)
**BERT-based Interaction Model for Entity Alignment**

Documentation for the BERT-INT alignment model.

**Key Features:**
- BERT-based entity representation
- Interaction-based similarity computation
- Efficient training on large graphs
- Support for cross-lingual alignment

**Model Architecture:**
- Entity encoder (BERT)
- Interaction module (MLP)
- Contrastive learning objective

---

### [HybEA](hybea.md)
**Hybrid Entity Alignment Model**

Documentation for the HybEA alignment model.

**Key Features:**
- Hybrid architecture combining multiple signals
- Graph structure exploitation
- Attribute information integration
- Scalable to large knowledge graphs

---

## 🎯 Model Selection Guide

### Comparison Matrix

| Model | Strengths | Best Use Cases | Computational Cost |
|-------|-----------|----------------|-------------------|
| **BERT-INT** | • Strong semantic understanding<br>• Cross-lingual support<br>• Pretrained embeddings | • Cross-lingual alignment<br>• Text-rich entities<br>• Moderate-size graphs | High (GPU required) |
| **HybEA** | • Multi-signal fusion<br>• Graph structure exploitation<br>• Scalable architecture | • Large-scale graphs<br>• Structure-rich data<br>• Attribute-rich entities | Medium |

### When to Use Which Model

**Use BERT-INT when:**
- You have cross-lingual alignment tasks
- Entities have rich textual descriptions
- You have GPU resources available
- Semantic understanding is critical

**Use HybEA when:**
- You need to scale to very large graphs
- Graph structure is informative
- Computational resources are limited
- You want to combine multiple signals

---

## 🔧 Common Configuration

### Basic Model Configuration

```yaml
experiment:
  model: bert_int  # or rrea, hybea

  # Model-specific parameters
  bert_int:
    max_length: 128
    batch_size: 32
    learning_rate: 2e-5
    epochs: 10

  evaluation:
    metrics:
      - hits@1
      - hits@5
      - hits@10
      - mrr
```

### Training Configuration

```yaml
training:
  device: cuda  # or cpu
  mixed_precision: true
  gradient_accumulation_steps: 1
  warmup_steps: 100
  max_grad_norm: 1.0
```

---

## 📊 Performance Characteristics

### Typical Performance Ranges

| Dataset Size | BERT-INT | HybEA |
|--------------|----------|-------|
| Small (<10K entities) | 85-95% | 80-90% |
| Medium (10K-50K) | 75-85% | 75-85% |
| Large (>50K) | 65-75% | 70-80% |

*Note: Performance varies significantly based on dataset characteristics and training setup.*

### Training Time Estimates

| Model | Small Dataset | Medium Dataset | Large Dataset |
|-------|--------------|----------------|---------------|
| **BERT-INT** | 10-30 min | 1-3 hours | 5-10 hours |
| **HybEA** | 5-15 min | 30 min-1 hour | 2-5 hours |

*Estimates based on single GPU (Tesla V100) training.*

---

## 🚀 Getting Started

### Quick Start with BERT-INT

```bash
# 1. Configure experiment
cat > config/experiments/my_experiment.yaml <<EOF
experiment:
  name: bert_int_test
  dataset:
    name: openea/D_W_15K_V1
  reduction:
    method: random_entities
    ratio: 0.3
  model: bert_int
EOF

# 2. Run experiment
python -m experiments.runner.runner config/experiments/my_experiment.yaml

# 3. Check results
cat results/my_experiment/*/reduction/results.json
```

### Quick Start with HybEA

```bash
# Similar to BERT-INT, just change model: bert_int to model: hybea
```

---

## 📚 Related Documentation

- **Configuration**: See [configuration/models.md](../configuration/models.md) for detailed options
- **Experiments**: See [experiments/metrics.md](../experiments/metrics.md) for evaluation metrics
- **Guides**: See [guides/quality-evaluation.md](../guides/quality-evaluation.md) for running experiments

---

**Last Updated:** 2025-12-15
