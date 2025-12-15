# Configuration

This section contains documentation for configuring DAKGEA experiments, models, and augmentation methods.

## 📋 Contents

### [Augmentation Configuration](augmentation.md)
**Configuring Data Augmentation Methods**

Complete reference for augmentation configuration options.

**Key Topics:**
- PLM (Pretrained Language Model) augmentation
- BART configuration and fine-tuning
- Augmentation ratio settings
- Training mode selection (baseline, augmented, synthetic_only)
- Advanced options and parameters

---

### [Model Configuration](models.md)
**Configuring Entity Alignment Models**

Guide for configuring alignment models (BERT-INT, RREA, etc.).

**Key Topics:**
- Model selection and parameters
- Training hyperparameters
- Evaluation settings
- GPU configuration
- Model-specific options

---

### [Experiment Configuration](experiments.md)
**Setting Up Experiment Configurations**

How to structure and organize experiment configurations.

**Key Topics:**
- YAML configuration structure
- Dataset selection
- Reduction configuration
- Augmentation pipeline setup
- Evaluation options
- Directory structure

---

### [Synthetic Comparison Experiments](synthetic-comparison-experiments.md)
**Configuration for Synthetic Data Comparison Studies**

Specific configuration patterns for comparing synthetic vs real data.

**Key Topics:**
- Baseline configurations
- Synthetic-only configurations
- Fair comparison setup
- Batch experiment generation

---

## 🎯 Configuration Quick Reference

### Basic Experiment Structure

```yaml
experiment:
  name: my_experiment
  dataset:
    name: openea/D_W_15K_V1
    writer: bert_int
  reduction:
    method: random_entities
    ratio: 0.5
  augmentation:  # Optional - omit for baseline
    method: plm
    ratio: 1.0
    training_mode: augmented  # baseline | augmented | synthetic_only
  model: bert_int
```

### Training Modes

| Mode | Config | Description |
|------|--------|-------------|
| **Baseline** | No `augmentation` section | Original data only |
| **Augmented** | `training_mode: augmented` | Original + synthetic |
| **Synthetic-only** | `training_mode: synthetic_only` | Synthetic only |

### Configuration Files Location

```
config/
├── augmentation/       # Augmentation method configs
├── models/             # Model-specific configs
└── experiments/        # Experiment suite configs
    ├── massive/        # Large-scale experiments
    │   ├── bert_int_baseline/
    │   ├── bert_int_synthetic_only/
    │   └── rrea_baseline/
    └── synthetic_comparison/
```

---

## 📚 Related Documentation

- **Guides**: See [guides/quality-evaluation.md](../guides/quality-evaluation.md) for running experiments
- **Architecture**: See [architecture/training-mode.md](../architecture/training-mode.md) for training mode design
- **Models**: See [models/](../models/overview.md) for model-specific details

---

**Last Updated:** 2025-12-15
