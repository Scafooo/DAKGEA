# BERT-INT Configuration Guide

## Overview

BERT-INT is a **two-phase entity alignment model** that always executes both phases:
1. **Phase 1 (basic_unit)**: BERT fine-tuning for entity embeddings
2. **Phase 2 (interaction_model)**: Multi-view feature extraction + MLP for alignment

When you specify `model: bert_int`, both phases are automatically executed in cascade.

## Configuration Architecture

### Default Parameters: `config/models/bert_int.yaml`

All default model parameters are centralized in this file:
- Model architecture parameters (encoder, dimensions, etc.)
- Training hyperparameters (epochs, learning rate, batch size)
- Evaluation settings
- Device configuration

This file defines **production defaults** optimized for best performance.

### Experiment Overrides: `config/experiments/exp_*.yaml`

Experiment configuration files should be **minimal** and only specify:
1. **Required fields**: name, dataset, model, ratio, seed
2. **Parameter overrides**: Only parameters you want to change from defaults

## Example Configurations

### Minimal (Production)
```yaml
experiment:
  name: bert_int_production
  dataset: {name: D_W_15K_V1, reader: hybea, subtype: attribute_data}
  reduction_ratio: 0.1
  model: bert_int
  seed: 11037
```

This runs with all defaults from `config/models/bert_int.yaml`:
- basic_unit: 20 epochs, batch_size 256
- interaction_model: 100 epochs, batch_size 256

### With Custom Overrides
```yaml
experiment:
  name: bert_int_custom
  dataset: {name: D_W_15K_V1, reader: hybea, subtype: attribute_data}
  reduction_ratio: 0.1
  model: bert_int
  seed: 11037

  basic_unit:
    epochs: 50              # Override: train longer
    eval_frequency: 5       # Override: evaluate less frequently

  interaction_model:
    epochs: 200             # Override: train longer
    learning_rate: 0.0005   # Override: lower learning rate
```

### Quick Test
```yaml
experiment:
  name: bert_int_test
  dataset: {name: D_W_15K_V1, reader: hybea, subtype: attribute_data}
  reduction_ratio: 0.1
  model: bert_int
  seed: 11037

  basic_unit:
    epochs: 2               # Fast test: 2 epochs only

  interaction_model:
    epochs: 5               # Fast test: 5 epochs only
```

## Available Templates

- **`bert_int_template.yaml`**: Full template with all optional parameters commented
- **`exp_8.yaml`**: Production config with custom eval frequencies
- **`exp_8_test.yaml`**: Fast testing config with minimal epochs
- **`exp_bert_int_full.yaml`**: Alternative production config

## Key Principles

1. **Always two-phase**: BERT-INT always runs both basic_unit and interaction_model
2. **Defaults first**: All parameters have sensible defaults in `config/models/bert_int.yaml`
3. **Override only when needed**: Keep experiment configs minimal
4. **No `enabled` flag**: Selecting `model: bert_int` automatically enables both phases

## How It Works

When you run an experiment with `model: bert_int`:

1. System loads defaults from `config/models/bert_int.yaml`
2. System merges your experiment overrides on top
3. Phase 1 (basic_unit) executes with merged config
4. Phase 2 (interaction_model) executes automatically after Phase 1
5. Final results in `bert_int.json` come from interaction_model (Phase 2)

## Parameter Categories

### Basic Unit Parameters
- **Architecture**: `encoder_name`, `max_seq_length`, `dropout`, `output_dim`
- **Training**: `epochs`, `batch_size`, `learning_rate`, `margin`
- **Evaluation**: `eval_frequency`, `eval_batch_size`, `eval_top_k`
- **Device**: `device` (e.g., `cuda:0`, `cpu`)

### Interaction Model Parameters
- **Feature Extraction**: `kernel_num`, `entity_neigh_max_num`, `candidate_topk`
- **Model Architecture**: `mlp_hidden_dim`
- **Training**: `epochs`, `batch_size`, `learning_rate`, `margin`, `neg_num`
- **Evaluation**: `eval_every`
- **Device**: `device`

## See Also

- Full defaults: `config/models/bert_int.yaml`
- Template: `bert_int_template.yaml`
- Architecture documentation: `BERT_INT_PIPELINE.md` (in project root)
