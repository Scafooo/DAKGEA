# DAKGEA Configuration Guide

This guide explains all configuration options available in DAKGEA experiment files, with examples showing both legacy and modern syntax.

---

## Table of Contents

1. [Configuration File Structure](#configuration-file-structure)
2. [Dataset Configuration](#dataset-configuration)
3. [Augmentation Configuration](#augmentation-configuration)
4. [Model Configuration](#model-configuration)
5. [Advanced Options](#advanced-options)
6. [Complete Examples](#complete-examples)

---

## Configuration File Structure

Experiment configurations are YAML files located in `config/experiments/`. Each file defines a complete experimental pipeline.

### Basic Structure

```yaml
experiment:
  name: "experiment_name"
  dataset:
    name: "dataset_name"
    # ... dataset options
  augmentation:
    method: "augmentation_method"
    # ... augmentation options
  model: "model_name"
  seed: 42
  clear: true
```

**Fields:**
- `name`: Unique experiment identifier (used for output directory)
- `dataset`: Dataset specification (see [Dataset Configuration](#dataset-configuration))
- `augmentation`: Augmentation method and parameters (see [Augmentation Configuration](#augmentation-configuration))
- `model`: Alignment model to evaluate (e.g., `bert_int`, `hybea`)
- `seed`: Random seed for reproducibility
- `clear`: Whether to clean up intermediate files after completion

---

## Dataset Configuration

DAKGEA supports multiple dataset formats through a flexible reader/writer system.

### Format 1: Simple Dataset Name

The simplest configuration uses just the dataset name. The system will automatically search for it under all reader directories.

```yaml
dataset:
  name: "BBC_DB"
```

**Behavior:** System searches under `data/raw/*/BBC_DB/` and auto-detects the reader.

### Format 2: Explicit Reader/Dataset Path

Use the `reader/dataset` format to explicitly specify which reader to use:

```yaml
dataset:
  name: "hybea/BBC_DB"
```

**Behavior:** Uses the `hybea` reader for dataset `BBC_DB` located at `data/raw/hybea/BBC_DB/`.

**Examples:**
```yaml
# HybEA format dataset
dataset:
  name: "hybea/BBC_DB"

# RDF format dataset
dataset:
  name: "rdf/DW_15"

# BERT-INT format dataset
dataset:
  name: "bert_int/D_W_15K_V1"
```

### Format 3: Direct Path Mode

For pre-processed datasets or external data, use a direct path:

```yaml
dataset:
  path: "/absolute/path/to/dataset"
```

**Behavior:** Skips reduction and augmentation stages, reads dataset directly from the specified path. Useful for:
- Testing with reference implementations
- Using pre-reduced datasets
- External dataset sources

**Example:**
```yaml
experiment:
  name: "direct_test"
  dataset:
    path: "/home/user/data/preprocessed/BBC_DB_reduced"
  model: bert_int
  seed: 42
```

### Format 4: Full Configuration with Reader Override

You can override auto-detection by explicitly specifying the reader:

```yaml
dataset:
  name: "BBC_DB"
  reader: "hybea"
  subtype: "attribute_data"  # Optional: reader-specific variant
```

**Fields:**
- `reader`: Force a specific reader (`hybea`, `rdf`, `bert_int`)
- `subtype`: Reader-specific variant (e.g., `attribute_data` vs `knowformer_data` for HybEA)

---

## Augmentation Configuration

DAKGEA supports two configuration styles for augmentation.

### Modern Syntax (Recommended)

```yaml
augmentation:
  method: "stub"
  reduction: 0.1
```

**Fields:**
- `method`: Augmentation method name (`stub`, `plm_augmentation`, etc.)
- `reduction`: Reduction ratio (0.0-1.0) - fraction of training data to keep

**Example - No Augmentation, Just Reduction:**
```yaml
augmentation:
  method: "stub"  # Stub method performs no augmentation
  reduction: 0.1  # Keep 10% of training pairs
```

**Example - PLM Augmentation:**
```yaml
augmentation:
  method: "plm_augmentation"
  reduction: 0.2
  parameters:
    model_name: "bert-base-multilingual-cased"
    batch_size: 32
```

### Legacy Syntax (Still Supported)

The old format separates reduction ratios and augmentation methods:

```yaml
reduction_ratio: 0.1
augmentation_method: "stub"
```

Or with multiple ratios:

```yaml
reduction_ratios: [0.1, 0.2, 0.5]
augmentation_methods: ["stub", "plm_augmentation"]
```

**Note:** The modern syntax is preferred for clarity, but both formats are fully supported.

---

## Model Configuration

### Basic Model Selection

```yaml
model: "bert_int"
```

Supported models:
- `bert_int`: Two-phase BERT-INT alignment model
- `hybea`: HybEA alignment model
- (other models as registered in the system)

### Model with Parameters Override

Override model-specific parameters:

```yaml
model: "bert_int"
parameters:
  models:
    bert_int:
      basic_unit:
        epochs: 10
        batch_size: 256
      interaction_model:
        epochs: 50
        learning_rate: 5e-4
```

### Multi-Model Evaluation

Run multiple models in sequence:

```yaml
models_to_run: ["bert_int", "hybea"]
```

---

## Advanced Options

### Writer Configuration

Control which formats to write and when:

```yaml
dataset:
  name: "hybea/BBC_DB"
  writers:
    - type: "bert_int"
    - type: "hybea"
```

Or with more control:

```yaml
dataset:
  name: "BBC_DB"
  writer: "bert_int"  # Single writer
```

### Skip Training (Testing Mode)

For testing or re-evaluation without retraining:

```yaml
experiment:
  name: "test_evaluation"
  dataset:
    path: "/path/to/checkpoint"
  model: bert_int
  skip_training: true
```

### Overwrite Control

Control caching behavior:

```yaml
experiment:
  name: "my_experiment"
  # ... other config
  overwrite_existing: true  # Always recompute, ignore cache
```

Or use CLI flag: `./run.sh config.yaml --overwrite-existing`

### Resume Mode

Reuse all cached artifacts:

```yaml
experiment:
  name: "my_experiment"
  # ... other config
  resume: true
```

Or use CLI flag: `./run.sh config.yaml --resume`

---

## Complete Examples

### Example 1: Simple Reduction Experiment

Test BERT-INT with 10% of training data, no augmentation:

```yaml
experiment:
  name: "bert_int_reduction_10"
  dataset:
    name: "hybea/BBC_DB"
    writer: bert_int          # REQUIRED for BERT-INT model
  augmentation:
    method: "stub"
    reduction: 0.1
  model: bert_int
  seed: 11037
  clear: true
```

**Important:** BERT-INT requires `writer: bert_int` to convert data to the correct format.

### Example 2: Multiple Reduction Ratios (Legacy Format)

Compare different reduction levels:

```yaml
name: "multi_ratio_experiment"
datasets:
  - name: "BBC_DB"
    reader: "hybea"
reduction_ratios: [0.1, 0.2, 0.5, 1.0]
augmentation_method: "stub"
models_to_run: ["bert_int"]
parameters:
  experiment:
    seed: 42
```

### Example 3: PLM Augmentation

Use PLM-based augmentation with custom parameters:

```yaml
experiment:
  name: "plm_augmentation_test"
  dataset:
    name: "hybea/D_W_15K_V1"
  augmentation:
    method: "plm_augmentation"
    reduction: 0.2
  model: bert_int
  seed: 42
  parameters:
    augmentation:
      plm_augmentation:
        model_name: "bert-base-multilingual-cased"
        augmentation_factor: 2.0
```

### Example 4: Direct Path with Pre-Processed Data

Use reference implementation's preprocessed data:

```yaml
experiment:
  name: "reference_comparison"
  dataset:
    path: "/home/user/DAKGEA/Bert_int_reference/D_W_15K_V1/10/attribute_data"
  model: bert_int
  seed: 11037
  clear: true
```

### Example 5: Multi-Model Comparison

Compare BERT-INT and HybEA on the same reduced dataset:

```yaml
experiment:
  name: "model_comparison"
  dataset:
    name: "hybea/BBC_DB"
  augmentation:
    method: "stub"
    reduction: 0.3
  models_to_run: ["bert_int", "hybea"]
  seed: 42
  parameters:
    models:
      bert_int:
        basic_unit:
          epochs: 20
      hybea:
        max_epochs: 100
```

### Example 6: Full Configuration with All Options

Complete example showing all major features:

```yaml
experiment:
  name: "comprehensive_experiment"

  # Dataset configuration
  dataset:
    name: "hybea/BBC_DB"
    reader: "hybea"      # Explicit reader (optional when using reader/dataset format)
    subtype: "attribute_data"  # Use attribute_data variant
    writers:
      - type: "bert_int"  # Write in BERT-INT format
      - type: "hybea"     # Also keep HybEA format

  # Augmentation configuration
  augmentation:
    method: "plm_augmentation"
    reduction: 0.2

  # Model configuration
  model: bert_int

  # Experiment settings
  seed: 11037
  clear: true
  overwrite_existing: false

  # Model-specific parameters
  parameters:
    experiment:
      seed: 11037

    models:
      bert_int:
        basic_unit:
          epochs: 20
          batch_size: 256
          learning_rate: 5.0e-5
        interaction_model:
          epochs: 100
          batch_size: 128
          learning_rate: 5e-4
          candidate_topk: 50

    augmentation:
      plm_augmentation:
        model_name: "bert-base-multilingual-cased"
        augmentation_factor: 2.0
        batch_size: 32
```

---

## Configuration Validation

The system validates configurations at runtime:

- **Missing required fields**: Error with helpful message
- **Invalid dataset paths**: Checks existence before starting
- **Unknown readers/models**: Lists available options
- **Parameter type mismatches**: Validates numeric ranges

**Tip:** Use `--dry-run` flag to validate configuration without running the experiment:

```bash
./run.sh config/experiments/my_config.yaml --dry-run
```

---

## Best Practices

1. **Use explicit reader/dataset format** (`hybea/BBC_DB`) for clarity
2. **Use modern augmentation syntax** for new experiments
3. **Set consistent seeds** for reproducibility
4. **Use descriptive names** for experiments (include date, parameters)
5. **Enable `clear: true`** to save disk space
6. **Document custom parameters** in comments
7. **Version control configs** alongside code changes

---

## Troubleshooting

### "Unable to infer reader for dataset"

**Cause:** Dataset not found or ambiguous path.

**Solution:** Use explicit `reader/dataset` format:
```yaml
dataset:
  name: "hybea/BBC_DB"  # Instead of just "BBC_DB"
```

### "Direct path mode requires 'path' field"

**Cause:** System entered direct path mode but no path provided.

**Solution:** Either:
- Add `path:` field for direct mode
- Or add `reduction:` to use standard mode:
  ```yaml
  augmentation:
    method: "stub"
    reduction: 0.1
  ```

### Parameters not taking effect

**Cause:** Wrong nesting level in YAML.

**Solution:** Ensure parameters are under `parameters.models.<model_name>`:
```yaml
parameters:
  models:
    bert_int:
      basic_unit:
        epochs: 10  # ✓ Correct nesting
```

---

## See Also

- [User Guide](user-guide.md) - Installation and basic usage
- [Dataset Guide](dataset-guide.md) - Dataset formats and readers
- [BERT-INT Guide](bert-int-guide.md) - BERT-INT specific documentation
- [Developer Guide](developer-guide.md) - Extending the framework
