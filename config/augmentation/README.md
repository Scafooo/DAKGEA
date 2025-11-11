# PLM Augmentation Configuration

This directory contains configuration files for the PLM (Pretrained Language Model) augmentation method.

## Default Configuration

**The PLM augmenter automatically loads `config/augmentation/plm.yaml` as its default configuration.**

You don't need to specify the config file explicitly - it will be loaded automatically when you instantiate the augmenter:

```python
from src.augmentation.methods.plm.plm_augmenter import PLMAugmenter

# Automatically loads config/augmentation/plm.yaml
augmenter = PLMAugmenter()
```

## Available Configurations

### 1. `plm.yaml` - Full Configuration (DEFAULT)
Complete configuration with all available parameters and their default values.

**Use case**: Production runs, full experimentation with BART fine-tuning.

**Key settings**:
- `max_depth: 2` - Moderate BFS expansion
- `ratio: 0.5` - 50% augmentation
- `bart.epochs: 10` - Full BART fine-tuning
- `bart.enable_finetuning: true` - BART enabled by default

### 2. `plm_minimal.yaml` - Fast Testing
Minimal configuration for rapid development and testing.

**Use case**: Quick tests, debugging, development iterations.

**Key settings**:
- `max_depth: 1` - Shallow expansion
- `ratio: 0.2` - Small augmentation
- `bart.epochs: 3` - Quick fine-tuning
- `bart.max_train_samples: 1000` - Limited training data

### 3. `plm_no_bart.yaml` - Structural Only
Disables BART fine-tuning, uses only structural BFS expansion.

**Use case**: Baseline comparisons, when BART is not needed, faster execution.

**Key settings**:
- `max_depth: 2` - Standard expansion
- `bart.enable_finetuning: false` - No language model

## Configuration Behavior

### Automatic Loading
When you instantiate `PLMAugmenter()` without arguments, it automatically:
1. Looks for `config/augmentation/plm.yaml`
2. Loads it if it exists
3. Falls back to hardcoded defaults if file not found

```python
# Case 1: Auto-load default config
augmenter = PLMAugmenter()
# → Loads config/augmentation/plm.yaml

# Case 2: Override specific values
custom_config = {
    "augmentation": {
        "max_depth": 3,
        "bart": {"epochs": 5}
    }
}
augmenter = PLMAugmenter(custom_config)
# → Merges custom_config with config/augmentation/plm.yaml
#   (custom values take precedence)

# Case 3: Empty config uses defaults
augmenter = PLMAugmenter({})
# → Loads config/augmentation/plm.yaml
```

### Configuration Merging
When you provide a custom config, it is **deep-merged** with the default:
- Custom values override defaults
- Missing values are filled from defaults
- Nested dictionaries are merged recursively

**Example**:
```python
# Default has: bart.epochs=10, bart.batch_size=16
# Custom has:  bart.epochs=5
# Result:      bart.epochs=5, bart.batch_size=16 (merged!)
```

## Configuration Parameters

### Expansion Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | int | 2 | Maximum BFS depth for entity expansion |
| `ratio` | float | 0.5 | Augmentation ratio (0.5 = 50% more entities) |
| `max_pairs` | int/null | null | Absolute maximum pairs (overrides ratio if set) |

### Provenance Tracking

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `add_derived_predicate` | bool | false | Add derivedFrom triples for lineage |
| `derived_predicate` | URI | "http://dakgea.org/augmentation/derivedFrom" | Predicate URI for derivation |

### BART Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bart.enable_finetuning` | bool | true | Enable/disable BART fine-tuning |
| `bart.model_name` | str | "facebook/bart-base" | Base BART model to use |
| `bart.out_dir` | str | "./bart_plm_model" | Directory for fine-tuned model |
| `bart.epochs` | int | 10 | Number of training epochs |
| `bart.batch_size` | int | 16 | Training batch size |
| `bart.force_retrain` | bool | false | Force retraining even if model exists |
| `bart.max_train_samples` | int | 4000 | Max training samples (null = unlimited) |
| `bart.val_split` | float | 0.1 | Validation split ratio |
| `bart.patience` | int | 3 | Early stopping patience (epochs) |
| `bart.max_per_predicate` | int | 5000 | Max examples per predicate |

### BART Interpolation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bart.base_alpha` | float | 0.35 | Base interpolation coefficient |
| `bart.alpha_spread` | float | 0.25 | Alpha variation range |
| `bart.max_len_in` | int | 96 | Max input token length |
| `bart.max_len_out` | int | 48 | Max output token length |

### Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generation.max_new_tokens` | int | 32 | Max tokens to generate |
| `generation.do_sample` | bool | true | Enable sampling (vs greedy) |
| `generation.top_k` | int | 50 | Top-k sampling |
| `generation.top_p` | float | 0.95 | Nucleus sampling threshold |
| `generation.temperature` | float | 1.2 | Sampling temperature |
| `generation.num_beams` | int | 1 | Beam search width (1 = disabled) |
| `generation.repetition_penalty` | float | 2.0 | Penalty for repetitions |
| `generation.no_repeat_ngram_size` | int | 4 | Prevent n-gram repetition |

## Usage Examples

### Basic Usage (Default Config)

```python
from src.augmentation.methods.plm.plm_augmenter import PLMAugmenter
from src.core.dataset import Dataset

# Load dataset
dataset = Dataset.from_path("path/to/dataset")

# Create augmenter (auto-loads config/augmentation/plm.yaml)
augmenter = PLMAugmenter()

# Augment dataset
augmented_dataset = augmenter.augment(dataset)
```

### Override Specific Parameters

```python
# Use default config but change max_depth and disable BART
custom_config = {
    "augmentation": {
        "max_depth": 3,
        "bart": {
            "enable_finetuning": false
        }
    }
}

augmenter = PLMAugmenter(custom_config)
augmented_dataset = augmenter.augment(dataset)
```

### Using Alternative Config Files

```python
import yaml

# Load alternative config file
with open("config/augmentation/plm_minimal.yaml") as f:
    config = yaml.safe_load(f)

augmenter = PLMAugmenter(config)
```

## Performance Considerations

### Memory Usage

- **BART fine-tuning**: Requires GPU with ~8GB VRAM (for bart-base)
- **Batch size**: Reduce if OOM errors occur
- **Max train samples**: Limit to reduce memory and time

### Training Time

Typical fine-tuning times (on V100 GPU):
- `plm_minimal.yaml`: ~5-10 minutes
- `plm.yaml`: ~20-30 minutes
- First run includes model download and tokenization

### Disk Space

- Fine-tuned BART model: ~500MB
- Training logs: ~10-50MB
- Model is cached and reused across runs (unless `force_retrain: true`)

## Troubleshooting

### CUDA Out of Memory
```yaml
bart:
  batch_size: 8              # Reduce batch size
  max_train_samples: 2000    # Limit training data
```

### Slow Fine-tuning
```yaml
bart:
  epochs: 5                  # Reduce epochs
  max_train_samples: 2000    # Limit data
```

### Model Already Exists Warning
```yaml
bart:
  force_retrain: true        # Force retraining
  # OR
  out_dir: "./bart_model_v2" # Use different directory
```

### Config Not Loading
Check that `config/augmentation/plm.yaml` exists and is valid YAML:
```bash
python -c "import yaml; yaml.safe_load(open('config/augmentation/plm.yaml'))"
```
