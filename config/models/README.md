# Alignment Model Configuration

This directory bundles the YAML files that drive the alignment models registered in `src/alignment_models`.  
Each model class looks for a specific file under `config/models` and automatically loads it (together with optional
local overrides) before training or evaluation.

## Auto-loading behaviour

| Model name | Loader | Default file(s) | Notes |
|------------|--------|-----------------|-------|
| `bert_int` | `load_bert_int_config` (`src/alignment_models/methods/bert_int/config.py`) | `config/models/bert_int.yaml` (mergeable with runtime overrides) | Paths are resolved relative to `PROJECT_ROOT` and missing keys fall back to hardcoded defaults. |
| `openea` | `OpenEAAlignment` (`src/alignment_models/methods/openea`) | `config/models/openea.yaml` | Contains both attribute and structural sub-model settings plus dataset-specific dimensions. |
| `hybea` | `HybEA` (`src/alignment_models/methods/hybea/model.py`) | `config/models/hybea.yaml` (+ optional `hybea.local.yaml`, `hybea.stage.yaml`) | The base file is not versioned by default. Create it locally to tune HybEA. Files are merged in the order `hybea.yaml` → `hybea.local.yaml` → `hybea.stage.yaml` → stage overrides. |
| `stub` | `StubAlignment` (smoke tests) | `config/models/stub.yaml` | Placeholder file, kept empty because the stub model returns constant metrics. |

When you instantiate a model (e.g. `bert_int`) you normally do **not** need to pass a config: the loader reads the YAML shown above and deep-merges any overrides that you provide programmatically.

## Available configurations

### 1. `bert_int.yaml` — Two-phase BERT-INT pipeline (DEFAULT)
End-to-end settings for the reference BERT-INT implementation:

- **Paths** (`model.paths`): cache location, dataset root, description dictionary, and checkpoint prefix. Non-absolute paths are resolved against `PROJECT_ROOT`.
- **Basic unit** (phase 1): multilingual BERT encoder, sequence limits, dropout, margin-based loss, negative sampling, batch sizes, evaluation cadence, and CUDA device targeting.
- **Interaction model** (phase 2): neighbour/attribute caps, Gaussian kernel count, MLP hidden size, long-run training schedule (epochs, eval frequency), and additional negatives.
- **Seed block**: stores the global random seed used by both phases.

Use this file as the canonical template when experimenting with BERT-INT. The helper `load_bert_int_config` deep-merges your overrides with these defaults.

### 2. `openea.yaml` — Knowformer/OpenEA hybrid
Configuration for the OpenEA/Knowformer alignment stack:

- **Top-level switches**: device, operating mode (`hybea_without_factual`, etc.), structural model name, reduction ratio, pipeline seeds.
- **`attribute` section**: encoder dimension, epochs, Adam settings, batch sizes, negatives per positive, CSLS factor, and dataset split ratios.
- **`structure` section**: transformer depth, hidden sizes, attention heads, multiple dropout knobs, learning rate, batch sizes, early stopping, soft-label smoothing, and Stochastic Weight Averaging parameters.
- **`datasets` table**: per-dataset candidate top-k and input dimensions for both source and target graphs.

Edit this file when you need to fine-tune OpenEA’s attribute or structural sub-models for a specific benchmark.

### 3. `hybea*.yaml` — HybEA experiment presets
HybEA looks for `config/models/hybea.yaml` and optionally merges `hybea.local.yaml` (developer machine) and `hybea.stage.yaml`
(CI/staging). These files are not committed so you can keep credentials or dataset-specific paths private.

Minimal skeleton:

```yaml
model:
  device: cuda
  encoder:
    hidden_dim: 512
  training:
    epochs: 200
```

Place `hybea.yaml` under `config/models/` and add optional `hybea.local.yaml` / `hybea.stage.yaml` for environment-specific overrides.

### 4. `stub.yaml` — No-op model
An empty file kept for completeness. The `stub` alignment model ignores every field and always reports zero metrics,
making it useful for smoke tests in the pipeline.

## Working with model configs

### Editing guidelines
- Keep every file rooted at a `model:` key, mirroring the structure loaded by the corresponding model class.
- Paths should be relative to the project root whenever possible; the loaders resolve them automatically.
- Prefer YAML anchors or references only if the target model loader supports them.

### Providing overrides
All loaders accept in-memory overrides. Example for BERT-INT:

```python
from src.alignment_models.methods.bert_int.config import load_bert_int_config

config = load_bert_int_config(
    overrides={
        "basic_unit": {"epochs": 8, "batch_size": 32},
        "paths": {"dataset_root": "data/processed/BBC_DB"},
    }
)
```

For HybEA, drop a `hybea.local.yaml` with the fields you want to override; it will be merged automatically.

### Keeping multiple presets
You can duplicate any YAML (e.g., `bert_int.fast.yaml`) and point the loader to it when needed:

```python
config = load_bert_int_config(path=PROJECT_ROOT / "config/models/bert_int.fast.yaml")
```

This allows you to maintain production, debugging, and research variants side by side.

## Related tools
- `src/tools/download_hf_assets.py` resolves weights using `config/models/bert_int.yaml` by default.
- Stage specifications (e.g., `config/experiments/*`) reference the model names registered in `src/alignment_models/registry.py`; those names map directly to the YAMLs documented here.

Keeping this folder organised ensures the training pipeline can reproduce alignment experiments reliably across environments.
