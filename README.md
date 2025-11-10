# (DOCUMENTATION TO BE FIXED)
# DAKGEA - Data Augmentation for Knowledge Graph Entity Alignment

A **modular experimentation framework** for **Entity Alignment (EA)** on Knowledge Graphs. DAKGEA combines dataset reduction, data augmentation, and model training to measure how each component impacts alignment accuracy.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🛠️ Quick Start

### Installation

```bash
# Clone repository
git clone <REPOSITORY_URL>
cd DAKGEA

# Create environment (choose one)
conda env create -f install/HybEA_env.yml && conda activate hybea
# OR
python -m venv .venv && source .venv/bin/activate && pip install -r install/requirements.txt
```

### Run Your First Experiment

```bash
# Using the helper script
./scripts/run_experiment.sh config/experiments/your_config.yaml

# Or directly with Python
python experiments/runner/run.py config/experiments/your_config.yaml
```

### Create a Custom Experiment

```yaml
# config/experiments/my_experiment.yaml
experiment:
  name: "my_first_experiment"
  dataset:
    name: "reader_name/dataset_name"  # Explicit reader/dataset format
  augmentation:
    method: "stub"                     # Augmentation method
    reduction: 0.1                     # Use 10% of training data
  model: model_name                    # Your model
  seed: 42
  clear: true                          # Clean up intermediate files
```

Run it:
```bash
./scripts/run_experiment.sh config/experiments/my_experiment.yaml
```

Check results:
```bash
cat results/my_first_experiment/<dataset>/<ratio>/evaluation/reduced/<model>.json
```

---

## 🧱 Repository Layout

```
src/
  alignment_models/    # Pluggable EA model registry and implementations
  augmentation/        # Data augmentation registry and components
  reduction/           # Dataset reduction strategies
  config/              # YAML loader utilities
  core/                # Canonical dataset/knowledge-graph domain objects + IO
  util/                # Registry utilities, readers/writers helpers, logging
experiments/           # Experiment orchestration and analysis tools
  runner/              # Experiment runner (reduction → augmentation → evaluation)
  dataset_analysis/    # Dataset structure analysis and validation
scripts/               # Utility scripts and tools
  run_experiment.sh    # Main experiment runner
  analyze_dataset.sh   # Dataset analysis tool
  convert_hybea_to_rdf.py  # Format conversion utility
examples/              # Example usage scripts
tests/                 # Smoke/unit tests
config/
  experiments/         # Experiment configurations
  models/              # Model configurations
  global.yaml          # Global settings
```

---

## 🧪 Testing

Run the pytest suite:

```bash
pytest tests/
```

This exercises registry registration, pipeline stages, and IO helpers. Extend the suite whenever you add reducers, augmenters, or models.

---

## 📊 Results

Experiment metrics, intermediate artefacts, and logs are written to:

```
results/<experiment>/<dataset>/<ratio>/
├─ reduction/artefacts/<writer>/...
├─ augmentation/<method>/artefacts/<writer>/...
└─ evaluation/<variant>/<model>.json
```

Stage summaries (`summary.json`) sit beside each folder to capture method metadata.
`<variant>` is `baseline` for the unaugmented run, otherwise the augmentation key.
The default root is controlled by `paths.results` (and `paths.log_file`) inside `config/global.yaml`.

---

## 📚 Documentation

- [User Guide](docs/user-guide.md) – Installation, experiment execution, output overview, troubleshooting
- [Developer Guide](docs/developer-guide.md) – Architecture, plugin registration, code structure, testing guidance

---

## ⚙️ Configuration

### Dataset Formats

DAKGEA supports multiple dataset specification formats:

**Explicit reader/dataset:**
```yaml
dataset:
  name: "reader_name/dataset_name"  # e.g., "hybea/BBC_DB"
```

**Simple name (auto-detect reader):**
```yaml
dataset:
  name: "dataset_name"  # Searches all reader directories
```

**Direct path:**
```yaml
dataset:
  path: "/absolute/path/to/dataset"  # Pre-processed data
```

### Augmentation Configuration

**Modern syntax (recommended):**
```yaml
augmentation:
  method: "method_name"  # Augmentation method
  reduction: 0.1         # Reduction ratio (0-1)
```

**Legacy syntax (still supported):**
```yaml
reduction_ratio: 0.1
augmentation_method: "method_name"
```

### Model Configuration

```yaml
model: "model_name"

# Or for multiple models
models_to_run: ["model_1", "model_2"]

# Override parameters
parameters:
  models:
    model_name:
      parameter1: value1
      parameter2: value2
```

---

## 🔧 CLI Options

```bash
# Overwrite cached data
./run.sh config.yaml --overwrite-existing

# Resume from cache
./run.sh config.yaml --resume

# Disable progress bar
./run.sh config.yaml --no-progress
```

---

## 🚀 Advanced Features

### Multiple Reduction Ratios

```yaml
reduction_ratios: [0.1, 0.2, 0.5, 1.0]  # Test multiple ratios
```

### Custom Writers

```yaml
dataset:
  name: "hybea/BBC_DB"
  writer: "writer_name"  # Specify output format
  # Or multiple writers
  writers:
    - type: "writer_1"
    - type: "writer_2"
```

### Skip Training (Re-evaluation)

```yaml
dataset:
  path: "/path/to/checkpoint"
skip_training: true  # Load existing model
```

---

## 🐛 Common Issues

### "Unable to infer reader for dataset"

**Solution:** Use explicit `reader/dataset` format:
```yaml
dataset:
  name: "reader_name/dataset_name"
```

### "Direct path mode requires 'path' field"

**Solution:** Add reduction ratio for standard mode:
```yaml
augmentation:
  method: "stub"
  reduction: 0.1
```

### Model-specific errors

Check that you've specified the correct writer for your model. Some models require specific dataset formats.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🔗 References

- [HybEA GitHub](https://github.com/fanourakis/HybEA)
- [BERT-INT](https://github.com/kosugi11037/bert-int)

---

## 🙏 Contributing

We welcome contributions! Please see the [Developer Guide](docs/developer-guide.md) for:
- Code structure and conventions
- How to add new models
- How to add new dataset readers/writers
- How to add augmentation methods
- Testing guidelines

---

## 📞 Support

- **Documentation**: Check [docs/](docs/) for guides
- **Issues**: Report bugs via [GitHub Issues](https://github.com/Scafooo/DataAug-KG-EntityResolution/issues)
- **Questions**: Open a discussion on GitHub

---
