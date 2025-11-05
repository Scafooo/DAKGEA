# DAKGEA - Data Augmentation for Knowledge Graph Entity Alignment

A **modular experimentation framework** for **Entity Alignment (EA)** on Knowledge Graphs. DAKGEA combines dataset reduction, data augmentation, and model training to measure how each component impacts alignment accuracy.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

- **Multi-Format Support**: HybEA, BERT-INT, RDF formats with automatic conversion
- **Flexible Dataset Configuration**: Simple names, explicit readers, or direct paths
- **Two-Phase BERT-INT**: Complete implementation with attribute support
- **Modular Pipeline**: Reduction → Augmentation → Evaluation stages
- **Production Ready**: Comprehensive logging, checkpointing, and error handling

---

## 🛠️ Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Scafooo/DataAug-KG-EntityResolution
cd DAKGEA

# Create environment (choose one)
conda env create -f install/HybEA_env.yml && conda activate hybea
# OR
python -m venv .venv && source .venv/bin/activate && pip install -r install/requirements.txt
```

### Run Your First Experiment

```bash
# Using the helper script
./run.sh config/experiments/01_exp_direct.yaml

# Or directly with Python
python experiments/run.py config/experiments/01_exp_direct.yaml
```

### Create a Custom Experiment

```yaml
# config/experiments/my_experiment.yaml
experiment:
  name: "my_first_experiment"
  dataset:
    name: "hybea/BBC_DB"           # Explicit reader/dataset format
  augmentation:
    method: "stub"                  # No augmentation
    reduction: 0.1                  # Use 10% of training data
  model: bert_int
  seed: 42
  clear: true                       # Clean up intermediate files
```

Run it:
```bash
./run.sh config/experiments/my_experiment.yaml
```

Check results:
```bash
cat results/my_first_experiment/BBC_DB/0.1/evaluation/reduced/bert_int.json
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
experiments/           # Experiment entry points, stage orchestration
tests/                 # Smoke/unit tests
```

Developers should read the [Developer Guide](docs/developer-guide.md) for a deep dive into registries, stages, and coding conventions.

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
├─ augmentation/<augmentation>/artefacts/<writer>/...
└─ evaluation/<variant>/<model>.json
```

Stage summaries (`summary.json`) sit beside each folder to capture method metadata.
`<variant>` is `baseline` for the unaugmented run, otherwise the augmentation key.
The default root is controlled by `paths.results` (and `paths.log_file`) inside `config/global.yaml`.

More context, including troubleshooting tips, lives in the [User Guide](docs/user-guide.md#3-understand-the-outputs).

---

## 📚 Documentation

### User Documentation
- **[User Guide](docs/user-guide.md)** - Installation, running experiments, understanding outputs
- **[Configuration Guide](docs/configuration-guide.md)** - Complete configuration reference with examples
- **[Dataset Guide](docs/dataset-guide.md)** - Dataset formats, readers, writers, and conversions
- **[BERT-INT Guide](docs/bert-int-guide.md)** - BERT-INT model architecture, training, and tuning
- **[FAQ](docs/faq.md)** - Frequently asked questions and troubleshooting

### Developer Documentation
- **[Developer Guide](docs/developer-guide.md)** - Architecture, registries, extending the framework

### Quick Links
- **Common Tasks**
  - [Create an experiment](docs/configuration-guide.md#complete-examples)
  - [Add a custom dataset](docs/dataset-guide.md#adding-custom-datasets)
  - [Configure BERT-INT](docs/bert-int-guide.md#configuration)
  - [Troubleshoot errors](docs/faq.md#errors--troubleshooting)

---

## 📊 Example Results

BERT-INT on BBC_DB (reduction=0.1, seed=11037):

```json
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
  }
}
```

**Note:** All metrics are fractions (0-1 range), not percentages.

---

## 🔧 Key Capabilities

### Multiple Dataset Formats

```yaml
# HybEA format
dataset:
  name: "hybea/BBC_DB"

# BERT-INT format
dataset:
  name: "bert_int/D_W_15K_V1"

# RDF format
dataset:
  name: "rdf/DW_15"

# Direct path (pre-processed data)
dataset:
  path: "/path/to/preprocessed/data"
```

### Flexible Augmentation

```yaml
# Reduction only (no augmentation)
augmentation:
  method: "stub"
  reduction: 0.1      # Keep 10% of training data

# PLM-based augmentation
augmentation:
  method: "plm_augmentation"
  reduction: 0.2
  parameters:
    model_name: "bert-base-multilingual-cased"
    augmentation_factor: 2.0
```

### Multi-Model Evaluation

```yaml
# Compare multiple models
models_to_run: ["bert_int", "hybea"]

# Or sweep reduction ratios
reduction_ratios: [0.1, 0.2, 0.5, 1.0]
```

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🔗 References

- [HybEA GitHub](https://github.com/fanourakis/HybEA)
- [Bert-int](https://github.com/kosugi11037/bert-int?tab=readme-ov-file)
