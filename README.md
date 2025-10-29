# Data Augmentation for Knowledge Graph Entity Resolution

This repository implements a **modular experimentation framework** for **Entity Alignment (EA)** on Knowledge Graphs. It combines dataset reduction, data augmentation, and model training so you can measure how each component impacts alignment accuracy with minimal plumbing.

---

## ⚙️ Configuration

All behaviour is controlled with YAML files:

- `config/global.yaml` – shared defaults such as path aliases and logging level.
- `config/models/*.yaml` – per-model hyperparameters (HybEA, KnowFormer, BERT-INT, …).
- `config/experiments/*.yaml` – dataset slices plus augmentation/reduction knobs, with `overwrite_existing` to force regeneration and `writers` to choose one or more export formats (e.g. HybEA, RDF).

The loader (`src/config/loader.py`) merges files in that order and automatically resolves relative paths to absolute locations rooted at the repository. This allows you to move configuration files without breaking downstream consumers.

---

## 🛠️ Installation

Clone and enter the repository:

```bash
git clone <PROJECT_URL>
cd <PROJECT_NAME>
```

Create a Python environment and install dependencies:

```bash
conda env create -f install/HybEA_env.yml
# or
pip install -r install/requirements.txt
```

---

## 🚀 Running Experiments

The pipeline executes in four stages:

1. **Reduction** – shrink the dataset while preserving coverage of key entities.
2. **Augmentation** – generate additional triples or alignments to diversify training data.
3. **Model training** – fit an EA model (e.g., HybEA) on the processed dataset.
4. **Evaluation** – compute alignment metrics and export detailed diagnostics.

Example invocation:

```bash
python experiments/run.py config/experiments/exp_1.yaml
```

Append `--overwrite-existing` to recompute reductions, augmentations, and results even when cached outputs are present (you can also set `overwrite_existing: true` inside the experiment YAML). Use multiple `writers` entries in the config to emit additional formats such as RDF/Turtle alongside the default HybEA structure.

The command wires up the requested experiment configuration with the chosen model and kicks off the full reduction→augmentation→training sequence.

---

## 🧱 Repository Layout

```
src/
  alignment_models/    # Pluggable EA model registry and implementations
  augmentation/        # Data augmentation registry and components
  reduction/           # Dataset reduction strategies
  config/              # YAML loader utilities
  dataset/             # Dataset wrappers around paired knowledge graphs
  knowledge_graph/     # Thin rdflib-based graph helpers
  util/                # Reader/write helpers and shared utilities
experiments/           # Experiment entry points and output artefacts
tests/                 # Lightweight regression tests for loaders and I/O
```

Registries automatically discover concrete implementations placed under their respective `methods/` packages, allowing you to drop in new strategies without modifying the core pipeline.

---

## 🧭 Working with Data

- Raw datasets belong in `data/raw`; reduction and augmentation outputs are stored under `data/reduced` and `data/augmented`.
- For HybEA inputs the raw location is `data/raw/hybea/<dataset>/{attribute_data,knowformer_data}`; the runner auto-detects the correct reader from the directory layout.
- Reduced/augmented outputs mirror the selected writers: HybEA exports recreate the original `attribute_data`/`knowformer_data` folders, while the RDF writer stores Turtle graphs (`graph_source.ttl`, `graph_target.ttl`) plus `aligned_entities.tsv`.
- The logger resolves all path aliases via the config loader, so updating locations in `config/global.yaml` is enough to relocate storage directories.
- TSV files are read via `src/util/reader.py`, which accepts both `Path` objects and strings and provides compatibility aliases for legacy APIs.

---

## 🧪 Testing

A lightweight regression suite is provided to validate configuration merging and TSV I/O helpers:

```bash
python -m unittest discover -s tests
```

Add your own tests under `tests/` to keep future changes honest, especially when introducing new reduction or augmentation strategies.

---

## 📊 Results

Experiment metrics, intermediate artefacts, and logs are written to:

```
experiments/results/
```

The default location is controlled by `paths.log_file` and other entries inside `config/global.yaml`.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🔗 References

- [HybEA GitHub](https://github.com/fanourakis/HybEA)
