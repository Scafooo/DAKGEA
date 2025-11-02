# Data Augmentation for Knowledge Graph Entity Resolution

This repository implements a **modular experimentation framework** for **Entity Alignment (EA)** on Knowledge Graphs. It combines dataset reduction, data augmentation, and model training so you can measure how each component impacts alignment accuracy with minimal plumbing.

---

## 🛠️ Installation

Clone and enter the repository:

```bash
git clone https://github.com/Scafooo/DataAug-KG-EntityResolution
cd DAKGEA
```

Create a Python environment and install dependencies:

```bash
conda env create -f install/HybEA_env.yml  # or: pip install -r install/requirements.txt
# activate: conda activate hybea   |   source .venv/bin/activate
```

> Need more detail? See the [User Guide](docs/user-guide.md#1-install--setup).

---

## 🚀 Running Experiments

Example invocation:

```bash
python experiments/run.py config/experiments/exp_1.yaml
```

CLI flags like `--overwrite-existing`, `--resume`, and `--no-progress` tweak caching and logging behaviour. All configuration options are documented in the [User Guide](docs/user-guide.md#2-run-an-experiment).

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

## 📚 Further Reading

- [User Guide](docs/user-guide.md) – installation, experiment execution, artefact overview, troubleshooting.
- [Developer Guide](docs/developer-guide.md) – architecture, plugin registration, code structure, testing guidance.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🔗 References

- [HybEA GitHub](https://github.com/fanourakis/HybEA)
- [Bert-int](https://github.com/kosugi11037/bert-int?tab=readme-ov-file)
