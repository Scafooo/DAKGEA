# DAKGEA User Guide

This document explains how to install the framework, launch experiments, and interpret the artefacts it produces. It assumes you want to _use_ the pipeline, not extend its internals—see `docs/developer-guide.md` if you are hacking on the code base.

---

## 1. Install & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Scafooo/DataAug-KG-EntityResolution.git
   cd DAKGEA
   ```

2. **Create a Python environment**

   - Conda:

     ```bash
     conda env create -f install/HybEA_env.yml
     conda activate hybea
     ```

   - Or plain Python:

     ```bash
     python -m venv .venv
     source .venv/bin/activate  # .venv\Scripts\activate on Windows
     pip install -r install/requirements.txt
     ```

3. **(Optional) Download vendor assets**

   Some experiments rely on pre-packaged HybEA/BERT-INT resources. If you cloned the repo without the large blobs, run:

   ```bash
   git submodule update --init --recursive
   ```

---

## 2. Run an Experiment

Experiments are defined in `config/experiments/*.yaml`. Each file declares:

- datasets and their readers
- reduction ratios and methods
- augmentation strategies
- EA models to train/evaluate
- writers to persist outputs (HybEA, RDF, …)

To launch an experiment:

```bash
python experiments/run.py config/experiments/exp_1.yaml
```

Useful CLI flags:

- `--overwrite-existing` – recompute every stage even if cached artefacts exist
- `--resume` – force reuse of cached data (overrides config)
- `--no-progress` – disable tqdm progress bars (cleaner logs)

Alternatively, the helper script resolves configs and activates the virtualenv:

```bash
./run.sh config/experiments/exp_1.yaml
```

---

## 3. Understand the Outputs

Generated artefacts land under `results/<experiment>/<dataset>/<ratio>/`:

```
reduction/
  summary.json
  artefacts/<writer>/...
augmentation/
  <augmentation>/
    summary.json
    artefacts/<writer>/...
evaluation/
  <variant>/
    summary.json
    <model>.json
metadata.json
```

- `summary.json` files capture parameters, counts, and paths for each stage.
- `<variant>` is `baseline` for the reduced dataset or the augmentation name.
- Writers mirror the requested formats (e.g., `hybea`, `rdf`).

Intermediate datasets also appear under:

- `data/reduced/<reduction_method>/<writer>/<dataset>/<ratio>/`
- `data/augmented/<reduction>/<augmentation>/<writer>/<dataset>/<ratio>/`

Default paths are configured in `config/global.yaml`.

---

## 4. Customise an Experiment

1. **Copy an existing YAML** from `config/experiments/`.
2. **Edit the sections**:

   ```yaml
   name: "my_experiment"
   datasets:
     - name: "BBC_DB"
       reader: "hybea"
   reduction_method: "random_entities"
   reduction_ratios: [0.1, 0.5]
   augmentation_methods: ["stub"]    # or "plm_augmentation"
   models_to_run: ["hybea"]
   writers:
     - type: "hybea"
       write_reduced: true
       write_augmented: true
       write_results: true
   parameters:
     experiment:
       seed: 42
     reduction:
       random_seed: 42
   ```

3. **Run the new config** with `python experiments/run.py <config-path>`.

Notes:

- Set `write_results: true` on at least one writer to persist metrics.
- To add multiple writers, supply a list (e.g., HybEA + RDF).
- Seeds live under `parameters.experiment.seed` and cascade to reducers/augmenters when not overridden.

---

## 5. Troubleshooting

| Problem | Quick Checks |
|---------|--------------|
| Missing reader or writer | Ensure the `reader`/`writer` key matches a built-in module (`hybea`, `rdf`, …). See the developer guide for extending the registry. |
| Experiment reuses old artefacts | Use `--overwrite-existing` or set `overwrite_existing: true` in the YAML. |
| CUDA out-of-memory (BERT-INT) | Override `parameters.models.bert_int.device: "cpu"` or reduce batch sizes in `config/models/bert_int.yaml`. |
| HybEA pipeline fails | confirm vendor assets are present (`hybea/` folder) and the support Excel files are writable (`data/hybea_support/`). |
| Unexpected path | Update `config/global.yaml` and ensure the destination exists or is writable. |

---

## 6. Keep the Project Healthy

- Run `python -m compileall experiments/runner` to check syntax quickly.
- Execute the smoke tests (requires `pytest`):

  ```bash
  pytest tests/
  ```

- Changing or adding a reducer/augmenter/model? Update the relevant section in `config/experiments/*.yaml` and rerun the experiment.

For deeper architectural details, coding conventions, or instructions on adding new modules, read `docs/developer-guide.md`.
