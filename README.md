# Data Augmentation for Knowledge Graph Entity Resolution

<div style="display: flex; justify-content: center;">
  <img src="assets/images/banner.png" alt="Data Augmentation for Knowledge Graph Entity Resolution" style="max-width: 85%; height: auto;">
</div>

DAKGEA is a modular Python 3.11 pipeline for dataset reduction, augmentation, and evaluation in Entity Alignment tasks.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r install/requirements.txt
./scripts/run_experiment.sh config/experiments/exp_direct.yaml
```

Artifacts are stored in `results/<experiment>/`, including caches, logs, and JSON summaries.

## Project layout

```
src/
  alignment_models/   # BERT-INT, OpenEA/Knowformer, HybEA, stub implementations
  augmentation/       # PLM augmenter + registry helpers
  reduction/          # Reducers (random_entities, stub, ...)
  core/               # Dataset / knowledge-graph wrappers, readers, writers
config/
  experiments/        # Multi-stage experiment recipes
  augmentation/       # PLM parameter sets (default + variants)
  models/             # Alignment model configurations
scripts/
  run_experiment.sh   # Convenience wrapper around the experiment runner
install/
  requirements.txt    # Development + runtime dependencies
```

Additional notes live in `docs/` (PLM logging, BART tuning, predicate matching).

## Running an experiment

1. Pick a YAML under `config/experiments/` (e.g. `test_structure_final.yaml`).
2. Launch the runner:
   ```bash
   python -m experiments.runner --config config/experiments/test_structure_final.yaml
   ```
3. The pipeline executes reduction → augmentation → evaluation according to the `eval/save` flags.

Each stage writes outputs below `results/<name>/<stage>/`, following the `WriterPlan` definitions in `experiments/runner/specs.py`.

## PLM augmentation

- Default config: `config/augmentation/plm.yaml` (auto-loaded).
- Key knobs:
  - `max_depth`: BFS depth over fused set nodes
  - `ratio`: number of synthetic aligned pairs vs. originals
  - `bart.*`: optional fine-tuning / generation settings
- Log format is documented in `docs/plm_logging_structure.md`.

## Alignment models

- **BERT-INT** — `config/models/bert_int.yaml`, loader in `src/alignment_models/methods/bert_int/config.py`.
- **OpenEA / Knowformer** — `config/models/openea.yaml`.
- **HybEA** — create `config/models/hybea.yaml` (plus optional `.local` / `.stage`).
- **Stub** — `config/models/stub.yaml` for smoke tests.

See `config/models/README.md` for a deeper breakdown.

## Dataset I/O

Dataset readers live under `src/core/dataset/reader`; writers and their names live in `experiments/runner/specs.py`.
Experiments can reference a dataset by registry name (`experiment.dataset.name`) or by explicit filesystem path (`experiment.dataset.path`).

## Contributing & License

- Issues and pull requests are welcome—remember to keep configs and PLM logging in sync.
- License: see `LICENSE`.
