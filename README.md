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
  alignment_models/   # Alignment Models: BERT-INT, OpenEA/Knowformer ...
  augmentation/       # Augmenter: PLM augmenter ...
  reduction/          # Reducers: Random Entities, ...
  core/               # Core classes I/O ...
config/
  experiments/        # Multi-stage experiment recipes
  augmentation/       # PLM parameter sets (default + variants)
  models/             # Alignment model configurations
scripts/
  run_experiment.sh   # Convenience wrapper around the experiment runner
install/
  requirements.txt    # Development + runtime dependencies
```

Additional notes live in `docs/`.

## Running an experiment

1. Pick a YAML under `config/experiments/` (e.g. `test_structure_final.yaml`).
2. Launch the runner:
   ```bash
   python -m experiments.runner --config config/experiments/test_structure_final.yaml
   ```
3. The pipeline executes reduction → augmentation → evaluation according to the `eval/save` flags.

Each stage writes outputs below `results/<name>/<stage>/`, following the `WriterPlan` definitions in `experiments/runner/specs.py`.

## Contributing & License

- Issues and pull requests are welcome—remember to keep configs and PLM logging in sync.
- License: see `LICENSE`.
Non 