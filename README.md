# DAKGEA – Data Augmentation for Knowledge-Graph Entity Alignment

DAKGEA orchestrates reduction → augmentation → evaluation pipelines to measure how data preparation affects Entity Alignment models. It combines reusable readers/writers, registry-driven reducers/augmenters/models, and a lightweight runner that records metadata, cache hints, and artefacts in `results/…`.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick start

1. **Install**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r install/requirements.txt
   ```

2. **Run an experiment**
   ```bash
   ./scripts/run_experiment.sh config/experiments/02_exp_reduction.yaml
   ```

3. **Inspect outputs**
   ```
   tree results/02_exp_reduction
   cat results/02_exp_reduction/metadata.json
   ```

## Repository structure

```
config/               # experiment + model + global settings
docs/                 # high-level user/developer guides
experiments/runner/   # runner, specs, stages, registries
results/              # experiments write reduction/augmentation/evaluation artefacts + metadata
scripts/              # helpers (run, analyze, convert)
src/
  alignment_models/   # pluggable EA models (bert_int, hybea, stub, etc.)
  augmentation/        # augmentation registry & implementations
  reduction/           # reducers (random_entities, etc.)
  core/                # dataset/knowledge-graph abstractions + IO helpers
  util/                # registries, readers/writers, logging helpers
tests/                # smoke/unit tests
```

## Configuration notes

- `experiment.dataset` accepts `reader/dataset`, simple names, or `path`.
- Reduction defaults to `random_entities`; set `reduction.save:true` to persist intermediate datasets.
- Augmentation is optional; use `augmentation.eval:false` to skip evaluation.
- Models are selected via `model`/`models_to_run`; overrides go under `parameters.models.<name>`.
- `config/experiments/massive/bert_int_only_red` contains generated configs covering the common HybEA datasets × reduction ratios.

## Documentation & testing

- **User + developer guides**: `docs/user-guide.md`, `docs/developer-guide.md`.
- **Testing**: `pytest tests/`
- **Logging**: Root logs go under `results/logs/log.txt`; structured metadata lives in each experiment workspace.

## Next steps

- Run the generated configs under `config/experiments/massive/bert_int_only_red` to sweep datasets × ratios automatically.
- Inspect `src/alignment_models/methods/bert_int` step-by-step: `basic_unit` trains embeddings, then `interaction_model` aggregates neighbor/description views + the MLP.
- Extend `src/reduction` and `src/augmentation` registries for new strategies and wire them into the runner via YAML.
- Track experiments using `results/<experiment>/metadata.json` plus the central `results/logs/log.txt`; cleanups triggered by `clear: true` remove only `artifact/` checkpoints.
- Add diagrams/notes in `docs/` if you want visual depictions of the reduction→augmentation→evaluation pipeline or data flow between `reader → reducer → model`.
