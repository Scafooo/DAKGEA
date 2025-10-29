# DAKGEA Usage Guide

This document summarises how to run experiments, configure reductions/augmentations, and where to find the generated artefacts. It complements the quick overview in the project README.

## 1. Running an Experiment

1. Ensure the project root is on `PYTHONPATH`. The `run.sh` helper does this automatically:

   ```bash
   ./run.sh                       # uses config/experiments/exp_3.yaml
   ```

   You can pass a different configuration:

   ```bash
   PYTHONPATH=. python experiments/run.py config/experiments/custom.yaml --no-progress
   ```

2. Optional flags:
   - `--resume` reuses cached reductions/augmentations when the target directory already exists.
   - `--overwrite-existing` forces recomputation of artefacts even when cached copies are present.
   - `--no-progress` disables tqdm progress bars (handy for CI logs).

## 2. Experiment Configuration Schema

Experiment files live under `config/experiments/`. Key sections recognised by the runner:

| Section | Description |
|---------|-------------|
| `name` | Experiment identifier used for result directories. |
| `datasets` | List of datasets to load. Each entry can supply `reader`, `subtype`, and per-dataset writer overrides. |
| `reduction_ratios` | Fractions (0–1) applied to the number of aligned entities to determine reduction targets. |
| `reduction_method` / `augmentation_methods` | Keys resolved via the registries in `src/reduction` and `src/augmentation`. |
| `writers` | Output writers applied after each stage. Each object accepts `type`, `write_reduced`, `write_augmented`, `write_results`. |
| `parameters` | Free-form settings forwarded to reducers, augmenters, and models. Use this to control seeds and heuristics. |

### 2.1 Reduction Parameters

`parameters.reduction` is passed verbatim to the reducer in addition to the auto-injected `target_entities` value. Recognised keys for the bundled `random_entities` reducer:

| Key | Default | Effect |
|-----|---------|--------|
| `random_seed` | `None` | Seed for deterministic sampling. Falls back to `parameters.experiment.seed` if unset. |
| `filter_alignment` | `true` | Whether to drop aligned pairs whose entities lose either relation or attribute triples after reduction. Set to `false` to preserve all sampled pairs (useful for 100 % dumps). |

### 2.2 Experiment-Level Metadata

`parameters.experiment` is merged into the stage config:

```yaml
parameters:
  experiment:
    seed: 42           # shared default for components that accept a seed
    notes: "baseline"  # any extra metadata to log in downstream steps
```

Reducers and augmenters can read additional fields from this object if needed.

### 2.3 Model Configuration

Model-specific defaults live under `config/models/`. For instance, BERT-INT reads its hyper-parameters from `config/models/bert_int.yaml`, which you can override per experiment by adding:

```yaml
parameters:
  models:
    bert_int:
      device: "cuda:1"
      basic_unit:
        epochs: 2
      interaction:
        epochs: 20
```

## 3. Directory Layout

Paths are resolved from `config/global.yaml` and default to:

```
data/raw/
data/reduced/
data/augmented/
experiments/results/
```

### 3.1 Reduced Artefacts

```
data/reduced/<reduction_method>/<writer>/<dataset>/<ratio>/
```

- `ratio` is the percentage (e.g. `10`, `100`).
- HybEA writer recreates `attribute_data/` and `knowformer_data/`.
- RDF writer emits `graph_source.nt`, `graph_target.nt`, and `aligned_entities.tsv`.

### 3.2 Augmented Artefacts

```
data/augmented/<reduction_method>/<augmentation_method>/<writer>/<dataset>/<ratio>/
```

Each augmentation reuses the reduced dataset as input, then writes its artefacts under the requested writers.

## 4. Reproducibility Tips

- Set `parameters.reduction.random_seed` (or `parameters.experiment.seed`) to make random reductions deterministic across runs.
- Consider enabling `filter_alignment: true` when you want strictly consistent triples/alignment connectivity, and `false` when you need to preserve the original pair count.
- Use `--overwrite-existing` when updating reducer logic so cached directories are regenerated with the new behaviour.
- Alignment models may surface additional metrics besides precision/recall/F1; for example, BERT-INT also reports `hits@1`, `hits@10`, and `mrr`.

## 5. Extending the Pipeline

1. Drop new reducers under `src/reduction/methods/` and register them via `@REDUCTION_REGISTRY.register("<name>")`.
2. Add augmenters in `src/augmentation/methods/` and models in `src/alignment_models/methods/`.
3. Update experiment YAML with the new method keys. The runner auto-discovers the implementations via the registries.
