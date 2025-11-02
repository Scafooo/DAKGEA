# DAKGEA Developer Guide

This document targets contributors and integrators who want to extend or modify the code. It covers the architecture, plugin mechanisms, testing strategy, and coding conventions.

---

## 1. High-Level Architecture

```
experiments/
  runner/
    runner.py          # Orchestrates experiments
    stages.py          # Reduction/Augmentation/Evaluation stage classes
    specs.py           # Lightweight dataclasses (DatasetSpec, WriterPlan)
src/
  reduction/           # Reduction registry + methods/*
  augmentation/        # Augmentation registry + methods/*
  alignment_models/    # Alignment model registry + methods/*
  core/
    dataset/           # Dataset domain objects, readers, writers
    knowledge_graph/   # KnowledgeGraph domain objects, readers, writers
  config/              # YAML loader (merges global/model/experiment)
  util/                # Registry base, reader/writer helpers, logging utilities
```

The runtime is composed of three stages implemented in `experiments/runner/stages.py`:

1. **`ReductionStage`** – takes a `Dataset`, runs the selected reducer, writes artefacts, and caches metadata.
2. **`AugmentationStage`** – clones the reduced dataset, applies an augmenter, and persists the augmented artefacts.
3. **`EvaluationStage`** – loads EA models from `MODEL_REGISTRY`, runs each model, and writes result JSON plus stage summaries.

`ExperimentRunner` wires these stages together, feeding them the merged configuration and recording metadata under `results/<experiment>/metadata.json`.

---

## 2. Registries and Plugin Loading

Registries live in `src/reduction/registry.py`, `src/augmentation/registry.py`, and `src/alignment_models/registry.py`. Each exports:

- A typed instance (`REDUCTION_REGISTRY`, `AUGMENTATION_REGISTRY`, `MODEL_REGISTRY`).
- A helper such as `load_builtin_reducers()` that imports the built-in modules so their classes can register via decorators.

### Registering a new implementation

1. Create a package under the relevant `methods/` directory (e.g., `src/reduction/methods/my_reducer/model.py`).
2. Implement the class and decorate it:

   ```python
   from src.reduction.registry import REDUCTION_REGISTRY

   @REDUCTION_REGISTRY.register("my_reducer")
   class MyReducer:
       ...
   ```

3. Import the package inside your bootstrap path (for built-ins, add the module path to `_BUILTIN_*_MODULES` in the registry). External plugins can simply `import` their module before invoking the pipeline.

Registries no longer scan packages automatically; this keeps API usage explicit and testable.

---

## 3. Dataset & Knowledge Graph IO

- `src/core/dataset/dataset.py` wraps a pair of `KnowledgeGraph` instances plus aligned entities and provides `clone()`.
- Readers and writers are organised per format (e.g., `src/core/dataset/reader/hybea/model.py`). Every reader/writer subclass sets `file_type` and is auto-registered via `ReaderFactory`/`WriterFactory`.
- To add a new format:
  1. Create a subpackage in `reader/<format>` or `writer/<format>`.
  2. Implement the class, assigning `file_type = "<format>"`.
  3. Export it via the package `__init__`.

Factories still use lazy autoload inside the `core` namespace to avoid import cycles; keep new classes under the existing packages for discovery.

---

## 4. Configuration Flow

`src/config/loader.py` merges YAML in this order:

1. `config/global.yaml`
2. Optional model overrides (`config/models/*.yaml`)
3. Experiment file (`config/experiments/*.yaml`)

Relative paths become absolute using `PROJECT_ROOT`. The merged config is passed to the runner, which extracts:

- Dataset specs (`name`, `reader`, `writers`)
- Reduction ratios and methods
- Augmentation methods and models
- Stage parameters (`parameters.reduction`, `parameters.experiment`, …)

When you add configuration keys, document them in `docs/user-guide.md` (for user-facing settings) or in the developer guide if they’re internal.

---

## 5. Results & Metadata Layout

The runner writes outputs under `results/<experiment>/<dataset>/<ratio>/` following this structure:

```
reduction/
  summary.json
  artefacts/<writer>/
augmentation/
  <augmentation>/
    summary.json
    artefacts/<writer>/
evaluation/
  <variant>/
    summary.json
    <model>.json
metadata.json             # experiment-level index
```

Each stage summary captures method identifiers, ratios, counts, writers, and file paths. This layout intentionally mirrors the stage classes to ease post-processing or tooling built on top of the results folder.

---

## 6. Testing & Tooling

- **Unit tests** reside in `tests/`. `pytest` is the preferred runner:

  ```bash
  pytest tests/
  ```

  Smoke tests cover registry registration and stage behaviour with in-memory stubs. Extend the suite when adding new reducers/augmenters/models.

- **Syntax checks**: `python -m compileall experiments/runner` is a quick way to catch SyntaxError across the orchestrator.

- **Logging**: `src/logger.py` wires coloured console output plus optional file logging. To adjust verbosity globally, edit `config/global.yaml` or call `set_global_level()` in code.

- **style**: The repository targets Python ≥3.10 with type hints. Keep new modules snake_case, prefer small packages (`model.py`, `__init__.py`) mirroring the existing layout. When modifying import paths, ensure the registries or factories still discover the classes.

---

## 7. Adding a New Stage or Writer

1. Create a package in the appropriate namespace (`src/core/dataset/writer/<name>` or `src/augmentation/methods/<name>`).
2. Implement the class, register it via decorator, and expose it in `__init__`.
3. Update the relevant built-in loader tuple so it becomes part of the default bootstrap (unless it’s an optional plugin).
4. Add tests (ideally minimal smoke-like tests to verify registration and basic behaviour).
5. Document usage if users must configure YAML differently.

For new pipeline stages (beyond the reduction/augmentation/evaluation trio), extend `experiments/runner/stages.py` or create a sibling module and invoke it from `ExperimentRunner`. Keep the stage API consistent: `execute(stage_cfg, dataset, ...) -> Dataset`.

---

## 8. Release Checklist

- Update `docs/user-guide.md` and this developer guide with any configuration or architecture changes.
- Ensure bootstrap functions reference new modules.
- Run `pytest` and, if applicable, integration experiments to confirm no regression in results layout.
- Bump version numbers or change logs if you maintain them externally.

---

Happy hacking! If you need to publish third-party plugins or want to refactor further, sync by opening an issue or PR so the community can align on conventions.***
