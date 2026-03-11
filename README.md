<h1>LUFFY — Data Augmentation for Knowledge Graph Entity Resolution</h1>

<div style="display: flex; justify-content: center;">
  <img src="assets/images/banner.png" alt="Data Augmentation for Knowledge Graph Entity Resolution" style="max-width: 55%; height: auto;">
</div>

**LUFFY** (*Latent-space Unified Framework for data augmentation in KG Entity resolution*) — or **DAKGEA** — is a modular Python 3.11 pipeline for dataset reduction, augmentation, and evaluation in Entity Alignment tasks.

## 📚 Documentation

**Complete documentation is available in the [Documentation Wiki](docs/index.md)**

Quick links:
- 🚀 [Getting Started Guide](docs/guides/getting-started.md)
- 🏗️ [Architecture](docs/architecture/overview.md)
- ⚙️ [Configuration](docs/configuration/overview.md)
- 🧪 [Experiments & Metrics](docs/experiments/overview.md)
- 🤖 [Models](docs/models/overview.md)

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

---

<div style="text-align: center; margin-top: 2em; opacity: 0.6;">
  <img src="assets/images/onepiece_flag.jpg" alt="Straw Hat Pirates" style="height: 40px;"><br>
  <sub><em>"I don't want to conquer anything. I just think the guy with the most freedom in this whole ocean is the Pirate King."</em> — Monkey D. Luffy</sub><br><br>
  <sub><strong>Hito Hito no Mi, Model: Nika</strong> — The most ridiculous power in the world. A Mythical Zoan fruit that transforms its user into the legendary Sun God Nika, granting a rubber body whose only limit is the user's imagination.</sub>
</div>