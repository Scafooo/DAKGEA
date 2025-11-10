# DAKGEA

Lightweight runner for reduction → augmentation → evaluation of EA models.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r install/requirements.txt
./scripts/run_experiment.sh config/experiments/02_exp_reduction.yaml
```

Results land under `results/<experiment>/`; metadata and logs keep track of cache hits.

## Config tips

- Single-run configs live in `config/experiments`; sweeping configs are under `config/experiments/massive/bert_int_only_red`.
- Use `reduction.save:false` + `overwrite_existing:false` to reuse cached reductions.
- Disable `augmentation.eval` when only the baseline matters.
- Model overrides go inside `parameters.models.<name>`; defaults live in `config/models/bert_int.yaml`.

## Testing

```bash
pytest tests/
```
