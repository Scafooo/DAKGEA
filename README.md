# Data Augmentation for Knowledge Graph Entity Resolution (DAKGEA)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r install/requirements.txt
./scripts/run_experiment.sh config/experiments/02_exp_reduction.yaml
```

Results land under `results/<experiment>/`; metadata and logs keep track of cache hits.

---

## Testing

```bash
pytest tests/
```
