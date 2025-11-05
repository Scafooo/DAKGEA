# DAKGEA Quick Reference Cheatsheet

Fast reference for common tasks and commands in DAKGEA.

---

## 🚀 Quick Commands

```bash
# Run experiment
./run.sh config/experiments/my_config.yaml

# Overwrite cached data
./run.sh config.yaml --overwrite-existing

# Resume from cache
./run.sh config.yaml --resume

# Disable progress bar
./run.sh config.yaml --no-progress

# Run tests
pytest tests/

# Check Python syntax
python -m compileall experiments/
```

---

## 📝 Configuration Templates

### Minimal Experiment
```yaml
experiment:
  name: "my_experiment"
  dataset:
    name: "hybea/BBC_DB"
    writer: bert_int        # Required for BERT-INT
  augmentation:
    method: "stub"
    reduction: 0.1
  model: bert_int
  seed: 42
```

### Direct Path Mode
```yaml
experiment:
  name: "direct_test"
  dataset:
    path: "/absolute/path/to/data"
  model: bert_int
  skip_training: true
```

### Multi-Model Comparison
```yaml
experiment:
  name: "comparison"
  dataset:
    name: "hybea/BBC_DB"
  augmentation:
    method: "stub"
    reduction: 0.2
  models_to_run: ["bert_int", "hybea"]
  seed: 42
```

### Parameter Override
```yaml
experiment:
  name: "custom_params"
  dataset:
    name: "hybea/BBC_DB"
  augmentation:
    method: "stub"
    reduction: 0.1
  model: bert_int
  parameters:
    models:
      bert_int:
        basic_unit:
          epochs: 10
          batch_size: 128
        interaction_model:
          epochs: 50
          learning_rate: 1e-3
```

---

## 🗂️ Directory Structure

```
DAKGEA/
├── config/
│   ├── experiments/*.yaml      # Your experiment configs
│   ├── models/*.yaml          # Model default configs
│   └── global.yaml            # Global settings
├── data/
│   └── raw/                   # Raw datasets
│       ├── hybea/
│       ├── bert_int/
│       └── rdf/
├── results/                   # Experiment outputs
│   └── <experiment>/
│       └── <dataset>/
│           └── <ratio>/
│               ├── reduction/
│               ├── augmentation/
│               └── evaluation/
├── src/                       # Source code
└── docs/                      # Documentation
```

---

## 📊 Results Locations

| Item | Path |
|------|------|
| Metrics JSON | `results/<exp>/<dataset>/<ratio>/evaluation/<variant>/<model>.json` |
| Checkpoints | `results/<exp>/evaluation/bert_int/<variant>/*.pth` |
| Logs | `results/<exp>/log.txt` |
| Metadata | `results/<exp>/metadata.json` |
| Reduced dataset | `results/<exp>/<dataset>/<ratio>/reduction/artefacts/<writer>/` |
| Augmented dataset | `results/<exp>/<dataset>/<ratio>/augmentation/<method>/artefacts/<writer>/` |

---

## 🎯 Dataset Formats

### HybEA
```yaml
dataset:
  name: "hybea/BBC_DB"
  # Optional: specify variant
  subtype: "attribute_data"
```

### BERT-INT
```yaml
dataset:
  name: "bert_int/D_W_15K_V1"
  # Will auto-convert if source is HybEA
```

### RDF
```yaml
dataset:
  name: "rdf/DW_15"
```

### Direct Path
```yaml
dataset:
  path: "/path/to/preprocessed/data"
```

---

## 🔧 Common Parameters

### BERT-INT Basic Unit
```yaml
parameters:
  models:
    bert_int:
      basic_unit:
        encoder_name: "bert-base-multilingual-cased"
        max_seq_length: 128
        output_dim: 300
        epochs: 20
        batch_size: 256
        learning_rate: 5e-5
        margin: 3.0
```

### BERT-INT Interaction Model
```yaml
parameters:
  models:
    bert_int:
      interaction_model:
        kernel_num: 21
        candidate_topk: 50
        epochs: 100
        batch_size: 128
        learning_rate: 5e-4
        margin: 1.0
```

### Device Configuration
```yaml
parameters:
  models:
    bert_int:
      device: "cuda:0"      # GPU
      # device: "cpu"       # CPU
```

---

## 🐛 Troubleshooting Quick Fixes

### "Unable to infer reader"
```yaml
# ❌ Wrong
dataset:
  name: "BBC_DB"

# ✅ Right
dataset:
  name: "hybea/BBC_DB"
```

### "Direct path mode requires 'path'"
```yaml
# ❌ Missing reduction
augmentation:
  method: "stub"

# ✅ Add reduction
augmentation:
  method: "stub"
  reduction: 0.1
```

### "dataset_workspace not found in lineage"
```yaml
# ❌ Missing writer for BERT-INT
dataset:
  name: "hybea/BBC_DB"
model: bert_int

# ✅ Add writer
dataset:
  name: "hybea/BBC_DB"
  writer: bert_int        # Required!
model: bert_int
```

### CUDA Out of Memory
```yaml
# Reduce batch sizes
parameters:
  models:
    bert_int:
      basic_unit:
        batch_size: 128       # Down from 256
      interaction_model:
        batch_size: 64        # Down from 128
```

### Too Slow (CPU)
```yaml
# Use GPU
parameters:
  models:
    bert_int:
      device: "cuda:0"
      basic_unit:
        batch_size: 256       # Larger with GPU
```

---

## 📖 File Operations

### Check Dataset Stats
```python
from src.core.dataset.reader import DatasetReaderFactory

reader = DatasetReaderFactory.create("hybea")
dataset = reader.read("data/raw/hybea/BBC_DB/attribute_data")

print(f"Source: {len(dataset.knowledge_graph_source)} triples")
print(f"Target: {len(dataset.knowledge_graph_target)} triples")
print(f"Alignments: {len(dataset.entity_alignment)} pairs")
```

### Load Results
```python
import json

with open("results/my_exp/BBC_DB/0.1/evaluation/reduced/bert_int.json") as f:
    results = json.load(f)

print(f"Hits@1: {results['hits@1']:.4f}")
print(f"Hits@10: {results['hits@10']:.4f}")
print(f"MRR: {results['mrr']:.4f}")
```

### Convert Format
```python
from src.core.dataset.reader import DatasetReaderFactory
from src.core.dataset.writer import DatasetWriterFactory

# Read HybEA
reader = DatasetReaderFactory.create("hybea")
dataset = reader.read("data/raw/hybea/BBC_DB/attribute_data")

# Write BERT-INT
writer = DatasetWriterFactory.create("bert_int")
writer.write(dataset, "output/BBC_DB_bert_int")
```

---

## 📈 Metrics Reference

| Metric | Range | Description |
|--------|-------|-------------|
| Hits@1 | 0-1 | Fraction with correct match in top-1 |
| Hits@5 | 0-1 | Fraction with correct match in top-5 |
| Hits@10 | 0-1 | Fraction with correct match in top-10 |
| MRR | 0-1 | Mean reciprocal rank |
| MR | 1-∞ | Mean rank (lower is better) |

**All metrics are fractions (0-1), not percentages!**

Example:
```json
{
  "hits@1": 0.3456,    // 34.56% accuracy
  "hits@10": 0.7823,   // 78.23% in top-10
  "mrr": 0.5234        // Mean reciprocal rank
}
```

---

## 🔑 Keyboard Shortcuts (Terminal)

```bash
# Cancel running experiment
Ctrl+C

# Send to background
Ctrl+Z
bg

# View last log entries
tail -f results/<experiment>/log.txt

# Follow multiple logs
tail -f results/*/log.txt

# Search in logs
grep "ERROR" results/<experiment>/log.txt
grep -i "hits@1" results/<experiment>/log.txt
```

---

## 💾 Git Workflow

```bash
# Check what changed
git status

# Add experiment config
git add config/experiments/my_exp.yaml

# Commit
git commit -m "Add experiment: my_exp"

# Don't commit large files
# Add to .gitignore:
results/
*.pth
*.pt
data/reduced/
data/augmented/
```

---

## 🔍 Useful Filters

### Find Best Result
```bash
# Best Hits@1
find results -name "bert_int.json" -exec jq -r '"\(.hits@1) \(input_filename)"' {} \; | sort -rn | head -1

# Best MRR
find results -name "bert_int.json" -exec jq -r '"\(.mrr) \(input_filename)"' {} \; | sort -rn | head -1
```

### Compare Experiments
```bash
# List all results
find results -name "bert_int.json" | xargs jq '{exp: input_filename, hits1: .["hits@1"], mrr: .mrr}'

# Compare specific metric
for f in results/*/BBC_DB/0.1/evaluation/reduced/bert_int.json; do
    echo "$f: $(jq -r '.["hits@1"]' $f)"
done
```

---

## 🎓 Common Workflows

### Development Workflow
```bash
# 1. Create config
nano config/experiments/test.yaml

# 2. Test run
./run.sh config/experiments/test.yaml

# 3. Check results
cat results/test/*/0.1/evaluation/reduced/bert_int.json | jq

# 4. Iterate
nano config/experiments/test.yaml
./run.sh config/experiments/test.yaml --overwrite-existing
```

### Production Workflow
```bash
# 1. Use descriptive name
cp config/experiments/template.yaml config/experiments/20250105_production_run.yaml

# 2. Set seed for reproducibility
# seed: 42

# 3. Run
./run.sh config/experiments/20250105_production_run.yaml

# 4. Save results
cp -r results/20250105_production_run results/archive/

# 5. Commit config
git add config/experiments/20250105_production_run.yaml
git commit -m "Production run 2025-01-05"
```

---

## 📚 Documentation Links

| Topic | Link |
|-------|------|
| Installation | [User Guide](user-guide.md#1-install--setup) |
| Configuration | [Configuration Guide](configuration-guide.md) |
| Datasets | [Dataset Guide](dataset-guide.md) |
| BERT-INT | [BERT-INT Guide](bert-int-guide.md) |
| Troubleshooting | [FAQ](faq.md#errors--troubleshooting) |
| Development | [Developer Guide](developer-guide.md) |

---

## 🆘 Emergency Help

```bash
# Something crashed? Check logs
tail -100 results/<experiment>/log.txt

# Config not working? Validate syntax
python -c "import yaml; yaml.safe_load(open('config/experiments/my_config.yaml'))"

# Import error? Check environment
which python
pip list | grep torch

# Dataset not found? List available
ls -la data/raw/*/

# Results missing? Check experiment ran
ls -la results/<experiment>/

# Still stuck? Check FAQ
cat docs/faq.md | grep -A 5 "your error message"
```

---

## 💡 Pro Tips

1. **Use explicit reader format**: `hybea/BBC_DB` instead of `BBC_DB`
2. **Set seeds consistently**: Ensures reproducibility
3. **Enable clear mode**: Saves disk space after experiments
4. **Use descriptive names**: Include date and parameters
5. **Version control configs**: Track experiment evolution
6. **Don't commit results**: Too large for git
7. **Monitor GPU usage**: `watch -n 1 nvidia-smi`
8. **Use tmux/screen**: For long-running experiments
9. **Validate first**: Dry-run before full experiment
10. **Document changes**: Update CHANGELOG.md

---

## 📞 Get Help

- **Documentation**: [docs/README.md](README.md)
- **FAQ**: [docs/faq.md](faq.md)
- **Issues**: [GitHub Issues](https://github.com/Scafooo/DataAug-KG-EntityResolution/issues)

---

**Print this cheatsheet and keep it handy!** 📎
