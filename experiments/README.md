# Experiments Module

This directory contains tools for running and analyzing experiments in DAKGEA.

## Structure

```
experiments/
├── runner/              # Experiment runner (main tool)
│   ├── run.py          # Entry point for running experiments
│   ├── runner.py       # Core experiment orchestration
│   ├── stages.py       # Pipeline stages (reduction, augmentation, evaluation)
│   ├── config.py       # Configuration handling
│   ├── specs.py        # Dataset and experiment specifications
│   ├── progress.py     # Progress tracking
│   └── registry.py     # Component registry
│
├── dataset_analysis/   # Dataset analysis and validation tool
│   ├── run.py         # Entry point for dataset analysis
│   ├── analyzer.py    # Dataset structure analyzer
│   └── README.md      # Detailed documentation
│
└── README.md          # This file
```

## Tools

### 1. Experiment Runner

Run entity alignment experiments with various configurations.

**Quick Start:**
```bash
# Run a single experiment
./run.sh config/experiments/my_experiment.yaml

# Run with specific options
./run.sh config/experiments/my_experiment.yaml --overwrite-existing
```

**Features:**
- Multi-stage pipeline (reduction → augmentation → evaluation)
- Support for multiple datasets and models
- Configurable reduction ratios and augmentation methods
- Phase-specific dataset writers
- Resume support for interrupted experiments
- Progress tracking and metadata logging

**Configuration Example:**
```yaml
experiment:
  name: "my_experiment"
  dataset:
    name: "hybea/BBC_DB"
    writer: bert_int
  reduction:
    method: random_entities
    ratio: 0.1
    writer: bert_int
  augmentation:
    method: stub
    writer: bert_int
  model: bert_int
  seed: 42
  clear: true
```

**See Also:**
- [Main README](../README.md) for general framework overview
- [Config Examples](../config/experiments/) for more configuration samples

---

### 2. Dataset Analysis Tool

Analyze and validate HybEA attribute_data format datasets.

**Quick Start:**
```bash
# Analyze a dataset
./run_analysis.sh data/raw/hybea/BBC_DB/attribute_data

# Save results to JSON
./run_analysis.sh data/raw/hybea/BBC_DB/attribute_data -o results.json
```

**Features:**
- Verify structural invariants (alignment coverage, entity mapping)
- Analyze data distribution (train/test/valid splits)
- Check attribute and relation coverage
- Compute data density statistics
- Export results to JSON for further analysis

**Verified Invariants:**
- ✅ No overlap between alignment splits (ref/sup/valid are disjoint)
- ✅ Complete entity coverage (all entities are aligned)
- ✅ All aligned entities appear in triples
- ✅ All entities in triples are aligned
- ✅ Separate index spaces for KG1 and KG2
- ⚠️ Attributes are optional (some entities may have no attributes)

**See Also:**
- [Dataset Analysis README](dataset_analysis/README.md) for detailed documentation
- [Example Output](dataset_analysis/README.md#output) for sample results

---

## Common Workflows

### Running a Complete Experiment

1. **Validate dataset** (optional but recommended):
   ```bash
   ./run_analysis.sh data/raw/hybea/BBC_DB/attribute_data
   ```

2. **Run experiment**:
   ```bash
   ./run.sh config/experiments/my_experiment.yaml
   ```

3. **Check results**:
   ```bash
   cat results/my_experiment/metadata.json
   cat results/my_experiment/BBC_DB/evaluation/stub/bert_int.json
   ```

### Batch Processing

**Run multiple experiments:**
```bash
for config in config/experiments/*.yaml; do
    ./run.sh "$config"
done
```

**Analyze multiple datasets:**
```bash
for dataset in data/raw/hybea/*/attribute_data; do
    dataset_name=$(basename $(dirname "$dataset"))
    ./run_analysis.sh "$dataset" -o "results/analysis_${dataset_name}.json"
done
```

### Development Workflow

1. **Create new configuration:**
   ```bash
   cp config/experiments/template.yaml config/experiments/my_new_exp.yaml
   # Edit my_new_exp.yaml
   ```

2. **Test with small dataset first:**
   ```yaml
   reduction:
     ratio: 0.01  # Use only 1% of data for quick testing
   ```

3. **Run experiment:**
   ```bash
   ./run.sh config/experiments/my_new_exp.yaml --overwrite-existing
   ```

4. **Analyze results:**
   ```bash
   python -c "
   import json
   with open('results/my_new_exp/BBC_DB/evaluation/stub/bert_int.json') as f:
       results = json.load(f)
   print(f\"Hits@1: {results['hits@1']:.2%}\")
   print(f\"Hits@10: {results['hits@10']:.2%}\")
   "
   ```

---

## Directory Structure After Running

After running experiments, the structure will look like:

```
results/
└── experiment_name/
    ├── metadata.json                    # Experiment metadata
    └── dataset_name/
        ├── reduction/
        │   ├── dataset_format_bert_int/  # Reduced dataset in BERT-INT format
        │   └── summary.json              # Reduction statistics
        ├── augmentation/
        │   └── stub/
        │       ├── dataset_format_bert_int/
        │       └── summary.json
        └── evaluation/
            └── stub/                     # Augmentation method
                ├── bert_int.json         # Model evaluation results
                └── bert_int/             # Model checkpoints
```

---

## Configuration Reference

### Dataset Configuration

```yaml
dataset:
  name: "reader/dataset_name"  # e.g., "hybea/BBC_DB"
  writer: writer_name           # Default writer for all stages
```

### Reduction Configuration

```yaml
reduction:
  method: random_entities       # Reduction method
  ratio: 0.1                    # 10% of data
  writer: bert_int              # Optional: override dataset writer
```

### Augmentation Configuration

```yaml
augmentation:
  method: stub                  # Augmentation method
  writer: bert_int              # Optional: override dataset writer
```

### Model Configuration

```yaml
model: bert_int                 # Model to evaluate
seed: 42                        # Random seed for reproducibility
clear: true                     # Clean up intermediate files
```

---

## Troubleshooting

### Experiment Runner

**Problem: "dataset_workspace not found in lineage"**
- Ensure `dataset.writer` is specified in config
- Check that writer format matches model expectations

**Problem: Experiment fails to resume**
- Use `--overwrite-existing` to start fresh
- Check `results/experiment_name/metadata.json` for state

**Problem: Out of memory**
- Reduce `reduction.ratio` to use less data
- Check model-specific memory requirements

### Dataset Analysis

**Problem: "Dataset path not found"**
- Verify path points to `attribute_data` directory
- Check all required files exist (ent_ids, triples, attr_triples, *_pairs)

**Problem: "INVARIANT VIOLATED"**
- Dataset may be corrupted or in wrong format
- Check specific invariant that failed for details

---

## Adding New Components

### New Reduction Method

1. Create method in `src/reduction/methods/`
2. Register in `src/reduction/registry.py`
3. Use in config: `reduction.method: your_method`

### New Augmentation Method

1. Create method in `src/augmentation/methods/`
2. Register in `src/augmentation/registry.py`
3. Use in config: `augmentation.method: your_method`

### New Model

1. Create model in `src/alignment_models/methods/`
2. Register in `src/alignment_models/registry.py`
3. Use in config: `model: your_model`

---

## Performance Tips

1. **Use resume mode** (default) to avoid recomputing stages
2. **Enable `clear: true`** to save disk space
3. **Run dataset analysis first** to understand data characteristics
4. **Start with small ratios** (0.01-0.1) for quick iteration
5. **Use `--overwrite-existing`** only when config changes

---

## See Also

- [Main Documentation](../README.md)
- [Dataset Analysis Tool](dataset_analysis/README.md)
- [Configuration Examples](../config/experiments/)
- [BERT-INT Model](../src/alignment_models/methods/bert_int/)
