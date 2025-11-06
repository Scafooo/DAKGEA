# Scripts

This directory contains utility scripts and tools for working with DAKGEA.

## Contents

### Experiment Runner

**`run_experiment.sh`** - Run entity alignment experiments

```bash
# Run a single experiment
./scripts/run_experiment.sh config/experiments/my_experiment.yaml

# Run all experiments in a directory
./scripts/run_experiment.sh config/experiments/

# Overwrite existing results
./scripts/run_experiment.sh config/experiments/my_experiment.yaml --overwrite-existing
```

**Features:**
- Multi-stage pipeline (reduction → augmentation → evaluation)
- Support for multiple datasets and models
- Configurable reduction ratios and augmentation methods
- Resume support for interrupted experiments
- Progress tracking and metadata logging

**See also:** [Experiments Documentation](../experiments/README.md)

---

### Dataset Analysis

**`analyze_dataset.sh`** - Analyze and validate HybEA attribute_data format datasets

```bash
# Analyze a dataset
./scripts/analyze_dataset.sh data/raw/hybea/BBC_DB/attribute_data

# Save results to JSON
./scripts/analyze_dataset.sh data/raw/hybea/BBC_DB/attribute_data -o results.json

# Verbose mode
./scripts/analyze_dataset.sh data/raw/hybea/BBC_DB/attribute_data --verbose

# Batch analysis of all datasets
for dataset in data/raw/hybea/*/attribute_data; do
    dataset_name=$(basename $(dirname "$dataset"))
    ./scripts/analyze_dataset.sh "$dataset" -o "results/analysis_${dataset_name}.json"
done
```

**Verifies:**
- Structural invariants (alignment coverage, entity mapping)
- Data distribution (train/test/valid splits)
- Attribute and relation coverage
- Data density statistics

**See also:** [Dataset Analysis Documentation](../experiments/dataset_analysis/README.md)

---

### Format Conversion

**`convert_hybea_to_rdf.py`** - Convert HybEA-formatted datasets to RDF outputs

```bash
# Convert a dataset
python scripts/convert_hybea_to_rdf.py \
    data/raw/hybea/BBC_DB \
    data/rdf/hybea/BBC_DB
```

**Usage:**
```python
# Edit the script to specify input/output paths
input_dir: Path = "data/raw/hybea/fr_en"
output_dir: Path = "data/rdf/hybea/fr_en"
```

**Note:** This script uses hardcoded paths. Edit the `main()` function to change the conversion target.

---

## Directory Structure

```
scripts/
├── README.md                   # This file
├── run_experiment.sh           # Experiment runner
├── analyze_dataset.sh          # Dataset analysis tool
└── convert_hybea_to_rdf.py     # Format converter
```

---

## Common Workflows

### Complete Experiment Workflow

1. **Validate dataset** (optional but recommended):
   ```bash
   ./scripts/analyze_dataset.sh data/raw/hybea/BBC_DB/attribute_data
   ```

2. **Run experiment**:
   ```bash
   ./scripts/run_experiment.sh config/experiments/my_experiment.yaml
   ```

3. **Check results**:
   ```bash
   cat results/my_experiment/metadata.json
   cat results/my_experiment/BBC_DB/evaluation/stub/bert_int.json
   ```

### Batch Processing Multiple Datasets

```bash
# Analyze all datasets
for dataset in data/raw/hybea/*/attribute_data; do
    name=$(basename $(dirname "$dataset"))
    ./scripts/analyze_dataset.sh "$dataset" -o "analysis_${name}.json"
done

# Run experiments on all configs
for config in config/experiments/*.yaml; do
    ./scripts/run_experiment.sh "$config"
done
```

### Development Iteration

```bash
# 1. Quick test with small data
./scripts/run_experiment.sh config/experiments/test.yaml

# 2. Analyze results
cat results/test_experiment/BBC_DB/evaluation/stub/bert_int.json

# 3. Iterate on configuration
vim config/experiments/test.yaml

# 4. Run again with overwrite
./scripts/run_experiment.sh config/experiments/test.yaml --overwrite-existing
```

---

## Notes

- All scripts should be run from the project root directory
- Virtual environment will be activated automatically if `.venv/` exists
- Use `--help` flag for detailed usage information on each script
- Scripts assume the standard DAKGEA project structure

---

## See Also

- [Main Documentation](../README.md)
- [Experiments Module](../experiments/README.md)
- [Dataset Analysis Tool](../experiments/dataset_analysis/README.md)
- [Examples](../examples/)
