# Experiment Suites

**Grouping Related Experiments for Better Organization**

Starting from version 0.2.0, DAKGEA supports **experiment suites** - a way to group related experiments into a common directory for better organization and easier analysis.

---

## 🎯 Problem Statement

Without suites, running massive experiments creates hundreds of directories scattered in `results/`:

```
results/
├── D_W_15K_V1_baseline_0.1/
├── D_W_15K_V1_baseline_0.2/
├── D_W_15K_V1_baseline_0.3/
├── ... (697 more directories!)
├── D_W_15K_V1_synthetic_0.1_0.1/
├── D_W_15K_V1_synthetic_0.1_0.2/
└── ... (too many to manage!)
```

**Problems:**
- ❌ Hard to find specific experiments
- ❌ Difficult to analyze related experiments together
- ❌ Cluttered results directory
- ❌ No clear grouping by experiment campaign

---

## ✅ Solution: Experiment Suites

With suites, related experiments are grouped together:

```
results/
├── massive_baseline_bert_int/          # ← Suite
│   ├── D_W_15K_V1_baseline_0.1/
│   ├── D_W_15K_V1_baseline_0.2/
│   ├── ... (70 experiments)
│   └── metadata.json
│
├── massive_synthetic_only_bert_int/    # ← Suite
│   ├── D_W_15K_V1_synthetic_0.1_0.1/
│   ├── D_W_15K_V1_synthetic_0.1_0.2/
│   ├── ... (700 experiments)
│   └── metadata.json
│
└── quality_evaluation_custom/          # ← Custom suite
    ├── my_experiment_1/
    ├── my_experiment_2/
    └── ...
```

**Benefits:**
- ✅ Clear organization by experiment campaign
- ✅ Easy to analyze all experiments in a suite
- ✅ Clean results directory
- ✅ Better experiment tracking

---

## 📝 Configuration

### Adding a Suite to Your Experiment

Simply add the `suite` field to your experiment configuration:

```yaml
experiment:
  suite: "my_experiment_suite"  # ← Add this line
  name: my_experiment
  dataset:
    name: openea/D_W_15K_V1
  ...
```

### Suite Naming Conventions

**Recommended patterns:**

| Pattern | Example | Use Case |
|---------|---------|----------|
| `{purpose}_{model}` | `massive_baseline_bert_int` | Model-specific campaigns |
| `{purpose}_{date}` | `quality_eval_2025-12-15` | Timestamped campaigns |
| `{project}_{phase}` | `thesis_preliminary` | Project phases |
| `{dataset}_{variant}` | `dbpedia_cross_lingual` | Dataset-specific studies |

**Guidelines:**
- Use lowercase with underscores: `my_suite_name`
- Be descriptive but concise
- Include context (model, date, purpose)
- Avoid special characters

---

## 🔧 Usage Examples

### Example 1: Baseline Experiments

```yaml
experiment:
  suite: "baseline_bert_int_2025"
  name: D_W_15K_V1_baseline_0.3
  dataset:
    name: openea/D_W_15K_V1
  reduction:
    ratio: 0.3
    eval: true
  model: bert_int
```

**Result:** `results/baseline_bert_int_2025/D_W_15K_V1_baseline_0.3/`

### Example 2: Quality Evaluation

```yaml
experiment:
  suite: "quality_evaluation"
  name: D_W_15K_V1_synthetic_only_0.3_1.0
  dataset:
    name: openea/D_W_15K_V1
  reduction:
    ratio: 0.3
  augmentation:
    method: plm
    ratio: 1.0
    training_mode: synthetic_only
  model: bert_int
```

**Result:** `results/quality_evaluation/D_W_15K_V1_synthetic_only_0.3_1.0/`

### Example 3: No Suite (Backward Compatible)

```yaml
experiment:
  # No suite field - uses old behavior
  name: my_standalone_experiment
  dataset:
    name: openea/D_W_15K_V1
  model: bert_int
```

**Result:** `results/my_standalone_experiment/` (old behavior)

---

## 🛠️ Generating Configs with Suites

The config generation scripts automatically add appropriate suite names:

### Baseline Configs

```bash
python scripts/tools/generate_massive_baseline_configs.py
```

**Generates configs with:**
- Suite: `massive_baseline_bert_int` (for BERT-INT)
- Suite: `massive_baseline_rrea` (for RREA)

### Synthetic-Only Configs

```bash
python scripts/tools/generate_massive_synthetic_only_configs.py
```

**Generates configs with:**
- Suite: `massive_synthetic_only_bert_int`
- Suite: `massive_synthetic_only_rrea`

---

## 📊 Analyzing Suite Results

### Option 1: Using `--suite` Flag

```bash
# Analyze a specific suite
bash scripts/analyze_results.sh --suite massive_baseline_bert_int

# With full analysis
bash scripts/analyze_results.sh --suite quality_evaluation --full
```

### Option 2: Direct Path

```bash
# Analyze suite by path
bash scripts/analyze_results.sh results/massive_baseline_bert_int/
```

### Option 3: All Results (Default)

```bash
# Analyze everything in results/
bash scripts/analyze_results.sh
```

---

## 📂 Directory Structure

### With Suites

```
results/
├── massive_baseline_bert_int/
│   ├── metadata.json                   # Suite-level metadata
│   ├── D_W_15K_V1_baseline_0.1/
│   │   ├── reduction_010/
│   │   │   ├── reduction/
│   │   │   │   └── results.json
│   │   │   └── summary.json
│   │   └── metadata.json
│   ├── D_W_15K_V1_baseline_0.2/
│   └── ...
│
└── massive_synthetic_only_bert_int/
    ├── metadata.json
    ├── D_W_15K_V1_synthetic_0.1_0.1/
    └── ...
```

### Metadata Files

**Suite-level metadata** (`results/suite_name/metadata.json`):
```json
{
  "suite": "massive_baseline_bert_int",
  "experiments": [
    "D_W_15K_V1_baseline_0.1",
    "D_W_15K_V1_baseline_0.2",
    ...
  ],
  "created_at": "2025-12-15T14:30:00",
  "total_experiments": 70
}
```

**Experiment-level metadata** (unchanged):
```json
{
  "name": "D_W_15K_V1_baseline_0.1",
  "suite": "massive_baseline_bert_int",
  "reduction_method": "random_entities",
  "model": "bert_int",
  ...
}
```

---

## 🎯 Best Practices

### 1. Group by Purpose

✅ **Good:**
```yaml
suite: "quality_evaluation_thesis"
suite: "hyperparameter_tuning_bert"
suite: "cross_lingual_comparison"
```

❌ **Bad:**
```yaml
suite: "experiments"          # Too generic
suite: "test123"              # Not descriptive
suite: "My Experiments!"      # Spaces and special chars
```

### 2. Include Context

✅ **Good:**
```yaml
suite: "massive_baseline_bert_int"  # Model included
suite: "quality_eval_2025-12-15"    # Date included
```

❌ **Bad:**
```yaml
suite: "baseline"             # Which model? Which campaign?
suite: "experiment_1"         # What's the purpose?
```

### 3. Be Consistent

If using `massive_baseline_bert_int`, also use:
- `massive_synthetic_only_bert_int`
- `massive_augmented_bert_int`

Not:
- `synthetic-only-bert-int` (different separator)
- `BERT_INT_SYNTHETIC` (different style)

---

## 🔄 Migration Guide

### Migrating Existing Experiments

If you have existing experiments without suites:

1. **Leave them as-is**: Old experiments still work fine
2. **Use suites for new experiments**: Start using suites going forward
3. **Optionally reorganize**:

```bash
# Create suite directory
mkdir -p results/legacy_experiments

# Move old experiments
mv results/D_W_15K_V1_* results/legacy_experiments/

# Update metadata
# (manually edit metadata.json files to add suite field)
```

### Backward Compatibility

**Suites are completely optional!**

- ✅ Old configs without `suite` still work
- ✅ Results go to `results/experiment_name/` as before
- ✅ Analysis scripts work with both structures
- ✅ No breaking changes

---

## 📚 Related Documentation

- [Experiment Configuration](experiments.md) - General experiment setup
- [Quality Evaluation Guide](../guides/quality-evaluation.md) - Running quality experiments
- [Synthetic Comparison](../guides/synthetic-comparison.md) - Comparing training modes

---

## 💡 Tips

### Tip 1: List All Suites

```bash
# List all suite directories
ls -d results/*/

# Count experiments per suite
for suite in results/*/; do
  echo "$(basename $suite): $(find $suite -mindepth 1 -maxdepth 1 -type d | wc -l) experiments"
done
```

### Tip 2: Suite-Specific Analysis

```bash
# Generate LaTeX tables for specific suite
bash scripts/analyze_results.sh \
  --suite massive_baseline_bert_int \
  --export-formats latex

# Quick TSV export only
bash scripts/analyze_results.sh \
  --suite quality_evaluation \
  --basic
```

### Tip 3: Cleaning Up

```bash
# Remove all experiments from a suite
rm -rf results/old_suite_name/

# Archive a suite
tar -czf archive_quality_eval.tar.gz results/quality_evaluation/
```

---

**Last Updated:** 2025-12-15
