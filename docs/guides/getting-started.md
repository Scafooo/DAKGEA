# Getting Started with DAKGEA

This guide will walk you through installing DAKGEA and running your first experiments.

## 📋 Prerequisites

- **Python**: 3.11 or higher
- **GPU**: Recommended (CUDA-compatible) for BERT-based models
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Disk Space**: ~10GB for dependencies and datasets

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/DAKGEA.git
cd DAKGEA
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r install/requirements.txt
```

This will install all required packages including:
- PyTorch (with CUDA support if available)
- Transformers (for BERT models)
- RDFLib (for knowledge graph handling)
- And other dependencies

### 4. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.11+

# Check if PyTorch can see GPU (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🎯 Your First Experiment

Let's run a simple baseline experiment on a small dataset.

### Step 1: Understand the Experiment Config

Create a simple experiment configuration:

```bash
cat > config/experiments/my_first_experiment.yaml <<EOF
experiment:
  name: my_first_experiment

  # Dataset configuration
  dataset:
    name: openea/D_W_15K_V1  # DBpedia-Wikidata 15K entities
    writer: bert_int

  # Reduction: use 30% of entities
  reduction:
    method: random_entities
    ratio: 0.3
    writer: bert_int
    eval: true  # Evaluate baseline (no augmentation)

  # Model configuration
  model: bert_int
EOF
```

**What this config does:**
- Loads the D_W_15K_V1 dataset (DBpedia-Wikidata alignment)
- Reduces it to 30% of entities
- Evaluates BERT-INT model on the reduced dataset (baseline)
- No augmentation (baseline experiment)

### Step 2: Run the Experiment

```bash
python -m experiments.runner.runner config/experiments/my_first_experiment.yaml
```

**Expected output:**
```
[INFO] Loading dataset: openea/D_W_15K_V1
[INFO] Reduction stage: random_entities (ratio=0.3)
[INFO] Reduced dataset: 4500 entities, 1350 aligned pairs
[INFO] Evaluating baseline...
[INFO] BERT-INT training...
[INFO] Evaluation complete!
[SUCCESS] Experiment finished. Results saved to results/my_first_experiment/
```

### Step 3: Check Results

```bash
# View results
cat results/my_first_experiment/reduction_030/reduction/results.json
```

**Example output:**
```json
{
  "bert_int": {
    "hits@1": 0.7543,
    "hits@5": 0.8621,
    "hits@10": 0.9012,
    "mrr": 0.8123
  }
}
```

**Interpretation:**
- **Hits@1**: 75.43% - The model correctly identifies the aligned entity in the top-1 prediction
- **Hits@5**: 86.21% - Correct entity is in top-5 predictions
- **MRR**: 0.8123 - Mean reciprocal rank (higher is better)

---

## 🔬 Adding Data Augmentation

Now let's add augmentation to see if we can improve performance.

### Step 1: Create Augmented Experiment

```bash
cat > config/experiments/my_augmented_experiment.yaml <<EOF
experiment:
  name: my_augmented_experiment

  dataset:
    name: openea/D_W_15K_V1
    writer: bert_int

  reduction:
    method: random_entities
    ratio: 0.3
    writer: bert_int
    eval: true  # Evaluate baseline

  # Add augmentation
  augmentation:
    method: plm
    ratio: 1.0  # Generate same number of synthetic pairs as original
    training_mode: augmented  # Use original + synthetic
    writer: bert_int
    eval: true  # Evaluate augmented

    # PLM-specific settings
    bart_finetuning:
      enable: true
      epochs: 3
      batch_size: 8

  model: bert_int
EOF
```

### Step 2: Run Augmented Experiment

```bash
python -m experiments.runner.runner config/experiments/my_augmented_experiment.yaml
```

**This will:**
1. Reduce dataset (same as before)
2. Fine-tune BART on entity pairs
3. Generate synthetic pairs
4. Train BERT-INT on original + synthetic data
5. Evaluate and compare

### Step 3: Compare Results

```bash
# Baseline results (original data only)
cat results/my_augmented_experiment/reduction_030/reduction/results.json

# Augmented results (original + synthetic)
cat results/my_augmented_experiment/reduction_030/augmentation/results.json
```

**Expected improvement:**
```
Baseline:   Hits@1 = 75.43%
Augmented:  Hits@1 = 78.21%  (+2.78%)
```

---

## 🎓 Next Steps

### 1. Evaluate Synthetic Data Quality

Learn how to evaluate if synthetic data is high quality:

```bash
# See the Quality Evaluation Guide
cd docs/guides/
cat quality-evaluation.md
```

**Quick command:**
```bash
bash scripts/run_quality_evaluation.sh --model bert_int --fair-comparison
```

### 2. Experiment with Different Models

Try different alignment models:

```yaml
# In your config, change:
model: bert_int   # BERT-based (good for cross-lingual)
# to:
model: rrea       # GNN-based (good for structure-rich)
# or:
model: hybea      # Hybrid approach
```

See [models documentation](../models/overview.md) for details.

### 3. Tune Hyperparameters

Optimize performance with hyperparameter tuning:

```bash
# See the Hyperparameter Tuning Guide
cat docs/testing/hyperparameter-tuning.md
```

### 4. Run Batch Experiments

Run multiple experiments in parallel:

```bash
# Generate baseline configs for all datasets
python scripts/tools/generate_massive_baseline_configs.py

# Run all in parallel (4 jobs)
bash scripts/run_experiments_parallel.sh \
    --dir config/experiments/massive/bert_int_baseline \
    --jobs 4
```

### 5. Generate Publication Tables

Create LaTeX tables from results:

```bash
python experiments/statistics/generate_latex_tables.py
```

See [LaTeX output guide](latex-output.md) for details.

---

## 📊 Understanding the Pipeline

DAKGEA follows a clear pipeline:

```
1. Load Dataset
   └─> openea/D_W_15K_V1 (15,000 entities)

2. Reduction Stage
   └─> Random sampling to 30% → 4,500 entities
   └─> Evaluate baseline (optional)

3. Augmentation Stage (optional)
   └─> Fine-tune BART on entity pairs
   └─> Generate synthetic pairs
   └─> Augmented dataset = Original + Synthetic

4. Filtering Stage (optional)
   └─> Select training mode:
       • baseline: original only
       • augmented: original + synthetic
       • synthetic_only: synthetic only

5. Training & Evaluation
   └─> Train alignment model (BERT-INT, RREA, etc.)
   └─> Evaluate on test set
   └─> Save metrics (Hits@K, MRR, etc.)
```

---

## 🔧 Common Configuration Patterns

### Baseline (No Augmentation)

```yaml
experiment:
  name: baseline_experiment
  dataset:
    name: openea/D_W_15K_V1
  reduction:
    ratio: 0.3
    eval: true
  # NO augmentation section
  model: bert_int
```

### Augmented (Original + Synthetic)

```yaml
experiment:
  name: augmented_experiment
  dataset:
    name: openea/D_W_15K_V1
  reduction:
    ratio: 0.3
    eval: true
  augmentation:
    method: plm
    ratio: 1.0
    training_mode: augmented  # Default
    eval: true
  model: bert_int
```

### Synthetic-Only (For Quality Evaluation)

```yaml
experiment:
  name: synthetic_only_experiment
  dataset:
    name: openea/D_W_15K_V1
  reduction:
    ratio: 0.3
    eval: true
  augmentation:
    method: plm
    ratio: 1.0
    training_mode: synthetic_only  # Only synthetic pairs
    eval: true
  model: bert_int
```

---

## 🐛 Troubleshooting

### Out of Memory (GPU)

**Error**: `CUDA out of memory`

**Solution**:
```yaml
# Reduce batch size in your config
augmentation:
  bart_finetuning:
    batch_size: 4  # Was 8 or 16
```

Or use CPU:
```bash
CUDA_VISIBLE_DEVICES="" python -m experiments.runner.runner config.yaml
```

### Module Not Found

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Make sure you're running from project root
cd /path/to/DAKGEA

# Run with python -m
python -m experiments.runner.runner config.yaml
```

### Slow Performance

**Issue**: Experiments taking too long

**Solutions**:
1. Use smaller reduction ratio (e.g., 0.1 instead of 0.3)
2. Reduce BART fine-tuning epochs
3. Use smaller batch sizes
4. Run on GPU instead of CPU
5. Use parallel execution for multiple experiments

---

## 📚 Further Reading

- **[Quality Evaluation Guide](quality-evaluation.md)**: Deep dive into evaluating synthetic data
- **[Configuration Reference](../configuration/overview.md)**: Complete config documentation
- **[Architecture Guide](../architecture/overview.md)**: Understand the system design
- **[Metrics Guide](../experiments/metrics.md)**: Understanding evaluation metrics

---

## 💡 Tips for Success

1. **Start Small**: Use small datasets and low reduction ratios for testing
2. **Check Results**: Always verify results files exist after experiments
3. **Use Resume**: Add `resume: true` to skip completed stages
4. **Monitor Resources**: Watch GPU memory usage with `nvidia-smi`
5. **Save Configs**: Keep successful configs for reproducibility
6. **Read Logs**: Check `results/*/logs/` for detailed execution logs

---

## 🤝 Getting Help

- **Documentation**: Check [docs/index.md](../index.md) for all documentation
- **Issues**: Report problems on GitHub Issues
- **Examples**: See `config/experiments/` for example configs

---

**Happy experimenting! 🚀**

---

**Last Updated:** 2025-12-15
