# Testing and Hyperparameter Tuning

This section contains documentation for testing workflows and hyperparameter optimization.

## 📋 Contents

### [Hyperparameter Tuning](hyperparameter-tuning.md)
**Systematic Hyperparameter Optimization**

Guide for tuning hyperparameters to optimize model performance.

**Key Topics:**
- Hyperparameter search strategies
- Grid search vs random search
- Bayesian optimization
- Tuning workflows
- Best practices

**Common Hyperparameters:**
- Learning rate
- Batch size
- Model architecture parameters
- Augmentation ratio
- Reduction ratio

---

### [Tuning Results](tuning-results.md)
**Initial Hyperparameter Tuning Results**

Documentation of initial hyperparameter tuning experiments and findings.

**Key Topics:**
- Baseline tuning results
- Optimal parameter ranges
- Performance sensitivity analysis
- Recommendations per dataset
- Lessons learned

---

## 🎯 Testing Workflows

### Unit Testing

```bash
# Run all unit tests
pytest tests/unit/

# Run specific test module
pytest tests/unit/test_augmentation.py

# Run with coverage
pytest --cov=src tests/unit/
```

### Integration Testing

```bash
# Run integration tests
pytest tests/integration/

# Test full pipeline
pytest tests/integration/test_pipeline.py
```

### Hyperparameter Tuning Workflow

```bash
# 1. Define search space
python tests/hyperparameter_tuning/define_search_space.py

# 2. Run tuning experiment
python tests/hyperparameter_tuning/run_tuning.py \
    --dataset D_W_15K_V1 \
    --model bert_int \
    --trials 50

# 3. Analyze results
python tests/hyperparameter_tuning/analyze_results.py

# 4. Export best configuration
python tests/hyperparameter_tuning/export_best_config.py
```

---

## 🔬 Hyperparameter Tuning Strategies

### Grid Search

**Best for:**
- Small search spaces
- Known parameter interactions
- Exhaustive exploration

**Example:**
```python
param_grid = {
    'learning_rate': [1e-5, 2e-5, 5e-5],
    'batch_size': [16, 32, 64],
    'augmentation_ratio': [0.5, 1.0, 2.0]
}
```

### Random Search

**Best for:**
- Large search spaces
- Unknown parameter importance
- Limited computational budget

**Example:**
```python
param_distributions = {
    'learning_rate': loguniform(1e-6, 1e-3),
    'batch_size': randint(16, 128),
    'augmentation_ratio': uniform(0.1, 3.0)
}
```

### Bayesian Optimization

**Best for:**
- Expensive evaluations
- Complex parameter spaces
- Iterative refinement

**Example:**
```python
from optuna import create_study

study = create_study(direction='maximize')
study.optimize(objective_function, n_trials=100)
```

---

## 📊 Tuning Best Practices

### 1. Start with Defaults

Always establish a baseline with default parameters before tuning.

### 2. Tune Incrementally

Tune one parameter group at a time:
1. Learning rate and optimizer settings
2. Model architecture parameters
3. Data augmentation parameters
4. Regularization parameters

### 3. Use Cross-Validation

Validate on multiple splits to ensure robustness.

### 4. Track Everything

Log all experiments with:
- Hyperparameter values
- Performance metrics
- Training time
- Resource usage

### 5. Budget Your Time

Set computational budgets:
- Maximum trials per experiment
- Time limit per trial
- Early stopping criteria

---

## 🎯 Recommended Parameter Ranges

### BERT-INT

| Parameter | Recommended Range | Default |
|-----------|------------------|---------|
| Learning Rate | 1e-5 to 5e-5 | 2e-5 |
| Batch Size | 16 to 64 | 32 |
| Max Length | 64 to 256 | 128 |
| Epochs | 5 to 20 | 10 |

### Augmentation

| Parameter | Recommended Range | Default |
|-----------|------------------|---------|
| Aug Ratio | 0.5 to 2.0 | 1.0 |
| BART Epochs | 1 to 5 | 3 |
| Temperature | 0.5 to 1.5 | 1.0 |

### Reduction

| Parameter | Recommended Range | Default |
|-----------|------------------|---------|
| Reduction Ratio | 0.1 to 0.7 | 0.3 |

---

## 🔍 Debugging Failed Experiments

### Common Issues

**Out of Memory:**
```bash
# Reduce batch size
# Enable gradient accumulation
# Use mixed precision training
```

**Poor Convergence:**
```bash
# Lower learning rate
# Increase warmup steps
# Check data quality
```

**Overfitting:**
```bash
# Add regularization
# Reduce model capacity
# Increase dropout
# Use early stopping
```

---

## 📚 Related Documentation

- **Models**: See [models/](../models/overview.md) for model-specific tuning
- **Configuration**: See [configuration/](../configuration/overview.md) for parameter setup
- **Experiments**: See [experiments/metrics.md](../experiments/metrics.md) for evaluation metrics

---

**Last Updated:** 2025-12-15
