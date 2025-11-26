# Hyperparameter Tuning Tools

This directory contains tools for analyzing and optimizing PLM (Pre-trained Language Model) hyperparameters for the BART interpolation augmentation method.

## Scripts Overview

### 1. `analyze_current_parameters.py`
Analyzes the current configuration to identify transformation quality issues.

**Usage:**
```bash
CUDA_VISIBLE_DEVICES=0 python tests/hyperparameter_tuning/analyze_current_parameters.py \
  --config config/augmentation/plm.yaml \
  --tests 10
```

**Output:**
- Pattern analysis (swap/copy/interpolation rates)
- Specific recommendations for parameter adjustments
- Quality metrics for current configuration

**When to use:**
- Before starting hyperparameter tuning
- To diagnose transformation quality issues
- After changing parameters to verify improvements

---

### 2. `test_parameter_configs.py`
Compares multiple parameter configurations side-by-side.

**Usage:**
```bash
# Test specific configs
CUDA_VISIBLE_DEVICES=0 python tests/hyperparameter_tuning/test_parameter_configs.py \
  --configs current high_creativity balanced

# Test all configs
CUDA_VISIBLE_DEVICES=0 python tests/hyperparameter_tuning/test_parameter_configs.py \
  --configs all
```

**Pre-defined configurations:**
- `current` - Current configuration from plm.yaml
- `high_creativity` - Higher temperature + noise for more diversity
- `low_beams_high_temp` - Fewer beams + high temperature
- `high_noise` - Emphasizes noise injection
- `balanced` - Moderate settings across all parameters

**Output:**
- Comparison table showing swap/copy/interpolation rates
- Quality scores for each configuration
- Recommended best configuration

**When to use:**
- To find optimal parameter combinations
- When current config produces too much swapping
- For A/B testing different approaches

---

### 3. `visualize_augmentation_by_dataset.py`
Shows detailed transformation examples for each dataset separately.

**Usage:**
```bash
CUDA_VISIBLE_DEVICES=0 python tests/hyperparameter_tuning/visualize_augmentation_by_dataset.py \
  --datasets BBC_DB D_W_15K_V1 ICEW_WIKI ICEW_YAGO \
  --examples-per-dataset 3
```

**Output:**
- Per-dataset transformation examples
- Analysis of each transformation (changed/swapped/identical)
- Summary statistics per dataset

**When to use:**
- To see how parameters perform on real dataset values
- To identify dataset-specific issues
- Before running full experiments

---

### 4. `show_dataset_transformations.py`
Simplified version showing transformations with current config.

**Usage:**
```bash
CUDA_VISIBLE_DEVICES=0 python tests/hyperparameter_tuning/show_dataset_transformations.py \
  --datasets BBC_DB \
  --examples-per-dataset 5
```

**When to use:**
- Quick sanity check of transformations
- During interactive parameter adjustment

---

### 5. `hyperparameter_tuning.py`
Comprehensive grid search over parameter space (legacy script).

**Note:** This is a more comprehensive tool that runs many configurations systematically.

---

### 6. `phased_tuning.py`
Multi-phase hyperparameter optimization strategy.

**Phases:**
1. Coarse search over wide parameter ranges
2. Fine-tuning around best configurations
3. Final optimization with micro-adjustments

---

### 7. `interactive_tuning.py`
Interactive interface for manual parameter exploration.

**Features:**
- Live transformation visualization
- Manual parameter adjustment
- Save/load configurations
- Real-time feedback

**Usage:**
```bash
python tests/hyperparameter_tuning/interactive_tuning.py \
  --dataset BBC_DB \
  --ratio 0.1
```

---

## Common Issues and Solutions

### Issue: High Swapping Rate (>50%)

**Symptoms:**
- Outputs are just swapped inputs (input1 → output2, input2 → output1)
- No real interpolation happening

**Solutions:**
1. Increase `temperature` (0.85 → 1.0-1.2)
2. Increase `noise_std` (0.001 → 0.01-0.05)
3. Decrease `num_beams` (5 → 3)
4. Check if BART model is properly fine-tuned

### Issue: High Copying Rate (>30%)

**Symptoms:**
- Outputs are identical to inputs (no transformation)

**Solutions:**
1. Enable retry mechanism (`enable_retry_on_identical_tokens: true`)
2. Lower `identical_tokens_threshold` (0.3 → 0.2)
3. Increase `noise_std`
4. Verify generation parameters are being applied

### Issue: Low Interpolation Rate (<30%)

**Symptoms:**
- Few proper interpolations produced

**Solutions:**
1. Check BART model training quality
2. Verify training data diversity
3. Try different `alpha_spread` values
4. Consider retraining with better data

---

## Parameter Guide

### Generation Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `temperature` | 0.85 | 0.7-1.2 | Randomness (higher = more creative) |
| `top_p` | 0.9 | 0.85-0.95 | Nucleus sampling (higher = more diversity) |
| `num_beams` | 5 | 3-7 | Beam search (lower = more random) |
| `repetition_penalty` | 1.7 | 1.3-2.0 | Penalize repetition |
| `noise_std` | 0.001 | 0.0-0.1 | Hidden state noise |

### Interpolation Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `base_alpha` | 0.5 | 0.3-0.7 | Interpolation balance |
| `alpha_spread` | 0.45 | 0.2-0.5 | Alpha variation range |

### Retry Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `enable_retry_on_identical_tokens` | true | - | Enable/disable retry |
| `max_retries` | 100 | 10-200 | Max retry attempts |
| `identical_tokens_threshold` | 0.3 | 0.1-0.5 | When to trigger retry |
| `temperature_increment` | 0.02 | 0.01-0.05 | Temp increase per retry |

---

## Workflow

### 1. Initial Analysis
```bash
# Analyze current parameters
CUDA_VISIBLE_DEVICES=0 python tests/hyperparameter_tuning/analyze_current_parameters.py
```

### 2. Test Alternatives
```bash
# Compare different configurations
CUDA_VISIBLE_DEVICES=0 python tests/hyperparameter_tuning/test_parameter_configs.py --configs all
```

### 3. Verify on Real Data
```bash
# Test on actual dataset values
CUDA_VISIBLE_DEVICES=0 python tests/hyperparameter_tuning/visualize_augmentation_by_dataset.py \
  --datasets BBC_DB --examples-per-dataset 5
```

### 4. Apply Best Configuration
Edit `config/augmentation/plm.yaml` with the best parameters found.

### 5. Run Full Experiment
Test with complete augmentation pipeline to verify improvements.

---

## Tips

1. **Start conservative**: Small parameter changes can have big effects
2. **Test incrementally**: Change one parameter at a time when possible
3. **Use real data**: Synthetic test pairs may not reflect actual performance
4. **Monitor diversity**: High interpolation rate is good, but check output quality too
5. **Document results**: Keep notes on what works and what doesn't

---

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size
- Use CPU (`device: cpu` in config)
- Process fewer examples at once

### Slow Performance
- Use `--tests` to limit number of test pairs
- Enable caching where possible
- Use smaller test datasets

### Inconsistent Results
- Set `PYTHONHASHSEED=42` for reproducibility
- Check if deterministic mode is enabled
- Verify seed is being set correctly
