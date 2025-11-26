# Hyperparameter Tuning Guide

This guide explains how to find the optimal hyperparameters for PLM augmentation.

## Overview

We provide three approaches for hyperparameter tuning:

1. **Quick Test** - Fast testing of specific parameter combinations
2. **Phased Tuning** - Systematic optimization in sequential phases
3. **Grid/Random Search** - Exhaustive search over parameter space

## Parameters to Optimize

### Core Interpolation Parameters

- **`base_alpha`** (0.0-1.0): Base interpolation strength
  - Lower = more conservative (closer to inputs)
  - Higher = more creative mixing
  - Recommended range: 0.3-0.7

- **`alpha_spread`** (0.0-1.0): Variation range around base_alpha
  - Controls adaptivity to similarity
  - Recommended range: 0.2-0.7

### Generation Parameters

- **`temperature`** (0.1-2.0): Sampling randomness
  - Lower = more deterministic
  - Higher = more creative/diverse
  - Recommended range: 0.7-1.4

- **`top_p`** (0.0-1.0): Nucleus sampling threshold
  - Recommended range: 0.85-0.98

- **`top_k`** (0 or 10-100): Top-k sampling
  - 0 = disabled (use top_p only)
  - Recommended: 0, 30, 50

- **`num_beams`** (1-10): Beam search width
  - 1 = greedy/sampling only
  - Higher = better quality but slower
  - Recommended range: 1-7

- **`repetition_penalty`** (1.0-3.0): Penalize token repetition
  - 1.0 = no penalty
  - Higher = less repetition
  - Recommended range: 1.0-2.0

- **`no_repeat_ngram_size`** (0-5): Block n-gram repetition
  - 0 = disabled
  - Recommended: 2-4

### Noise Injection Parameters

- **`noise_std`** (0.0-0.2): Noise level for creativity
  - 0.0 = no noise
  - Higher = more variation
  - Recommended range: 0.0-0.1

### Retry Mechanism Parameters

- **`temperature_increment`** (0.0-0.1): Temperature increase per retry
  - 0.0 = fixed temperature
  - Recommended range: 0.0-0.05

- **`identical_tokens_threshold`** (0.0-1.0): Retry trigger threshold
  - Fraction of identical tokens to trigger retry
  - Recommended range: 0.2-0.5

### Sentence-Level Parameters

- **`sentence_chunk_max_tokens`** (40-100): Max tokens per chunk
  - Must be < max_len_in (96)
  - Recommended range: 60-90

- **`sentence_min_length_for_chunking`** (30-80): Min length for chunking
  - Texts shorter than this use standard interpolation
  - Recommended range: 40-70

## Usage

### 1. Quick Test (Fastest - Single Configuration)

Test a specific parameter combination:

```bash
# Test with specific parameters
python experiments/quick_test.py \\
    --dataset BBC_DB \\
    --ratio 0.1 \\
    --temperature 1.0 \\
    --alpha 0.5 \\
    --top-p 0.9 \\
    --beams 5

# Test with custom config
python experiments/quick_test.py \\
    --dataset BBC_DB \\
    --ratio 0.1 \\
    --config my_config.yaml
```

**Time:** ~5-10 minutes per test
**Use when:** Testing specific hypotheses or comparing 2-3 configurations

### 2. Phased Tuning (Recommended - Systematic)

Optimize parameters in sequential phases:

```bash
# Fast mode (smaller search spaces)
python experiments/phased_tuning.py \\
    --dataset BBC_DB \\
    --ratio 0.1 \\
    --fast

# Full mode (comprehensive)
python experiments/phased_tuning.py \\
    --dataset BBC_DB \\
    --ratio 0.1
```

**Phases:**
1. Alpha & Temperature (core interpolation)
2. Sampling parameters (top_p, top_k)
3. Beam search (num_beams, n-gram blocking)
4. Penalties (repetition)
5. Noise injection
6. Retry mechanism
7. Sentence-level parameters

**Time:**
- Fast mode: ~2-4 hours
- Full mode: ~8-12 hours

**Use when:** Finding optimal configuration for production use

### 3. Grid/Random Search (Exhaustive)

Search over full parameter space:

```bash
# Random search (recommended)
python experiments/hyperparameter_tuning.py \\
    --config config/augmentation/plm.yaml \\
    --dataset BBC_DB \\
    --ratio 0.1 \\
    --search-type random \\
    --max-trials 100

# Grid search (all combinations)
python experiments/hyperparameter_tuning.py \\
    --config config/augmentation/plm.yaml \\
    --dataset BBC_DB \\
    --ratio 0.1 \\
    --search-type grid
```

**Time:**
- Random (100 trials): ~12-20 hours
- Grid (all combinations): Days/weeks

**Use when:** Maximum thoroughness needed, ample compute available

## Interpreting Results

### Output Files

All tuning runs produce:

- `tuning_results.json` - Complete results with all trials
- `best_config.yaml` - Best configuration found
- `phase*/trial*/` - Individual trial workspaces

### Key Metrics

Focus on:
- **hits@1** - Primary metric (alignment accuracy)
- **mrr** - Mean reciprocal rank
- **f-measure** - Harmonic mean of precision/recall

### Typical Improvements

Expected improvement over baseline:
- Good tuning: +5-10% hits@1
- Excellent tuning: +10-20% hits@1
- Domain-specific: Can vary significantly

## Recommended Workflow

### Step 1: Baseline

Run with default config:

```bash
python experiments/quick_test.py --dataset BBC_DB --ratio 0.1
```

Note baseline hits@1 score.

### Step 2: Fast Phased Tuning

```bash
python experiments/phased_tuning.py --dataset BBC_DB --ratio 0.1 --fast
```

This gives you a good starting point in ~2-4 hours.

### Step 3: Fine-Tune Critical Parameters

Based on results, do quick tests around best values:

```bash
# If best temperature was 0.85, try nearby values
python experiments/quick_test.py --dataset BBC_DB --ratio 0.1 --temperature 0.80
python experiments/quick_test.py --dataset BBC_DB --ratio 0.1 --temperature 0.90
```

### Step 4: Full Phased Tuning (Optional)

For production deployment:

```bash
python experiments/phased_tuning.py --dataset BBC_DB --ratio 0.1
```

### Step 5: Validation

Test best config on multiple ratios:

```bash
python experiments/quick_test.py --dataset BBC_DB --ratio 0.05 --config best_config.yaml
python experiments/quick_test.py --dataset BBC_DB --ratio 0.1 --config best_config.yaml
python experiments/quick_test.py --dataset BBC_DB --ratio 0.2 --config best_config.yaml
```

## Tips for Effective Tuning

### 1. Start with Core Parameters

Focus on alpha and temperature first - they have the biggest impact.

### 2. One Phase at a Time

Don't optimize everything simultaneously. Sequential phases are more efficient.

### 3. Monitor Overfitting

If validation scores decrease while tuning, you're overfitting to the specific dataset.

### 4. Dataset-Specific Tuning

Different datasets may need different parameters:
- **Structured data** (DBpedia): Lower temperature, higher alpha
- **Text-heavy data** (Bio): Higher temperature, sentence-level enabled

### 5. Trade-offs

- **Speed vs Quality**: Higher beams = better quality but slower
- **Creativity vs Accuracy**: Higher temperature = more diverse but less accurate
- **Consistency vs Diversity**: Enable retry = more consistent but slower

## Common Issues

### Issue: No Improvement

**Possible causes:**
- Search space too narrow
- Metric not sensitive to parameters
- Dataset too small/noisy

**Solutions:**
- Widen search ranges
- Try different metric (mrr, f-measure)
- Increase dataset size

### Issue: Unstable Results

**Possible causes:**
- High variance in generation
- Small test set
- Random seed effects

**Solutions:**
- Run multiple seeds and average
- Increase max_retries
- Lower temperature/noise

### Issue: Slow Tuning

**Possible causes:**
- Too many combinations
- Large dataset
- High max_retries

**Solutions:**
- Use fast mode
- Reduce dataset with --max-pairs
- Lower max_retries temporarily

## Parameter Interaction Guide

Some parameters interact:

### Temperature + Top_p

- High temperature + Low top_p: Diverse but controlled
- Low temperature + High top_p: Deterministic exploration
- High both: Maximum creativity (may be incoherent)
- Low both: Very conservative

### Beams + Temperature

- High beams + Low temperature: Best for quality
- Low beams + High temperature: Best for speed/diversity
- High both: Expensive but diverse quality

### Noise + Retry

- High noise + High threshold: Aggressive variation
- Low noise + Low threshold: Conservative with consistency
- High noise + Low threshold: May trigger unnecessary retries

## Example Configurations

### Conservative (High Precision)

```yaml
base_alpha: 0.4
alpha_spread: 0.3
temperature: 0.7
top_p: 0.85
num_beams: 7
repetition_penalty: 2.0
noise_std: 0.001
```

### Balanced (Recommended Starting Point)

```yaml
base_alpha: 0.5
alpha_spread: 0.45
temperature: 0.85
top_p: 0.9
num_beams: 5
repetition_penalty: 1.7
noise_std: 0.01
```

### Creative (High Diversity)

```yaml
base_alpha: 0.6
alpha_spread: 0.55
temperature: 1.2
top_p: 0.95
num_beams: 3
repetition_penalty: 1.5
noise_std: 0.05
```

## Next Steps

After finding optimal parameters:

1. **Update default config** - Save best parameters to `config/augmentation/plm.yaml`
2. **Test on other datasets** - Verify generalization
3. **Document results** - Record improvements and insights
4. **Monitor production** - Track metrics over time

## Support

For issues or questions:
- Check logs in `results/tuning/`
- Review failed trials
- Consult PLM_AUGMENTATION.md for implementation details
