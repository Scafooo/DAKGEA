# Initial Hyperparameter Testing Results

## Executive Summary

Initial testing of the PLM BART interpolation with hardcoded dataset examples reveals a **critical swapping issue**: the model is simply swapping inputs instead of creating proper interpolations.

**Test Date**: 2025-11-26
**Configuration**: `config/augmentation/plm.yaml` (default parameters)
**Tool Used**: `interactive_dataset_tests.py`

---

## Test Results

### BBC_DB Dataset (10 hardcoded examples)

| Metric | Count | Percentage | Status |
|--------|-------|------------|--------|
| **Swapping** | 9 | 90.0% | ❌ CRITICAL |
| **Copying** | 0 | 0.0% | ✅ |
| **Interpolation** | 1 | 10.0% | ❌ TOO LOW |
| **Other** | 0 | 0.0% | - |

**Quality Score**: -35.00 (target: > 30.00)

---

## Detailed Analysis

### Pattern Breakdown

**9 out of 10 examples showed simple swapping:**
- Input: `'braxtons'` / `'01t00 04 1989 00 01'`
- Output: `'01t00 04 1989 00 01'` / `'braxtons'` ❌

The model is not learning to interpolate, it's just reversing the input order.

**Only 1 example produced interpolation (Test #7):**
- Input: `'rock music'` / `'classical music'`
- Output: `'classical musicHouston radioTablesgoddard sculpturemodern style constructionTraditional music'` / `'rock musicHouston'`
- **Note**: This triggered 100 retry attempts and produced garbled output with random tokens

---

## Current Configuration

```yaml
generation:
  temperature: 0.85          # Too low - encourages deterministic swapping
  top_p: 0.9
  num_beams: 5               # Too high - reduces randomness
  repetition_penalty: 1.7
  noise_std: 0.001           # WAY TOO LOW - insufficient creativity

bart:
  base_alpha: 0.5
  alpha_spread: 0.45
```

---

## Root Cause Analysis

### Why is swapping happening?

1. **Low temperature (0.85)**: Makes the model too deterministic, favoring simple patterns like swapping
2. **Minimal noise (0.001)**: Not enough perturbation in hidden states to force interpolation
3. **High beam search (5)**: Beam search favors high-probability outputs, and swapping is a high-probability pattern
4. **Training bias**: The BART model may have learned that swapping is an acceptable transformation

### Why did retry mechanism fail?

The retry mechanism (test #7) shows that:
- It correctly detected high token overlap (50% for "music")
- It increased temperature up to 1.21 over 100 attempts
- But it only resulted in adding random tokens, not proper interpolation
- The constrained decoding (blocking "music") forced the model to hallucinate

---

## Encoding Issues

✅ **No encoding issues detected** in the hardcoded test examples:
- No escape sequences (`\u`, `\x`)
- No non-ASCII character problems
- No replacement characters (`�`, `\ufffd`)

The encoding detection function is working correctly.

---

## Recommendations

### Immediate Actions

1. **Increase `noise_std`**: 0.001 → 0.02-0.05
   - This will force more randomness in hidden state interpolation

2. **Increase `temperature`**: 0.85 → 1.0-1.2
   - Higher temperature reduces deterministic behavior

3. **Decrease `num_beams`**: 5 → 3
   - Lower beams = more randomness, less bias toward high-probability swaps

4. **Test with `high_creativity` config**:
   ```yaml
   temperature: 1.1
   top_p: 0.92
   num_beams: 3
   noise_std: 0.01
   ```

### Investigation Needed

1. **BART model training quality**
   - Check if fine-tuning data contains swap patterns
   - Verify that training examples show proper interpolation

2. **Alpha interpolation**
   - Verify that `base_alpha` and `alpha_spread` are being applied correctly
   - Check if hidden state interpolation is actually happening

3. **Non-matching attributes** (user concern)
   - User noted: "provi a interpolare valori derivati da attributi che non matchano"
   - Need to investigate why values from different attributes are being interpolated together

---

## Next Steps

1. Use `interactive_dataset_tests.py` to test alternative parameter configurations
2. Run `test_parameter_configs.py --configs all` for systematic comparison
3. If parameter tuning doesn't help, investigate BART model training
4. Consider retraining BART with better interpolation examples

---

## Usage

To reproduce these results:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/hyperparameter_tuning/interactive_dataset_tests.py \
  --config config/augmentation/plm.yaml
# Then: Menu option 2 → Dataset 1 (BBC_DB) → Menu option 5 (Quit)
```

To test alternative configurations:

```bash
# Option 1: Interactive adjustment
CUDA_VISIBLE_DEVICES=0 python tests/hyperparameter_tuning/interactive_dataset_tests.py

# Option 2: Systematic comparison
CUDA_VISIBLE_DEVICES=0 python tests/hyperparameter_tuning/test_parameter_configs.py \
  --configs all
```
