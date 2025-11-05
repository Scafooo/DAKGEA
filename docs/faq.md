# DAKGEA Frequently Asked Questions (FAQ)

Common questions and answers about using DAKGEA.

---

## General Questions

### What is DAKGEA?

DAKGEA (Data Augmentation for Knowledge Graph Entity Alignment) is a modular experimentation framework for entity alignment on Knowledge Graphs. It combines dataset reduction, data augmentation, and model training to measure how each component impacts alignment accuracy.

### What models are supported?

Currently supported:
- **BERT-INT**: Two-phase BERT-based alignment model
- **HybEA**: Hybrid entity alignment model

The framework is designed to be extensible - you can add custom models through the registry system.

### What datasets can I use?

DAKGEA supports multiple formats:
- **HybEA format**: Original format from HybEA project
- **BERT-INT format**: Native BERT-INT format
- **RDF format**: Standard RDF/Turtle format

See the [Dataset Guide](dataset-guide.md) for details.

---

## Installation & Setup

### How do I install DAKGEA?

```bash
git clone https://github.com/Scafooo/DataAug-KG-EntityResolution
cd DAKGEA
conda env create -f install/HybEA_env.yml
conda activate hybea
```

See [User Guide - Installation](user-guide.md#1-install--setup) for details.

### Do I need a GPU?

**No**, but it's strongly recommended. BERT-INT can run on CPU but will be 10-50x slower.

For CPU-only:
```yaml
parameters:
  models:
    bert_int:
      device: "cpu"
      basic_unit:
        batch_size: 32  # Smaller batches for CPU
```

### What Python version is required?

Python 3.8 or higher. Tested on Python 3.8, 3.9, and 3.10.

---

## Configuration

### How do I specify which dataset to use?

Use the `dataset.name` field with `reader/dataset` format:

```yaml
dataset:
  name: "hybea/BBC_DB"  # Uses HybEA reader for BBC_DB dataset
```

See [Configuration Guide - Dataset Configuration](configuration-guide.md#dataset-configuration) for all options.

### What's the difference between reduction and augmentation?

- **Reduction**: Keeps only a fraction of training data (e.g., 0.1 = 10%)
- **Augmentation**: Generates additional synthetic training data

Example:
```yaml
augmentation:
  method: "stub"        # No augmentation
  reduction: 0.1        # Use only 10% of training data
```

### How do I run an experiment?

```bash
./run.sh config/experiments/my_config.yaml
```

Or:
```bash
python experiments/run.py config/experiments/my_config.yaml
```

### Can I run multiple experiments in one config?

Yes, use arrays for reduction ratios or models:

**Legacy format:**
```yaml
reduction_ratios: [0.1, 0.2, 0.5]
models_to_run: ["bert_int", "hybea"]
```

This runs all combinations (3 ratios × 2 models = 6 experiments).

### Where are results saved?

```
results/<experiment_name>/<dataset>/<ratio>/
├── reduction/artefacts/          # Reduced dataset
├── augmentation/<method>/         # Augmented dataset
└── evaluation/<variant>/          # Model results
    └── <model>.json              # Metrics
```

---

## BERT-INT Specific

### How long does BERT-INT training take?

Typical times (on RTX 4070):
- **Phase 1** (Basic Unit): 5-15 minutes
- **Phase 2** (Interaction Model): 10-30 minutes
- **Total**: 15-45 minutes

Time varies by dataset size and configuration.

### Why are my metrics showing as decimals (0.34) instead of percentages (34%)?

**This is correct.** DAKGEA uses fractions (0-1 range) for all metrics internally and in JSON output.

```json
{
  "hits@1": 0.34,    // ✓ Correct: 34% hit rate
  "hits@10": 0.78    // ✓ Correct: 78% hit rate
}
```

Old versions had a bug showing percentages incorrectly.

### Can I use pre-trained BERT-INT checkpoints?

Yes! Use direct path mode:

```yaml
dataset:
  path: "/path/to/checkpoint/directory"
model: bert_int
skip_training: true
```

The directory must contain:
- `run_*.pth` files (Basic Unit checkpoints)
- `interaction_model.pt` (Interaction Model checkpoint)

### How do I compare with the reference implementation?

1. Use the same dataset and parameters:
```yaml
dataset:
  path: "/path/to/reference/data"
model: bert_int
seed: 11037  # Same seed
```

2. Run both implementations

3. Compare results - expect <3% difference due to:
   - PyTorch version differences
   - Hardware variations
   - CUDA randomness

### Attribute triples aren't being used - why?

**Make sure:**
1. Your dataset has attribute triples (check `attr_triples1`, `attr_triples2` files)
2. You're using the `bert_int` writer:
   ```yaml
   dataset:
     writer: "bert_int"
   ```
3. You're on the latest version (old versions had a bug skipping attributes)

Check logs for:
```
BERT-INT KG 1: wrote 10543 attribute triples
```

---

## Errors & Troubleshooting

### "Unable to infer reader for dataset 'BBC_DB'"

**Cause:** Dataset not found in expected location.

**Solution:** Use explicit `reader/dataset` format:
```yaml
dataset:
  name: "hybea/BBC_DB"  # Specifies reader explicitly
```

### "Direct path mode requires 'path' field"

**Cause:** System entered direct path mode but no path provided.

**Why:** No reduction ratio specified, so system assumes direct mode.

**Solution:** Add reduction ratio:
```yaml
augmentation:
  method: "stub"
  reduction: 0.1  # Required for standard mode
```

### "dataset_workspace not found in lineage"

**Cause:** BERT-INT model requires data in BERT-INT format, but writer not specified.

**Error message:**
```
ValueError: dataset_workspace not found in lineage.
Make sure to use writer: bert_int in experiment config
```

**Solution:** Add `writer: bert_int` to dataset configuration:
```yaml
dataset:
  name: "hybea/BBC_DB"
  writer: bert_int  # Required for BERT-INT model
```

This ensures the data is converted to BERT-INT format before training.

### "CUDA out of memory"

**Solutions:**

1. Reduce batch size:
   ```yaml
   parameters:
     models:
       bert_int:
         basic_unit:
           batch_size: 128  # Reduce from 256
         interaction_model:
           batch_size: 64   # Reduce from 128
   ```

2. Use CPU mode (slower but works):
   ```yaml
   parameters:
     models:
       bert_int:
         device: "cpu"
   ```

3. Use a machine with more GPU memory

### "FileNotFoundError: config/models/bert_int.yaml"

**Cause:** Running from wrong directory.

**Solution:** Always run from project root:
```bash
cd /path/to/DAKGEA
./run.sh config/experiments/my_config.yaml
```

### Results show 0.0 for all metrics

**Possible causes:**

1. **Training failed** - check logs for errors
2. **Empty test set** - verify dataset has test pairs
3. **Wrong dataset path** - check paths in config

**Debug:**
```bash
# Check logs
tail -100 results/<experiment>/log.txt

# Verify dataset
python -c "
from src.core.dataset.reader import DatasetReaderFactory
reader = DatasetReaderFactory.create('bert_int')
ds = reader.read('path/to/dataset')
print(f'Test pairs: {len(ds.test_alignment)}')
"
```

### Experiments are too slow

**Speed up:**

1. **Use GPU** instead of CPU (10-50x faster)

2. **Reduce epochs**:
   ```yaml
   parameters:
     models:
       bert_int:
         basic_unit:
           epochs: 10  # Reduce from 20
   ```

3. **Increase batch size** (if you have GPU memory):
   ```yaml
   basic_unit:
     batch_size: 512  # Increase from 256
   ```

4. **Use smaller datasets** for testing

---

## Data & Formats

### How do I convert HybEA format to BERT-INT format?

The system does this automatically! Just specify the writer:

```yaml
dataset:
  name: "hybea/BBC_DB"
  writers:
    - type: "bert_int"  # Writes in BERT-INT format
```

Output will be in:
```
results/<experiment>/<dataset>/<ratio>/reduction/artefacts/bert_int/
```

### Can I use my own dataset?

Yes! Two options:

**Option 1: Use existing format**

Format your data as HybEA or BERT-INT, place in `data/raw/`:
```
data/raw/hybea/MY_DATASET/
  ├── ent_ids_1
  ├── ent_ids_2
  # ... etc
```

**Option 2: Implement custom reader**

See [Dataset Guide - Adding Custom Datasets](dataset-guide.md#adding-custom-datasets).

### What's the entity ID offset in BERT-INT format?

Target entity IDs are offset by the number of source entities.

**Example:**
```
Source: 0, 1, 2, 3, 4       (5 entities)
Target: 5, 6, 7, 8, 9       (5 entities, offset by +5)

Alignment:
0 → 5   (source entity 0 aligns with target entity 5)
1 → 7   (source entity 1 aligns with target entity 7)
```

The BERT-INT writer handles this automatically.

### How are attributes represented?

Attributes are triples with literal values:

```
entity_id    predicate    literal_value
0            name         "London"
0            population   "8900000"
1            name         "Paris"
```

File: `attr_triples1` or `attr_triples2`

---

## Advanced Usage

### Can I skip certain pipeline stages?

Yes:

**Skip reduction (use full dataset):**
```yaml
augmentation:
  method: "stub"
  reduction: 1.0  # Use 100% of data
```

**Skip augmentation:**
```yaml
augmentation:
  method: "stub"  # Stub = no augmentation
```

**Skip training (re-evaluate existing model):**
```yaml
skip_training: true
```

### How do I override model parameters?

Use the `parameters` section:

```yaml
parameters:
  models:
    bert_int:
      basic_unit:
        epochs: 30           # Override default
        learning_rate: 3e-5  # Override default
```

### Can I use different seeds for different stages?

Yes:

```yaml
seed: 42  # Global seed

parameters:
  experiment:
    seed: 42  # Experiment-level

  reduction:
    random_seed: 123  # Reduction-specific

  models:
    bert_int:
      basic_unit:
        seed: 456  # Model-specific
```

Lower-level seeds override higher-level ones.

### How do I add a custom augmentation method?

1. Implement the augmenter:
   ```python
   from src.augmentation.base import Augmenter
   from src.augmentation.registry import AUGMENTATION_REGISTRY

   @AUGMENTATION_REGISTRY.register("my_method")
   class MyAugmenter(Augmenter):
       def augment(self, dataset):
           # Your augmentation logic
           return augmented_dataset
   ```

2. Use in config:
   ```yaml
   augmentation:
     method: "my_method"
   ```

See [Developer Guide](developer-guide.md) for details.

---

## Results & Metrics

### What metrics does BERT-INT report?

- **Hits@K**: Fraction of correct alignments in top-K predictions
  - Hits@1, Hits@5, Hits@10, Hits@25, Hits@50
- **MRR**: Mean Reciprocal Rank
- **MR**: Mean Rank

All metrics are fractions in [0, 1] range.

### How do I interpret the results JSON?

```json
{
  "model": "bert_int",
  "phases": {
    "basic_unit": {
      "hits@1": 0.34,     // Phase 1 results
      "hits@10": 0.75,
      "mrr": 0.52
    },
    "interaction_model": {
      "hits@1": 0.45,     // Phase 2 results (final)
      "hits@10": 0.83,
      "mrr": 0.61
    }
  },
  "hits@1": 0.45,         // Top-level = Phase 2
  "hits@10": 0.83,
  "mrr": 0.61,
  "evaluated": 1500       // Number of test pairs
}
```

**Key points:**
- Top-level metrics are from Phase 2 (final results)
- Both phase results are preserved under `phases`
- All values are fractions (0-1), not percentages

### Where are model checkpoints saved?

```
results/<experiment>/evaluation/bert_int/<variant>/
├── run_1.pth              # Basic Unit checkpoint (epoch 1)
├── run_20.pth             # Basic Unit final
└── interaction_model.pt   # Interaction Model final
```

### Can I visualize the results?

Yes! Results are in JSON format, easy to load and plot:

```python
import json
import matplotlib.pyplot as plt

# Load results
with open('results/my_exp/BBC_DB/0.1/evaluation/reduced/bert_int.json') as f:
    results = json.load(f)

# Plot
metrics = ['hits@1', 'hits@5', 'hits@10']
values = [results[m] for m in metrics]

plt.bar(metrics, values)
plt.ylabel('Hit Rate')
plt.title('BERT-INT Performance')
plt.show()
```

---

## Best Practices

### What are the recommended settings for production?

```yaml
experiment:
  name: "production_<date>"
  dataset:
    name: "hybea/MY_DATASET"
    writer: "bert_int"
  augmentation:
    method: "plm_augmentation"  # Or your best method
    reduction: 0.2              # Adjust based on dataset size
  model: bert_int
  seed: 42                      # Fixed seed for reproducibility
  clear: true                   # Clean up intermediate files

parameters:
  models:
    bert_int:
      basic_unit:
        epochs: 20              # Default is usually good
        batch_size: 256         # Adjust for your GPU
      interaction_model:
        epochs: 100             # Increase for better accuracy
        candidate_topk: 50      # Default is usually good
```

### How should I organize experiments?

**Naming convention:**
```
<YYYYMMDD>_<description>_<key_params>
```

**Examples:**
```
20250105_bert_int_bbc_reduction_sweep
20250106_plm_augmentation_test_v2
20250107_multi_model_comparison
```

**Directory structure:**
```
config/experiments/
├── production/
│   ├── 20250105_final_run.yaml
│   └── 20250106_comparison.yaml
├── development/
│   ├── test_new_feature.yaml
│   └── debug_config.yaml
└── archive/
    └── old_experiments/
```

### Should I version control results?

**Don't commit:**
- Large result files
- Model checkpoints (*.pth, *.pt)
- Intermediate datasets

**Do commit:**
- Configuration files
- Result summaries (JSON)
- Analysis scripts
- Documentation

**Use .gitignore:**
```
results/
data/reduced/
data/augmented/
*.pth
*.pt
*.log
```

---

## Getting Help

### Where can I find more documentation?

- [User Guide](user-guide.md) - Basic usage
- [Configuration Guide](configuration-guide.md) - All config options
- [Dataset Guide](dataset-guide.md) - Dataset formats
- [BERT-INT Guide](bert-int-guide.md) - BERT-INT specifics
- [Developer Guide](developer-guide.md) - Extending the framework

### How do I report a bug?

1. Check if it's a known issue (this FAQ)
2. Search existing GitHub issues
3. Create a new issue with:
   - Configuration file
   - Error message
   - Log files
   - Steps to reproduce

### How do I contribute?

See [Developer Guide](developer-guide.md) for:
- Code structure
- Coding conventions
- Testing guidelines
- Pull request process

---

## See Also

- [GitHub Repository](https://github.com/Scafooo/DataAug-KG-EntityResolution)
- [HybEA Project](https://github.com/fanourakis/HybEA)
- [BERT-INT Paper](https://arxiv.org/abs/2104.08095)
