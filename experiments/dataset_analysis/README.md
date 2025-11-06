# Dataset Analysis Tool

Comprehensive analysis tool for HybEA `attribute_data` format datasets to verify structural invariants and understand data characteristics.

## Features

- **Alignment Structure Analysis**: Verify splits (ref/sup/valid) and entity coverage
- **Triples Coverage Verification**: Check if all aligned entities appear in triples
- **Attribute Coverage Analysis**: Identify entities with/without attributes
- **Data Density Statistics**: Compute average attributes/relations per entity
- **Invariant Verification**: Automatically check dataset structural invariants

## Quick Start

```bash
# Analyze a dataset
./run_analysis.sh data/raw/hybea/BBC_DB/attribute_data

# Save results to JSON
./run_analysis.sh data/raw/hybea/BBC_DB/attribute_data -o results.json

# Verbose mode
./run_analysis.sh data/raw/hybea/BBC_DB/attribute_data --verbose
```

## Usage

### Command Line

```bash
./run_analysis.sh [OPTIONS] DATASET_PATH

Options:
  -o, --output FILE     Save results to JSON file
  -v, --verbose         Enable verbose logging
  -q, --quiet           Suppress all output except errors
  -h, --help            Show help message
```

### Python API

```python
from experiments.dataset_analysis import DatasetAnalyzer

# Initialize analyzer
analyzer = DatasetAnalyzer("data/raw/hybea/BBC_DB/attribute_data")

# Run full analysis
results = analyzer.run_full_analysis()

# Or run specific analyses
alignment_stats = analyzer.analyze_alignment_structure()
coverage = analyzer.verify_alignment_to_triples()
attributes = analyzer.analyze_attribute_coverage()
density = analyzer.analyze_data_density()
```

## Output

### Console Output

The analyzer provides detailed console output with:
- Entity and triple counts
- Alignment statistics (ref/sup/valid distribution)
- Coverage percentages
- Invariant verification results (✅/❌)
- Data density metrics per alignment category

### JSON Output

When using `-o/--output`, results are saved in JSON format:

```json
{
  "dataset_path": "data/raw/hybea/BBC_DB/attribute_data",
  "alignment_structure": {
    "total_unique_pairs": 9396,
    "ref_pairs": 6578,
    "sup_pairs": 1879,
    "valid_pairs": 939,
    "overlap_all_three": 0,
    "unique_entities_kg1": 9396,
    "unique_entities_kg2": 9396,
    "kg1_index_range": [0, 9395],
    "kg2_index_range": [9396, 18791]
  },
  "alignment_to_triples": {
    "aligned_kg1": 9396,
    "in_triples_kg1": 9396,
    "missing_in_triples_kg1": 0,
    "not_aligned_in_triples_kg1": 0
  },
  "attribute_coverage": {
    "total_entities_kg1": 9396,
    "only_triples_kg1": 38,
    "only_attr_kg1": 0,
    "in_both_kg1": 9358
  },
  "data_density": {
    "ref": {
      "kg1_avg_attrs": 1.93,
      "kg2_avg_attrs": 15.98,
      "kg1_avg_rels": 1.66,
      "kg2_avg_rels": 4.90
    },
    "sup": { ... },
    "valid": { ... }
  }
}
```

## Verified Invariants

The tool verifies these structural invariants:

1. ✅ **No overlap between alignment splits**: `ref ∩ sup ∩ valid = ∅`
2. ✅ **Complete entity coverage**: All entities in `ent_ids` are aligned
3. ✅ **Alignment → Triples**: All aligned entities appear in triples
4. ✅ **Triples → Alignment**: All entities in triples are aligned
5. ✅ **Separate index spaces**: KG1 and KG2 use non-overlapping indices
6. ⚠️ **Attributes optional**: Entities may not have attributes (especially in KG2)

## Batch Analysis

Analyze multiple datasets:

```bash
# Analyze all HybEA datasets
for dataset in data/raw/hybea/*/attribute_data; do
    dataset_name=$(basename $(dirname "$dataset"))
    ./run_analysis.sh "$dataset" -o "results/analysis_${dataset_name}.json"
done
```

## Integration with Experiments

The analyzer can be used to validate datasets before running experiments:

```bash
# Validate dataset
./run_analysis.sh data/raw/hybea/BBC_DB/attribute_data

# If validation passes, run experiment
./run.sh config/experiments/my_experiment.yaml
```

## File Format

Expected directory structure for `attribute_data`:

```
attribute_data/
├── ent_ids_1          # Entity ID mapping KG1
├── ent_ids_2          # Entity ID mapping KG2
├── triples_1          # Relation triples KG1 (indices)
├── triples_2          # Relation triples KG2 (indices)
├── attr_triples1      # Attribute triples KG1 (URIs)
├── attr_triples2      # Attribute triples KG2 (URIs)
├── rel_ids_1          # Relation ID mapping KG1
├── rel_ids_2          # Relation ID mapping KG2
├── ref_pairs          # Test alignment pairs (70%)
├── sup_pairs          # Train alignment pairs (20%)
└── valid_pairs        # Validation alignment pairs (10%)
```

## Examples

### Example 1: Basic Analysis

```bash
$ ./run_analysis.sh data/raw/hybea/BBC_DB/attribute_data

═══════════════════════════════════════════════════════════════════
   Dataset Analysis Tool
═══════════════════════════════════════════════════════════════════
📂 Dataset: data/raw/hybea/BBC_DB/attribute_data
───────────────────────────────────────────────────────────────────

======================================================================
ANALYZING ALIGNMENT STRUCTURE
======================================================================
Total unique alignment pairs: 9396
  ref_pairs:   6578 (70.0%)
  sup_pairs:   1879 (20.0%)
  valid_pairs: 939 (10.0%)

Overlap between sets:
  All three: 0
  ref ∩ sup: 0
  ref ∩ valid: 0
  sup ∩ valid: 0

✅ INVARIANT VERIFIED: All aligned entities appear in triples
✅ INVARIANT VERIFIED: All entities in triples are aligned

✅ Analysis completed successfully!
```

### Example 2: Save to JSON

```bash
$ ./run_analysis.sh data/raw/hybea/BBC_DB/attribute_data -o bbc_analysis.json

📄 Results saved to: bbc_analysis.json
✅ Analysis completed successfully!
```

### Example 3: Python API

```python
from pathlib import Path
from experiments.dataset_analysis import DatasetAnalyzer

# Analyze dataset
analyzer = DatasetAnalyzer(Path("data/raw/hybea/BBC_DB/attribute_data"))

# Load specific data
analyzer.load_entity_mappings()
analyzer.load_alignments()

# Access loaded data
print(f"Total entities KG1: {len(analyzer._index_to_uri_kg1)}")
print(f"Total entities KG2: {len(analyzer._index_to_uri_kg2)}")

# Run specific analysis
alignment_stats = analyzer.analyze_alignment_structure()
print(f"Ref pairs: {alignment_stats['ref_pairs']}")
print(f"Sup pairs: {alignment_stats['sup_pairs']}")
print(f"Valid pairs: {alignment_stats['valid_pairs']}")

# Full analysis with results
results = analyzer.run_full_analysis()
```

## Troubleshooting

### Error: "Dataset path not found"
- Check that the path points to an `attribute_data` directory
- Verify all required files exist (ent_ids, triples, attr_triples, *_pairs)

### Warning: "INVARIANT VIOLATED"
- This indicates the dataset doesn't follow expected structure
- Check the specific invariant that failed
- May indicate data corruption or incorrect format

### No output in quiet mode
- Use `-q/--quiet` only when running in batch mode
- Remove `--quiet` flag to see analysis results

## Contributing

When adding new analyses:

1. Add analysis method to `DatasetAnalyzer` class
2. Update `run_full_analysis()` to include new analysis
3. Document new invariants in this README
4. Add tests if applicable

## See Also

- [DAKGEA Main Documentation](../../README.md)
- [Dataset Formats](../../docs/dataset-guide.md)
- [Experiment Runner](../runner/README.md)
