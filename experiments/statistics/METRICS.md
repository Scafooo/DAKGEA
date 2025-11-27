# Metrics Support in DAKGEA Statistics Module

## Available Metrics

The statistics analysis module supports the following metrics:

### Ranking Metrics (Hits@K)
- **hits@1**: Percentage of correct entities in top-1 rank
- **hits@5**: Percentage of correct entities in top-5 rank
- **hits@10**: Percentage of correct entities in top-10 rank
- **hits@25**: Percentage of correct entities in top-25 rank
- **hits@50**: Percentage of correct entities in top-50 rank

### Aggregated Ranking Metrics
- **mrr**: Mean Reciprocal Rank - average of 1/rank for correct entities
- **mr**: Mean Rank - average rank of correct entities

### Classification Metrics
- **precision**: True positives / (True positives + False positives)
- **recall**: True positives / (True positives + False negatives)
- **f-measure**: Harmonic mean of precision and recall (2 * precision * recall / (precision + recall))

## Default Metrics

When no `--metrics` flag is provided, the analyzer uses all available metrics:
```bash
./scripts/analyze_results.sh
# Uses: hits@1, hits@5, hits@10, hits@25, hits@50, mrr, mr, precision, recall, f-measure
```

## Custom Metric Selection

You can specify which metrics to analyze:
```bash
# Only ranking metrics
./scripts/analyze_results.sh --metrics hits@1 hits@5 hits@10 mrr

# Only classification metrics
./scripts/analyze_results.sh --metrics precision recall f-measure

# Mixed selection
./scripts/analyze_results.sh --metrics hits@1 precision recall f-measure
```

## Output Locations

### TSV Export (Enhanced)
File: `results/statistics/dataset_summary.tsv`

Columns include:
- reduction_mean, reduction_std, reduction_min, reduction_max
- augmentation_mean, augmentation_std, augmentation_min, augmentation_max
- delta_mean, delta_percentage

### CSV Export
File: `results/statistics/dataset_summary.csv`

Same structure as TSV for compatibility with Excel and data analysis tools

### Markdown Export
File: `results/statistics/dataset_summary.md`

Formatted table showing:
- Dataset, Metric
- Red Mean, Red Std, Aug Mean, Aug Std
- Delta, Δ% (percentage change)

### Console Output

When analyzing, metrics are displayed per dataset:
```
=== Dataset: BBC_DB ===
  Reduction (ratio mean=0.350)
    - hits@1: mean=0.7059, std=0.1956, min=0.0125, max=0.8862, n=60
    - precision: mean=0.8234, std=0.1425, min=0.5000, max=0.9667, n=60
  Augmentation (ratio mean=0.550)
    - hits@1: mean=0.7158, std=0.1745, min=0.0125, max=0.9002, n=60
    - precision: mean=0.8456, std=0.1234, min=0.5200, max=0.9800, n=60
  Advanced statistics:
    - hits@1: Cohen's d = 0.0533
    - precision: Cohen's d = 0.1725
```

## Metric Value Ranges

All metrics are normalized to [0, 1] range:
- **1.0** = Perfect performance
- **0.0** = Worst performance
- **0.5** = 50% performance

## Statistical Analysis

Each metric gets statistical analysis including:
- **Mean**: Average value across all experiments
- **Std**: Standard deviation (spread of values)
- **Min/Max**: Minimum and maximum values observed
- **Count**: Number of data points

## Advanced Statistics (with --advanced-stats)

Additional statistical measures include:
- **Cohen's d**: Effect size between reduction and augmentation
- **Paired t-test**: Statistical significance test
- **p-value**: Probability the difference is due to chance
- **Confidence intervals**: Range where true value likely lies

## Visualization Support

All metrics can be visualized with:
- **Bar charts**: Reduction vs augmentation comparison
- **Heatmaps**: Metric values across ratio combinations
- **Boxplots**: Distribution and outliers
- **Violin plots**: Distribution shape
- **Scatter plots**: Correlation between stages
- **Delta charts**: Improvement/degradation

## Adding New Metrics

To add support for new metrics:

1. Include the metric name in `DEFAULT_METRICS` in `analyze_results.py`
2. Add color mapping in `METRIC_COLORS` if desired
3. Update `_sanitize_metric()` if special handling is needed

Example:
```python
DEFAULT_METRICS = [..., "custom_metric"]
METRIC_COLORS = {..., "custom_metric": "#ff0000"}
```

## Metric Interpretation

### For Ranking Metrics
- **Higher is better**: hits@K and mrr should be as high as possible
- **Lower is better**: mr (mean rank) should be as low as possible

### For Classification Metrics  
- **Higher is better**: precision, recall, and f-measure should all be high
- **F-measure trade-off**: balances precision and recall

### Delta Analysis
- **Positive delta**: Augmentation improved the metric
- **Negative delta**: Augmentation degraded the metric
- **Delta %**: Percentage improvement/degradation relative to reduction baseline
