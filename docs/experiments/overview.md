# Experiments and Analysis

This section contains documentation for running experiments, analyzing results, and understanding evaluation metrics.

## 📋 Contents

### [Metrics Reference](metrics.md)
**Entity Alignment Evaluation Metrics**

Complete reference for all evaluation metrics used in DAKGEA.

**Key Metrics:**
- **Hits@K**: Percentage of correct alignments in top-K predictions
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks
- **Precision, Recall, F1**: Standard classification metrics
- **Quality Gap**: Performance difference (Baseline - Synthetic-only)
- **Transferability Score**: Ratio (Synthetic-only / Baseline)

---

### [Statistics and Analysis](statistics.md)
**Statistical Analysis of Experimental Results**

Guide for analyzing and comparing experimental results.

**Key Topics:**
- Loading and processing results
- Statistical significance testing
- Comparison across datasets
- Trend analysis
- Result visualization

---

### [Qualitative Analysis](qualitative-analysis.md)
**In-Depth Analysis of Alignment Quality**

Methods for qualitative analysis of entity alignments.

**Key Topics:**
- Manual inspection workflows
- Error analysis
- Alignment quality patterns
- Case studies
- Common failure modes

---

### [EA Metrics Guide](ea-metrics-guide.md)
**Entity Alignment Metrics - Detailed Guide**

Detailed explanation of entity alignment metrics with examples and interpretation guidelines.

**Key Topics:**
- Metric definitions and formulas
- Interpretation guidelines
- Use cases for each metric
- Common pitfalls
- Best practices

---

## 🎯 Analysis Workflows

### Standard Analysis Pipeline

```bash
# 1. Run experiments
bash scripts/run_quality_evaluation.sh --model bert_int --jobs 4

# 2. Generate statistics
python experiments/statistics/compare_quality.py --model bert_int

# 3. Generate LaTeX tables
python experiments/statistics/generate_latex_tables.py

# 4. Perform qualitative analysis
python experiments/qualitative_analysis/analyze_alignments.py
```

### Quick Result Check

```bash
# Check experiment completion
python experiments/statistics/check_completion.py

# View summary statistics
python experiments/statistics/summarize_results.py --model bert_int
```

---

## 📊 Understanding Results

### Quality Evaluation Results

**Quality Gap** = Performance(Baseline) - Performance(Synthetic-only)

| Quality Gap | Interpretation |
|-------------|----------------|
| < 5% | **Excellent** - Synthetic can replace real data |
| 5-15% | **Good** - Synthetic maintains general patterns |
| > 15% | **Poor** - Significant quality issues |

**Transferability Score** = Synthetic-only / Baseline

| Score Range | Interpretation |
|-------------|----------------|
| > 0.95 | Excellent transferability |
| 0.85-0.95 | Good transferability |
| < 0.85 | Limited transferability |

### Result Files Location

```
results/
├── {dataset}/
│   ├── reduction_{ratio}/
│   │   ├── reduction/
│   │   │   └── results.json       # Baseline results
│   │   └── augmentation/
│   │       └── results.json       # Augmented/Synthetic results
│   └── summary.json
└── aggregated/
    └── comparison_tables.tex       # LaTeX output
```

---

## 📚 Related Documentation

- **Guides**: See [guides/quality-evaluation.md](../guides/quality-evaluation.md) for running experiments
- **Configuration**: See [configuration/](../configuration/overview.md) for experiment setup
- **Models**: See [models/](../models/overview.md) for model-specific metrics

---

**Last Updated:** 2025-12-15
