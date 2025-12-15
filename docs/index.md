# DAKGEA Documentation Wiki

> **Data Augmentation for Knowledge Graph Entity Alignment**

Welcome to the DAKGEA documentation wiki. This is your central hub for all documentation related to the DAKGEA framework.

---

## 🏠 Navigation

### 🏗️ [Architecture](architecture/overview.md)
**System Design and Technical Architecture**

Learn about the architectural design, training mode system, and technical debt.

**Key Pages:**
- [Training Mode Architecture](architecture/training-mode.md) - Stage-based filtering design
- [Technical Debt](architecture/technical-debt.md) - Known issues and improvement opportunities

---

### 📖 [User Guides](guides/overview.md)
**Practical Guides for Using DAKGEA**

Step-by-step guides for common tasks and workflows.

**Key Pages:**
- [Quality Evaluation Guide](guides/quality-evaluation.md) - Evaluate synthetic data quality
- [Synthetic Comparison Guide](guides/synthetic-comparison.md) - Compare training modes
- [LaTeX Output Generation](guides/latex-output.md) - Generate publication tables

---

### ⚙️ [Configuration](configuration/overview.md)
**Setting Up Experiments and Models**

Reference for configuring experiments, models, and augmentation methods.

**Key Pages:**
- [Augmentation Configuration](configuration/augmentation.md) - Configure PLM augmentation
- [Model Configuration](configuration/models.md) - Configure alignment models
- [Experiment Configuration](configuration/experiments.md) - Structure experiment configs

---

### 🧪 [Experiments](experiments/overview.md)
**Running Experiments and Analyzing Results**

Documentation for experiments, metrics, and analysis workflows.

**Key Pages:**
- [Metrics Reference](experiments/metrics.md) - Evaluation metrics explained
- [Statistics and Analysis](experiments/statistics.md) - Statistical analysis methods
- [Qualitative Analysis](experiments/qualitative-analysis.md) - In-depth quality analysis
- [EA Metrics Guide](experiments/ea-metrics-guide.md) - Entity alignment metrics

---

### 🤖 [Models](models/overview.md)
**Entity Alignment Models**

Documentation for supported alignment models.

**Key Pages:**
- [BERT-INT](models/bert-int.md) - BERT-based interaction model
- [HybEA](models/hybea.md) - Hybrid entity alignment model

---

### 🧪 [Testing](testing/overview.md)
**Testing and Hyperparameter Tuning**

Guides for testing workflows and optimization.

**Key Pages:**
- [Hyperparameter Tuning](testing/hyperparameter-tuning.md) - Systematic optimization
- [Tuning Results](testing/tuning-results.md) - Initial tuning findings

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/DAKGEA.git
cd DAKGEA

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Your First Experiment

```bash
# Run a simple baseline experiment
python -m experiments.runner.runner config/experiments/my_first_experiment.yaml
```

### Evaluate Synthetic Data Quality

```bash
# Run quality evaluation (baseline vs synthetic-only)
bash scripts/run_quality_evaluation.sh --model bert_int --jobs 4 --fair-comparison
```

---

## 📊 Project Overview

### What is DAKGEA?

DAKGEA is a framework for:
1. **Data Augmentation**: Generate synthetic entity alignment pairs using pretrained language models
2. **Quality Evaluation**: Evaluate the quality of synthetic data
3. **Entity Alignment**: Train and evaluate entity alignment models on augmented data

### Pipeline Workflow

```
Original Dataset
    │
    ├─> Reduction (reduce dataset size)
    │       └─> Reduced Dataset
    │
    ├─> Augmentation (generate synthetic pairs)
    │       └─> Augmented Dataset (original + synthetic)
    │
    ├─> Filtering (select training mode)
    │       ├─> Baseline: original only
    │       ├─> Augmented: original + synthetic
    │       └─> Synthetic-only: synthetic only
    │
    └─> Training & Evaluation
            └─> Performance Metrics
```

### Key Features

- ✅ **Modular Architecture**: Clean separation of concerns (SRP)
- ✅ **Multiple Training Modes**: Baseline, augmented, synthetic-only
- ✅ **Quality Evaluation**: Fair comparison methodology
- ✅ **Parallel Execution**: Efficient batch processing
- ✅ **Multiple Models**: BERT-INT, RREA, HybEA support
- ✅ **Comprehensive Metrics**: Hits@K, MRR, Quality Gap, Transferability

---

## 🎯 Common Tasks

### Running Experiments

```bash
# Run single experiment
python -m experiments.runner.runner config/experiments/experiment.yaml

# Run experiments in parallel
bash scripts/run_experiments_parallel.sh --dir config/experiments/massive/bert_int_baseline --jobs 4

# Run quality evaluation
bash scripts/run_quality_evaluation.sh --model bert_int --fair-comparison
```

### Generating Results

```bash
# Generate LaTeX tables
python experiments/statistics/generate_latex_tables.py

# Compare quality metrics
python experiments/statistics/compare_quality.py --model bert_int

# Analyze results
python experiments/statistics/analyze_results.py
```

### Configuration

```bash
# Generate baseline configs
python scripts/tools/generate_massive_baseline_configs.py

# Generate synthetic-only configs
python scripts/tools/generate_massive_synthetic_only_configs.py
```

---

## 📚 Additional Resources

### Project Structure

```
DAKGEA/
├── src/                      # Source code
│   ├── augmentation/         # Augmentation methods
│   ├── reduction/            # Reduction methods
│   ├── alignment_models/     # Entity alignment models
│   └── core/                 # Core components
├── experiments/              # Experiment orchestration
│   ├── runner/               # Experiment runner
│   └── statistics/           # Analysis scripts
├── config/                   # Configuration files
│   ├── augmentation/         # Augmentation configs
│   ├── models/               # Model configs
│   └── experiments/          # Experiment configs
├── scripts/                  # Utility scripts
├── tests/                    # Test suite
└── docs/                     # Documentation (this wiki)
```

### External Links

- **GitHub Repository**: [github.com/yourusername/DAKGEA](https://github.com/yourusername/DAKGEA)
- **Research Paper**: [link to paper]
- **Issue Tracker**: [GitHub Issues](https://github.com/yourusername/DAKGEA/issues)

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines (coming soon).

### Documentation Updates

All documentation is now centralized under `docs/`. When adding new documentation:

1. Place it in the appropriate category directory
2. Update the relevant `overview.md` file
3. Add a link in this index page if it's a major new section

---

## 📮 Contact & Support

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/yourusername/DAKGEA/issues)
- **Questions**: Open a discussion on GitHub Discussions
- **Email**: [your.email@example.com]

---

**Documentation Version:** 1.0
**Last Updated:** 2025-12-15
**Framework Version:** 0.1.0
