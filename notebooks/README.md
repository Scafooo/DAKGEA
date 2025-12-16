# DAKGEA Tutorial Notebooks

This directory contains interactive Jupyter notebooks for learning how to use DAKGEA.

## 📚 Available Notebooks

### 1. Introduction (`01_introduction.ipynb`)
- Introduction to DAKGEA
- Setup and installation verification
- Framework overview
- Project structure

**Estimated time**: 5 minutes

### 2. Simple Experiment (`02_simple_experiment.ipynb`)
- First baseline experiment
- YAML configuration
- Execution and results analysis
- Understanding metrics

**Estimated time**: 10-15 minutes (+ execution time)

### 3. Data Augmentation (`03_data_augmentation.ipynb`)
- Adding PLM augmentation
- Baseline vs augmented comparison
- Interpreting improvements
- Parameter tuning

**Estimated time**: 15-20 minutes (+ execution time)

### 4. Analyze Results (`04_analyze_results.ipynb`) [TODO]
- Detailed results analysis
- Generating plots
- Advanced statistics
- Multi-configuration comparison

### 5. Advanced Configuration (`05_advanced_configuration.ipynb`) [TODO]
- Experiment suites
- Massive configurations
- Parallel execution
- Best practices

## 🚀 Getting Started

### Prerequisites

1. **Install Jupyter**:
```bash
pip install jupyter notebook
# or
pip install jupyterlab
```

2. **Verify DAKGEA installation**:
```bash
python -c "from experiments.runner.runner import ExperimentRunner; print('✓ OK')"
```

### Starting the Notebooks

#### Option A: Jupyter Notebook
```bash
cd notebooks
jupyter notebook
```

#### Option B: JupyterLab
```bash
cd notebooks
jupyter lab
```

#### Option C: VSCode
1. Install the "Jupyter" extension in VSCode
2. Open a `.ipynb` file
3. Select the Python kernel from DAKGEA's virtualenv

### Python Kernel

Make sure to use Python from DAKGEA's virtualenv:

```bash
# Activate virtualenv
source ../.venv/bin/activate  # Linux/Mac
# or
..\.venv\Scripts\activate     # Windows

# Start Jupyter from virtualenv
jupyter notebook
```

## 📖 Recommended Order

Follow the notebooks in numerical order:

```
01_introduction.ipynb         ← Start here
    ↓
02_simple_experiment.ipynb    ← First experiment
    ↓
03_data_augmentation.ipynb    ← Add augmentation
    ↓
04_analyze_results.ipynb      ← Advanced analysis [TODO]
    ↓
05_advanced_configuration.ipynb  ← Massive configurations [TODO]
```

## 💡 Tips

### Faster Execution

To reduce execution time during tutorials:

1. **Use small datasets**: `D_W_15K_V1` is a good choice
2. **Reduce ratio**: Use `reduction.ratio: 0.1` instead of 0.3
3. **GPU**: Make sure CUDA is available for BERT-INT

### Modifying Notebooks

The notebooks are interactive - feel free to:
- Modify configuration parameters
- Experiment with different datasets
- Add cells for custom analysis
- Save your modified versions

### Common Issues

#### "Module not found"
```python
# Add this cell at the beginning of the notebook
import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))
```

#### "CUDA out of memory"
```yaml
# Reduce batch size in model config
# or use CPU (slower)
```

#### Experiment already exists
```yaml
# In config, change:
overwrite_existing: true
# or change the experiment name
```

## 📚 Additional Resources

- **Documentation**: `../docs/index.md`
- **Getting Started Guide**: `../docs/guides/getting-started.md`
- **Config Examples**: `../config/experiments/`
- **Scripts**: `../scripts/`

## 🆘 Support

If you have problems:

1. Verify prerequisites (Python 3.11+, dependencies installed)
2. Consult documentation in `../docs/`
3. Check examples in `../config/experiments/`
4. Open an issue on GitHub

## ✅ Completion

After completing the notebooks, you will be able to:

- ✓ Configure and run DAKGEA experiments
- ✓ Use data reduction and augmentation
- ✓ Analyze and interpret results
- ✓ Optimize parameters for your use case
- ✓ Run massive experiments in parallel

Happy learning! 🚀
