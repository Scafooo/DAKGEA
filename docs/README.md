# DAKGEA Documentation Index

Welcome to the DAKGEA documentation! This index helps you find the right guide for your needs.

---

## 🎯 I Want To...

### Get Started
- **Install and run my first experiment** → [User Guide](user-guide.md)
- **Understand what DAKGEA does** → [Main README](../README.md)
- **See example configurations** → [Configuration Guide - Examples](configuration-guide.md#complete-examples)

### Work with Datasets
- **Use an existing dataset** → [Configuration Guide - Dataset Configuration](configuration-guide.md#dataset-configuration)
- **Convert between formats** → [Dataset Guide - Converting Between Formats](dataset-guide.md#converting-between-formats)
- **Add my own dataset** → [Dataset Guide - Adding Custom Datasets](dataset-guide.md#adding-custom-datasets)
- **Understand dataset formats** → [Dataset Guide - Dataset Formats](dataset-guide.md#dataset-formats)

### Configure Experiments
- **Create an experiment config** → [Configuration Guide](configuration-guide.md)
- **Set reduction ratios** → [Configuration Guide - Augmentation Configuration](configuration-guide.md#augmentation-configuration)
- **Configure BERT-INT** → [BERT-INT Guide - Configuration](bert-int-guide.md#configuration)
- **Override model parameters** → [Configuration Guide - Advanced Options](configuration-guide.md#advanced-options)

### Use BERT-INT
- **Understand BERT-INT architecture** → [BERT-INT Guide - Architecture](bert-int-guide.md#architecture)
- **Train BERT-INT** → [BERT-INT Guide - Training Pipeline](bert-int-guide.md#training-pipeline)
- **Tune performance** → [BERT-INT Guide - Performance Tuning](bert-int-guide.md#performance-tuning)
- **Compare with reference implementation** → [BERT-INT Guide - Reference Comparison](bert-int-guide.md#reference-comparison)

### Troubleshoot Issues
- **Fix configuration errors** → [FAQ - Errors & Troubleshooting](faq.md#errors--troubleshooting)
- **Solve CUDA/GPU issues** → [FAQ - CUDA out of memory](faq.md#cuda-out-of-memory)
- **Debug dataset loading** → [Dataset Guide - Troubleshooting](dataset-guide.md#troubleshooting)
- **Understand error messages** → [FAQ](faq.md)

### Extend DAKGEA
- **Add a custom model** → [Developer Guide](developer-guide.md)
- **Add a custom augmentation method** → [FAQ - How do I add a custom augmentation method?](faq.md#how-do-i-add-a-custom-augmentation-method)
- **Implement a custom reader** → [Dataset Guide - Adding Custom Datasets](dataset-guide.md#adding-custom-datasets)
- **Understand the codebase** → [Developer Guide](developer-guide.md)

---

## 📚 Documentation by Type

### User Documentation

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [User Guide](user-guide.md) | Installation, basic usage, running experiments | First time setup, learning basics |
| [Configuration Guide](configuration-guide.md) | Complete configuration reference | Creating/modifying experiments |
| [Dataset Guide](dataset-guide.md) | Dataset formats, readers, writers | Working with datasets |
| [BERT-INT Guide](bert-int-guide.md) | BERT-INT model specifics | Using BERT-INT model |
| [FAQ](faq.md) | Common questions and answers | Troubleshooting, quick answers |

### Developer Documentation

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [Developer Guide](developer-guide.md) | Architecture, extending the framework | Adding features, understanding code |

---

## 📖 Documentation Structure

### [User Guide](user-guide.md)
1. Install & Setup
2. Run an Experiment
3. Understand the Outputs
4. Customise an Experiment
5. Troubleshooting
6. Keep the Project Healthy

### [Configuration Guide](configuration-guide.md)
1. Configuration File Structure
2. Dataset Configuration
   - Simple dataset name
   - Explicit reader/dataset path
   - Direct path mode
   - Full configuration
3. Augmentation Configuration
   - Modern syntax (recommended)
   - Legacy syntax
4. Model Configuration
5. Advanced Options
6. Complete Examples (6 detailed examples)

### [Dataset Guide](dataset-guide.md)
1. Overview
2. Dataset Formats
   - HybEA format
   - BERT-INT format
   - RDF format
3. Directory Structure
4. Readers
   - HybEA reader
   - BERT-INT reader
   - RDF reader
5. Writers
   - BERT-INT writer
   - HybEA writer
   - Multi-writer support
6. Working with Datasets
7. Adding Custom Datasets
8. Troubleshooting

### [BERT-INT Guide](bert-int-guide.md)
1. Overview
2. Architecture
   - Phase 1: Basic Unit
   - Phase 2: Interaction Model
3. Configuration
   - Basic configuration
   - Full configuration
   - Parameter reference
4. Dataset Requirements
5. Training Pipeline
6. Performance Tuning
   - GPU memory optimization
   - CPU-only mode
   - Speed vs. accuracy tradeoffs
7. Troubleshooting (6 common issues)
8. Reference Comparison
9. Advanced Topics

### [FAQ](faq.md)
1. General Questions (3)
2. Installation & Setup (3)
3. Configuration (6)
4. BERT-INT Specific (6)
5. Errors & Troubleshooting (8)
6. Data & Formats (5)
7. Advanced Usage (4)
8. Results & Metrics (4)
9. Best Practices (3)
10. Getting Help (3)

---

## 🔍 Quick Reference

### Configuration Snippets

**Basic Experiment:**
```yaml
experiment:
  name: "my_experiment"
  dataset:
    name: "hybea/BBC_DB"
  augmentation:
    method: "stub"
    reduction: 0.1
  model: bert_int
  seed: 42
```

**Direct Path Mode:**
```yaml
experiment:
  name: "direct_test"
  dataset:
    path: "/path/to/data"
  model: bert_int
```

**Multi-Model Comparison:**
```yaml
experiment:
  name: "comparison"
  dataset:
    name: "hybea/BBC_DB"
  augmentation:
    method: "stub"
    reduction: 0.2
  models_to_run: ["bert_int", "hybea"]
```

### Common Commands

```bash
# Run experiment
./run.sh config/experiments/my_config.yaml

# Run with overwrite
./run.sh config/experiments/my_config.yaml --overwrite-existing

# Run in resume mode
./run.sh config/experiments/my_config.yaml --resume

# Check results
cat results/<experiment>/<dataset>/<ratio>/evaluation/reduced/bert_int.json

# Run tests
pytest tests/
```

### File Locations

```
DAKGEA/
├── config/
│   ├── experiments/          # Experiment configurations
│   ├── models/              # Model configurations
│   └── global.yaml          # Global settings
├── data/
│   └── raw/                 # Raw datasets
│       ├── hybea/
│       ├── bert_int/
│       └── rdf/
├── results/                 # Experiment outputs
│   └── <experiment>/
│       └── <dataset>/
│           └── <ratio>/
├── docs/                    # Documentation
└── src/                     # Source code
```

---

## 🎓 Learning Path

### Beginner Path
1. Read [Main README](../README.md) - Overview
2. Follow [User Guide](user-guide.md) - Installation
3. Run example experiment - `./run.sh config/experiments/01_exp_direct.yaml`
4. Read [Configuration Guide - Examples](configuration-guide.md#complete-examples)
5. Create your first custom experiment
6. Consult [FAQ](faq.md) when stuck

### Intermediate Path
1. Study [Dataset Guide](dataset-guide.md) - Understand formats
2. Read [BERT-INT Guide](bert-int-guide.md) - Model details
3. Experiment with different configurations
4. Tune model parameters
5. Compare with reference implementation

### Advanced Path
1. Read [Developer Guide](developer-guide.md) - Architecture
2. Study source code structure
3. Implement custom augmentation method
4. Add custom dataset reader
5. Contribute to the project

---

## 💡 Tips for Using Documentation

### Finding Information Fast

1. **Use Ctrl+F / Cmd+F** to search within documents
2. **Check FAQ first** for common questions
3. **Use the index** (this page) to navigate to the right guide
4. **Read examples** in Configuration Guide for patterns

### When You're Stuck

1. **Check error message** against FAQ troubleshooting section
2. **Compare your config** with working examples
3. **Verify paths** using Dataset Guide
4. **Enable debug logging** in config:
   ```yaml
   logging:
     level: DEBUG
   ```

### Contributing to Documentation

Found an error? Have a suggestion?

1. Open an issue on GitHub
2. Include:
   - Document name and section
   - What's unclear or incorrect
   - Suggested improvement (optional)

---

## 📞 Getting Help

### Documentation Issues
- Unclear explanations? → Open a GitHub issue
- Missing examples? → Request in GitHub issues
- Found errors? → Submit a PR or issue

### Code Issues
- Configuration not working? → Check [FAQ](faq.md)
- Model failing? → See [BERT-INT Guide - Troubleshooting](bert-int-guide.md#troubleshooting)
- Dataset errors? → See [Dataset Guide - Troubleshooting](dataset-guide.md#troubleshooting)

### Contributing
- Want to add features? → Read [Developer Guide](developer-guide.md)
- Found a bug? → Open a GitHub issue
- Want to help? → Check open issues

---

## 📝 Documentation Conventions

### Code Examples

**Configuration files:**
```yaml
# Comments explain the purpose
key: "value"
```

**Python code:**
```python
# Comments explain the logic
def example_function():
    pass
```

**Shell commands:**
```bash
# Commands you can run
./run.sh config.yaml
```

### Formatting

- **Bold** - Important terms, actions
- `code` - Filenames, commands, code snippets
- *Italic* - Emphasis
- → - "See this document"

### Symbols

- ✓ - Correct approach
- ✗ - Incorrect approach
- 📊 - Results/metrics
- 🔧 - Configuration
- 🐛 - Bug/issue
- 💡 - Tip/hint
- ⚠️ - Warning

---

## 🗺️ Documentation Roadmap

### Current Version

All core documentation complete:
- ✅ User Guide
- ✅ Configuration Guide
- ✅ Dataset Guide
- ✅ BERT-INT Guide
- ✅ FAQ
- ✅ Developer Guide

### Planned Additions

- [ ] API Reference (auto-generated from docstrings)
- [ ] Tutorial videos (planned)
- [ ] Migration guide from reference implementations
- [ ] Performance benchmarks documentation
- [ ] Advanced augmentation techniques guide

---

## 📅 Last Updated

Documentation last updated: 2025-01-05

For the latest updates, check the [GitHub repository](https://github.com/Scafooo/DataAug-KG-EntityResolution).

---

## 🙏 Acknowledgments

Documentation structure inspired by:
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

**Ready to get started?** → [User Guide](user-guide.md)

**Need quick help?** → [FAQ](faq.md)

**Want examples?** → [Configuration Guide - Examples](configuration-guide.md#complete-examples)
