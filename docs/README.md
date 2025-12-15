# DAKGEA Documentation

This directory contains all documentation for the DAKGEA project, organized as a wiki.

## 🏠 Start Here

**Main Entry Point**: [index.md](index.md)

The documentation wiki provides comprehensive guides, references, and tutorials for using DAKGEA.

## 📁 Documentation Structure

```
docs/
├── index.md                    # 🏠 Wiki home page (start here!)
│
├── architecture/               # 🏗️ System Architecture
│   ├── overview.md            # Architecture overview
│   ├── training-mode.md       # Training mode design (Stage-based)
│   └── technical-debt.md      # Known issues and improvements
│
├── guides/                    # 📖 User Guides
│   ├── overview.md            # Guides overview
│   ├── getting-started.md     # ⭐ Start here for new users
│   ├── quality-evaluation.md  # Synthetic data quality evaluation
│   ├── synthetic-comparison.md # Comparing training modes
│   └── latex-output.md        # Generating LaTeX tables
│
├── configuration/             # ⚙️ Configuration Reference
│   ├── overview.md            # Configuration overview
│   ├── augmentation.md        # Augmentation settings
│   ├── models.md              # Model configuration
│   ├── experiments.md         # Experiment setup
│   └── synthetic-comparison-experiments.md
│
├── experiments/               # 🧪 Experiments & Analysis
│   ├── overview.md            # Experiments overview
│   ├── metrics.md             # Evaluation metrics reference
│   ├── statistics.md          # Statistical analysis
│   ├── qualitative-analysis.md # Quality analysis methods
│   └── ea-metrics-guide.md    # Detailed metrics guide
│
├── models/                    # 🤖 Alignment Models
│   ├── overview.md            # Models overview
│   ├── bert-int.md            # BERT-INT documentation
│   └── hybea.md               # HybEA documentation
│
├── testing/                   # 🧪 Testing & Tuning
│   ├── overview.md            # Testing overview
│   ├── hyperparameter-tuning.md # Tuning guide
│   └── tuning-results.md      # Initial results
│
└── DAKGEA_report/             # 📊 Project report (existing)
    └── ...
```

## 📝 Contributing to Documentation

### Adding New Documentation

When adding new documentation:

1. **Choose the right category**:
   - User-facing guides → `guides/`
   - Architecture/design → `architecture/`
   - Configuration reference → `configuration/`
   - Experiment workflows → `experiments/`
   - Model documentation → `models/`
   - Testing/tuning → `testing/`

2. **Follow naming conventions**:
   - Use kebab-case: `my-new-guide.md`
   - Be descriptive: `quality-evaluation.md` not `qe.md`

3. **Update the index**:
   - Add entry to the section's `overview.md`
   - Add link in `index.md` if it's a major new section

4. **Use consistent formatting**:
   - Start with `# Title` (H1)
   - Use emojis for visual hierarchy (📋 🎯 ✅ etc.)
   - Include "Last Updated" date at bottom
   - Link to related docs

### Updating Existing Docs

1. **Read the existing content** to understand context
2. **Make changes** preserving the structure
3. **Update "Last Updated"** date at bottom
4. **Check links** if you moved/renamed files

### Documentation Standards

#### File Structure

```markdown
# Title

Brief introduction paragraph.

## 📋 Contents (if long document)

- [Section 1](#section-1)
- [Section 2](#section-2)

## Section 1

Content...

## Section 2

Content...

---

## 📚 Related Documentation

- [Related Doc 1](../path/to/doc.md)
- [Related Doc 2](../path/to/doc.md)

---

**Last Updated:** YYYY-MM-DD
```

#### Writing Style

- **Clear and concise**: Get to the point quickly
- **Examples**: Include code examples and commands
- **Visual hierarchy**: Use emojis and formatting for scanning
- **Links**: Cross-reference related documentation
- **Actionable**: Provide clear steps and commands

#### Code Examples

Always include working examples:

```yaml
# Good: Complete, runnable example
experiment:
  name: my_experiment
  dataset:
    name: openea/D_W_15K_V1
  model: bert_int
```

```bash
# Good: Actual commands users can run
python -m experiments.runner.runner config.yaml
```

## 🔗 Internal Linking

### Relative Paths

Always use relative paths for internal links:

```markdown
<!-- From guides/getting-started.md to configuration/augmentation.md -->
[Augmentation Config](../configuration/augmentation.md)

<!-- From index.md to guides/getting-started.md -->
[Getting Started](guides/getting-started.md)
```

### Anchors

Link to specific sections:

```markdown
[Quality Metrics](experiments/metrics.md#quality-gap)
```

## 📊 Documentation Coverage

Current coverage by category:

| Category | Files | Coverage |
|----------|-------|----------|
| Architecture | 3 | ✅ Complete |
| Guides | 5 | ✅ Complete |
| Configuration | 5 | ✅ Complete |
| Experiments | 5 | ✅ Complete |
| Models | 3 | ✅ Complete |
| Testing | 3 | ✅ Complete |

## ❌ What NOT to Document Here

**Do NOT add:**
- Temporary notes (use project issues/discussions)
- Code comments (put them in code)
- Personal research notes (use separate notebooks)
- Auto-generated API docs (use separate tool)

**Instead:**
- Use inline code comments for implementation details
- Use GitHub Issues for bugs/features
- Use GitHub Discussions for questions
- Keep docs focused on user-facing content

## 🎯 Quick Links

- **Main Wiki**: [index.md](index.md)
- **Getting Started**: [guides/getting-started.md](guides/getting-started.md)
- **Architecture**: [architecture/overview.md](architecture/overview.md)
- **Project README**: [../README.md](../README.md)

---

**Documentation Version:** 1.0
**Last Updated:** 2025-12-15
