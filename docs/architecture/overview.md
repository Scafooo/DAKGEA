# Architecture Overview

This section contains architectural documentation for the DAKGEA project.

## 📋 Contents

### [Training Mode Architecture](training-mode.md)
**Stage-Based Architecture for Data Filtering**

Describes the clean, modular architecture for handling different training modes (baseline, augmented, synthetic-only) in the data augmentation pipeline.

**Key Topics:**
- Single Responsibility Principle implementation
- FilteringStage design pattern
- PLMAugmenter, FilteringStage, and ExperimentRunner separation
- Training mode configuration examples
- Evolution through 4 design iterations

**Status:** ✅ **Stage-Based Architecture** - Clean, modular, fully respects SOLID principles

---

### [Technical Debt](technical-debt.md)
**Known Architectural Issues and Improvement Opportunities**

Documents technical debt items identified during code reviews. These represent opportunities for future improvements in maintainability, testability, and extensibility.

**Priority Issues:**
- 🔴 **Critical**: BartInterpolatorPLM God Class (1335 lines)
- 🟠 **High**: ExperimentRunner long methods
- 🟠 **High**: NodeExpander complex methods
- 🟡 **Medium**: Code duplication patterns

**Status:** Documented for future refactoring (2025-12-15)

---

## 🎯 Architecture Principles

The DAKGEA architecture follows these core principles:

1. **Single Responsibility Principle (SRP)**: Each component has one clear responsibility
2. **Separation of Concerns**: Clear boundaries between data, logic, and orchestration
3. **Modularity**: Components are reusable and testable in isolation
4. **Stage-Based Pipeline**: Clean flow through Reduction → Augmentation → Filtering → Evaluation

## 🏗️ High-Level Architecture

```
DAKGEA Framework
    │
    ├─ Core Components
    │   ├─ Dataset Management (src/core/dataset/)
    │   ├─ Knowledge Graphs (src/core/knowledge_graph/)
    │   └─ Configuration (src/config/)
    │
    ├─ Data Pipeline
    │   ├─ Reduction (src/reduction/)
    │   ├─ Augmentation (src/augmentation/)
    │   └─ Filtering (experiments/runner/stages.py)
    │
    ├─ Alignment Models
    │   ├─ BERT-INT (src/alignment_models/methods/bert_int/)
    │   └─ RREA (src/alignment_models/methods/rrea/)
    │
    └─ Experiment Orchestration
        └─ ExperimentRunner (experiments/runner/)
```

## 📚 Related Documentation

- **Guides**: See [guides/](../guides/overview.md) for user-facing documentation
- **Configuration**: See [configuration/](../configuration/overview.md) for setup details
- **Models**: See [models/](../models/overview.md) for alignment model documentation

---

**Last Updated:** 2025-12-15
