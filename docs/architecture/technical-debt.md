# Technical Debt - DAKGEA Project

> **Last Updated**: 2025-12-15
>
> **Status**: Documented for future refactoring
>
> This document tracks architectural issues and technical debt identified during code reviews. The codebase is currently functional, but these items represent opportunities for improvement in maintainability, testability, and extensibility.

---

## 🔴 Critical Priority

### 1. BartInterpolatorPLM - God Class (1335 lines)

**File**: `src/augmentation/methods/plm/bart_interpolator.py`

**Issue**: Severe violation of Single Responsibility Principle. This class handles:
- BART model configuration (lines 174-245)
- Advanced training modules initialization (lines 247-331)
- Model fine-tuning (lines 475-668)
- Training pairs construction (lines 672-864)
- Latent interpolation (lines 900-1335)
- Generation with retry mechanism (lines 1028-1100)
- Edit distance constraints (lines 1229-1270)

**Impact**:
- Difficult to unit test
- Hard to maintain (1335 lines!)
- High coupling between different responsibilities
- Risky to add new features (high regression risk)

**Proposed Refactoring**:

Split into 4-5 specialized classes:

```python
# 1. BartModelManager - Model lifecycle management
class BartModelManager:
    """Handles BART model loading, initialization, and saving."""
    def load_or_init_model(self) -> Tuple[BartModel, BartTokenizer]: ...
    def save_model(self, path: Path) -> None: ...

# 2. BartFineTuner - Fine-tuning logic
class BartFineTuner:
    """Handles BART fine-tuning with advanced training modules."""
    def fine_tune(self, pairs: List[PairExample]) -> None: ...

# 3. TrainingPairBuilder - Dataset construction
class TrainingPairBuilder:
    """Builds training pairs from knowledge graphs."""
    def build_pairs_from_dataset(self, kg_source, kg_target) -> List[PairExample]: ...

# 4. BartInterpolator - ONLY interpolation
class BartInterpolator:
    """Performs latent interpolation between values."""
    def interpolate_pair(self, val_src: str, val_tgt: str) -> Tuple[str, str]: ...

# 5. Facade for backward compatibility
class BartInterpolatorPLM:
    """Facade that coordinates BART operations."""
    def __init__(self, ...):
        self.model_manager = BartModelManager(...)
        self.interpolator = BartInterpolator(...)
        # ...
```

**Effort**: High (1-2 weeks)
**Risk**: Medium (requires careful testing)

---

## 🟠 High Priority

### 2. ExperimentRunner - Long Methods

**File**: `experiments/runner/runner.py`

#### 2a. Method `_run_standard_mode` (lines 364-504) - 140 lines

**Issue**: Violates SRP by handling too many steps in a single method.

**Proposed Refactoring**:

```python
def _run_standard_mode(self) -> None:
    context = self._prepare_experiment_context()
    stages = self._create_pipeline_stages(context)
    self._execute_pipeline(context, stages, progress)

def _prepare_experiment_context(self) -> ExperimentContext:
    """Prepare dataset and workspace."""
    # Extract lines 376-396

def _create_pipeline_stages(self, context) -> PipelineStages:
    """Create reduction, augmentation, and evaluation stages."""
    # Extract lines 397-411

def _execute_pipeline(self, context, stages, progress) -> None:
    """Execute the pipeline stages."""
    # Extract lines 420-498
```

**Effort**: Medium (2-3 days)
**Risk**: Low

#### 2b. Method `_run_augmentation_with_retry` (lines 1108-1341) - 233 lines

**Issue**: Too long with inline retry logic.

**Proposed Refactoring**:

```python
class AugmentationRetryManager:
    """Manages retry logic for augmentation until improvement."""

    def execute_with_retry(
        self,
        augmentation_fn: Callable,
        evaluation_fn: Callable,
        baseline_fn: Callable,
    ) -> AugmentationResult: ...
```

**Effort**: Medium (3-4 days)
**Risk**: Low

---

### 3. NodeExpander._interpolate_literals - Feature Envy + Long Method

**File**: `src/augmentation/methods/plm/node_expander.py`
**Lines**: 199-473 (274 lines!)

**Issues**:
1. **Feature Envy**: Heavy access to `self.bart_interpolator`, `self.predicate_matcher`, `self.alignment_cache`
2. **Long Method**: 274 lines with very complex logic
3. **Nested Functions**: `normalize_cache_key` and `reorder_output_to_match_input` defined inline (lines 326-371)

**Proposed Refactoring**:

```python
# Extract nested functions as static methods or utilities
class ValueCacheNormalizer:
    """Normalizes and reorders values for cache consistency."""

    @staticmethod
    def normalize_cache_key(text: str) -> str: ...

    @staticmethod
    def reorder_output_to_match_input(input_text: str, output: str) -> str: ...

# Extract matching logic into dedicated class
class PredicateMatchingOrchestrator:
    """Orchestrates predicate matching with cache and fallback."""

    def find_matches(self, src_literals, tgt_literals, dataset) -> List[Match]: ...

# NodeExpander becomes leaner
class NodeExpander:
    def _interpolate_literals(self, ...):
        # ~50 lines instead of 274
        matcher = PredicateMatchingOrchestrator(...)
        matches = matcher.find_matches(...)

        for match in matches:
            self._process_match(match, ...)
```

**Effort**: Medium (3-4 days)
**Risk**: Medium

---

### 4. PLMAugmenter - Long Methods

**File**: `src/augmentation/methods/plm/plm_augmenter.py`

#### 4a. Method `_bfs_expansion` (lines 399-504) - 105 lines

**Proposed Refactoring**: Extract cache management into dedicated class:

```python
class InterNodeCacheManager:
    """Manages inter-node value consistency caches."""

    def get_or_create_cache(self, scope: str, seed_node: URIRef) -> Dict: ...
    def clear_for_scope(self, scope: str) -> None: ...
```

**Effort**: Medium (2-3 days)

#### 4b. Method `_expand_non_set_node_with_neighbors` (lines 969-1065) - 96 lines

**Proposed Refactoring**: Extract common relation creation logic:

```python
class RelationCreator:
    """Creates relations between augmented entities."""

    def create_relations(
        self,
        dataset: Dataset,
        source_aug: URIRef,
        target_aug: URIRef,
        neighbor_aug: Tuple[URIRef, URIRef],
        direction: str,
        predicate: URIRef,
    ) -> None: ...
```

**Effort**: Medium (2-3 days)

---

## 🟡 Medium Priority - Code Duplication

### 5. Repeated Pattern: Check Node in Graph

**Issue**: The `_node_in_graph` pattern appears in multiple files:
- `src/augmentation/methods/plm/neighbor_handler.py` (lines 161-178)
- `src/augmentation/methods/plm/plm_augmenter.py` (lines 822-831, 845-854, etc.)

**Proposed Solution**:

```python
# Create shared utility: src/augmentation/methods/plm/graph_utils.py

def node_exists_in_graph(graph: KnowledgeGraph, node: URIRef) -> bool:
    """Check if a node exists in the graph."""
    if not isinstance(node, URIRef):
        return False
    try:
        next(graph.triples((node, None, None)))
        return True
    except StopIteration:
        pass
    try:
        next(graph.triples((None, None, node)))
        return True
    except StopIteration:
        return False
```

**Effort**: Low (1 day)
**Risk**: Very Low

---

### 6. Configuration Duplication

**Issue**: Config merge/extraction logic repeated in:
- `PLMAugmenter._merge_configs` (lines 80-96)
- `ExperimentConfig._extract_*` multiple methods

**Proposed Solution**: Create centralized configuration module:

```python
# src/config/utils.py

def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two configuration dictionaries."""
    ...

class ConfigExtractor:
    """Extract and normalize configuration values."""

    @staticmethod
    def ensure_sequence(value: Any) -> List[Any]: ...

    @staticmethod
    def extract_with_fallback(payload: dict, *keys, default=None): ...
```

**Effort**: Low (1-2 days)
**Risk**: Very Low

---

## 🔵 Organizational Improvements

### 7. PLM Module Structure

**Current Structure** (fragmented):

```
plm/
  __init__.py
  augmenter.py        # Only imports
  plm_augmenter.py    # Main class
  plm_augmenter_old.py  # Legacy file (remove?)
  bart_interpolator.py
  bart_training_modules.py
  node_expander.py
  neighbor_handler.py
  predicate_alignment.py
  predicate_matcher.py
  expansion_node.py
  sentence_interpolator.py
  bart/
  models/
  services/
  set_knowledge_graph/
```

**Proposed Reorganization**:

```
plm/
  __init__.py
  augmenter.py           # Main PLMAugmenter

  # BART subsystem
  bart/
    __init__.py
    model_manager.py     # Model management
    fine_tuner.py        # Fine-tuning
    interpolator.py      # Interpolation
    training_modules.py  # Advanced modules
    pair_builder.py      # Pairs construction
    sentence_interpolator.py

  # Expansion subsystem
  expansion/
    __init__.py
    node_expander.py
    neighbor_handler.py
    relation_creator.py
    cache_manager.py

  # Predicate Matching subsystem
  matching/
    __init__.py
    predicate_matcher.py
    predicate_alignment.py
    alignment_cache.py

  # Data structures
  models/
    expansion_node.py
    expansion_context.py
    ...

  # Set KG (already well organized)
  set_knowledge_graph/
    ...
```

**Effort**: Medium (3-5 days)
**Risk**: Low (mostly moving files)

---

## ✅ Positive Patterns to Maintain

### Good Practices Already Present

1. **Registry Pattern**: Clean and reusable implementation in `src/utils/registry.py` (62 lines, well documented)

2. **Reader/Writer Separation**: The structure `src/core/dataset/reader/` and `writer/` correctly follows Factory pattern

3. **FilteringStage**: Excellent example of SRP - does ONE thing (filtering) and does it well (84 lines)

4. **Dataset/KnowledgeGraph**: Lean and focused classes (45 and 37 lines respectively)

5. **BasicBertUnit**: Well-structured PyTorch model (66 lines) with clear responsibility

6. **InteractionMLP**: Another good PyTorch model example (65 lines)

7. **Dataclass for configuration**: `ExperimentConfig` uses `@dataclass(frozen=True)` - good practice for immutability

8. **Stages well separated**: `ReductionStage`, `AugmentationStage`, `EvaluationStage`, `FilteringStage` are properly separated

---

## 📋 Action Plan (When Ready to Address)

### Phase 1 - Quick Wins (1-2 days)
- [ ] Remove `plm_augmenter_old.py` if no longer used
- [ ] Extract `_node_in_graph` to shared utility
- [ ] Create centralized `deep_merge` in `src/config/utils.py`

### Phase 2 - Medium Effort (3-5 days)
- [ ] Extract `ValueCacheNormalizer` from `NodeExpander`
- [ ] Extract `PredicateMatchingOrchestrator` from `NodeExpander`
- [ ] Refactor `_run_standard_mode` into smaller methods

### Phase 3 - Major Refactoring (1-2 weeks)
- [ ] Split `BartInterpolatorPLM` into 4+ classes
- [ ] Create `AugmentationRetryManager`
- [ ] Reorganize PLM module directory structure

### Phase 4 - Consolidation
- [ ] Add unit tests for newly extracted classes
- [ ] Update documentation
- [ ] Code review

---

## 📊 Priority Matrix

| Issue | Impact | Effort | Risk | Priority |
|-------|--------|--------|------|----------|
| BartInterpolatorPLM God Class | High | High | Medium | 🔴 Critical |
| NodeExpander Long Method | High | Medium | Medium | 🟠 High |
| ExperimentRunner Long Methods | Medium | Medium | Low | 🟠 High |
| PLMAugmenter Long Methods | Medium | Medium | Low | 🟠 High |
| Code Duplication | Low | Low | Very Low | 🟡 Medium |
| Config Duplication | Low | Low | Very Low | 🟡 Medium |
| PLM Structure | Low | Medium | Low | 🔵 Low |

---

## 🎯 Recommendation

**Current Status**: The codebase is functional and the recent `FilteringStage` refactoring demonstrates good architectural direction.

**Priority**: Address `BartInterpolatorPLM` first when time permits, as it's the most critical technical debt that impacts maintainability and testability.

**Approach**: Gradual refactoring with comprehensive tests to minimize regression risk.

---

## 📚 References

- **Recent Refactoring**: See `ARCHITECTURE_TRAINING_MODE.md` for example of clean stage-based architecture
- **Analysis Date**: 2025-12-15
- **Analysis Agent**: code-refactoring-architect (agent ID: abca2a3)
