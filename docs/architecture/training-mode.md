# Training Mode Architecture ✅

## Design Finale (Clean) - Stage-Based Architecture

### Principio Architetturale

**L'augmenter SEMPRE augmenta (ritorna SEMPRE original + synthetic). Il FilteringStage filtra in base a training_mode.**

## 🏗️ Architettura

```
Config (YAML)
    │
    ├─ NO "augmentation" section → Baseline Mode
    │     │
    │     └─> Runner: usa dataset originale (non chiama augmenter)
    │           Evaluation riceve: dataset_reduced
    │
    └─ HAS "augmentation" section → Augmentation Mode
          │
          ├─ Augmenter: SEMPRE genera original + synthetic
          │     │
          │     └─> FilteringStage decide cosa passare a evaluation:
          │           │
          │           ├─ training_mode: "augmented" (default)
          │           │     └─> Ritorna: original + synthetic
          │           │
          │           └─ training_mode: "synthetic_only"
          │                 └─> Ritorna: SOLO synthetic (rimuove original)
```

### Pipeline Completa

```
Dataset Originale
    │
    ├─> ReductionStage
    │       └─> dataset_reduced (baseline)
    │
    ├─> [SE augmentation configurata]
    │   │
    │   ├─> AugmentationStage
    │   │       └─> dataset_augmented (original + synthetic)
    │   │
    │   └─> FilteringStage (training_mode)
    │           ├─> "baseline": dataset_reduced
    │           ├─> "augmented": dataset_augmented (all pairs)
    │           └─> "synthetic_only": synthetic_pairs (original removed)
    │
    └─> EvaluationStage (riceve dataset filtrato)
```

## 📝 Configurazioni

### Mode 1: Baseline (No Augmentation)

```yaml
experiment:
  name: baseline_experiment
  dataset:
    name: openea/D_W_15K_V1
  reduction:
    method: random_entities
    ratio: 0.5
  # NO "augmentation" section ← Baseline mode
  model: bert_int
```

**Comportamento**:
- Runner vede che non c'è "augmentation" section
- Non istanzia `PLMAugmenter` né `FilteringStage`
- Passa `dataset_reduced` direttamente a evaluation
- EvaluationStage riceve: `dataset_reduced, dataset_reduced`

### Mode 2: Synthetic-only (Only Synthetic Data)

```yaml
experiment:
  name: synthetic_only_experiment
  dataset:
    name: openea/D_W_15K_V1
  reduction:
    method: random_entities
    ratio: 0.5
  augmentation:  # ← Augmentation section presente
    method: plm
    ratio: 1.0
    training_mode: synthetic_only  # ← Filtering rimuove coppie originali
  model: bert_int
```

**Comportamento**:
1. `AugmentationStage` chiama `PLMAugmenter.augment()`
   - Augmenter genera sintetici, ritorna `dataset_augmented` (original + synthetic)
2. `FilteringStage(training_mode="synthetic_only")` filtra:
   - Calcola: `synthetic_pairs = augmented_pairs - original_pairs`
   - Ritorna: `dataset_filtered` con SOLO synthetic pairs
3. EvaluationStage riceve: `dataset_reduced, dataset_filtered` (synthetic only)

### Mode 3: Augmented (Original + Synthetic)

```yaml
experiment:
  name: augmented_experiment
  dataset:
    name: openea/D_W_15K_V1
  reduction:
    method: random_entities
    ratio: 0.5
  augmentation:  # ← Augmentation section presente
    method: plm
    ratio: 1.0
    training_mode: augmented  # ← Default (può essere omesso)
  model: bert_int
```

**Comportamento**:
1. `AugmentationStage` chiama `PLMAugmenter.augment()`
   - Augmenter genera sintetici, ritorna `dataset_augmented` (original + synthetic)
2. `FilteringStage(training_mode="augmented")` filtra:
   - Ritorna: `dataset_augmented` senza modifiche (tutte le coppie)
3. EvaluationStage riceve: `dataset_reduced, dataset_augmented` (all pairs)

## 🔧 Implementazione

### PLMAugmenter (src/augmentation/methods/plm/plm_augmenter.py)

```python
class PLMAugmenter(AugmentationMethod):
    def __init__(self, config):
        # NO training_mode! Augmenter non si occupa più di filtering
        self.augmentation_cfg = config.get("augmentation", {})
        # ... resto dell'inizializzazione

    def augment(self, dataset: Dataset) -> Dataset:
        """SEMPRE augmenta e ritorna TUTTE le coppie (original + synthetic).

        Il filtering è responsabilità di FilteringStage, non dell'augmenter.
        """
        # Step 1: Fine-tune BART
        # Step 2: Generate synthetic pairs
        dataset_augmented = self._do_augmentation(dataset)

        # Ritorna TUTTE le coppie (original + synthetic)
        return dataset_augmented
```

### FilteringStage (experiments/runner/stages.py)

```python
class FilteringStage:
    """Stage separato per filtering basato su training_mode."""

    def __init__(self, training_mode: str):
        self.training_mode = training_mode
        # Valida: "baseline", "synthetic_only", o "augmented"

    def execute(
        self,
        dataset_original: Dataset,
        dataset_augmented: Dataset,
    ) -> Dataset:
        """Filtra dataset basato su training_mode."""

        if self.training_mode == "baseline":
            # Baseline: ritorna solo original (ignora augmented)
            return dataset_original

        elif self.training_mode == "augmented":
            # Augmented: ritorna tutte le coppie
            return dataset_augmented

        elif self.training_mode == "synthetic_only":
            # Synthetic-only: rimuove original pairs
            original_pairs = set(dataset_original.aligned_entities)
            augmented_pairs = set(dataset_augmented.aligned_entities)
            synthetic_pairs = augmented_pairs - original_pairs

            # Crea dataset filtrato con solo synthetic
            dataset_filtered = dataset_augmented.clone()
            dataset_filtered.aligned_entities = tuple(sorted(synthetic_pairs))
            return dataset_filtered
```

### ExperimentRunner (experiments/runner/runner.py)

```python
class ExperimentRunner:
    def __init__(self, exp_cfg):
        # Se config non ha "augmentation", self.augmentations è lista vuota
        augmentation_method = self.normalized_cfg.augmentation
        self.augmentations = [augmentation_method] if augmentation_method else []

    def _execute_ratio(self, ...):
        # Reduction stage
        dataset_reduced = reduction_stage.execute(...)

        # Baseline evaluation (se configurato)
        if self.reduction_eval:
            evaluation_stage.execute(
                "baseline",
                dataset_reduced,
                dataset_reduced,  # Stesso dataset per baseline
                ...
            )

        # Augmentation loop
        for aug_name in self.augmentations:
            # Step 1: Augmentation (genera SEMPRE original + synthetic)
            dataset_augmented = augmentation_stage.execute(...)

            # Step 2: Filtering (filtra basato su training_mode)
            training_mode = stage_cfg.get("augmentation", {}).get("training_mode", "augmented")
            filtering_stage = FilteringStage(training_mode)
            dataset_filtered = filtering_stage.execute(dataset_reduced, dataset_augmented)

            # Step 3: Evaluation (riceve dataset filtrato)
            evaluation_stage.execute(
                aug_name,
                dataset_reduced,
                dataset_filtered,  # Dataset dopo filtering
                ...
            )
```

## ✅ Vantaggi del Design

### 1. Single Responsibility Principle ⭐

Ogni componente ha UNA sola responsabilità:

- **Augmenter**: SOLO augmentation (genera synthetic pairs)
- **FilteringStage**: SOLO filtering (seleziona quali pairs usare)
- **Runner**: SOLO orchestrazione (coordina le fasi)
- **Config**: SOLO dichiarazione (specifica cosa vuoi)

### 2. Semantica Chiara

```python
# ✅ PERFETTO: ogni metodo fa esattamente quello che dice
augmenter.augment(dataset)         # → SEMPRE augmenta (original + synthetic)
filtering_stage.execute(...)       # → Filtra basato su mode
runner.execute()                   # → Orchestra tutto

# ❌ PRIMA (confuso): augment a volte non augmentava
augmenter.augment(dataset)         # → dataset originale se baseline???
```

### 3. Modularità e Riusabilità

```python
# FilteringStage è completamente indipendente e riusabile
filtering_stage = FilteringStage("synthetic_only")

# Può essere usato con qualsiasi augmenter (non solo PLM)
dataset_filtered = filtering_stage.execute(original, augmented)

# Facile testare in isolamento
assert len(dataset_filtered.aligned_entities) == expected_synthetic_count
```

### 4. Estensibilità

Facile aggiungere nuovi modi di filtering:
```python
class FilteringStage:
    def execute(self, original, augmented):
        if self.training_mode == "weighted_sample":
            # Sample by quality score
            return self._sample_by_quality(augmented)
        elif self.training_mode == "top_k":
            # Use only top-k highest quality pairs
            return self._select_top_k(augmented, k=1000)
```

### 5. Testabilità

```python
# Test augmenter in isolamento (sempre ritorna tutto)
augmenter = PLMAugmenter(config)
result = augmenter.augment(dataset)
assert len(result.aligned_entities) >= len(dataset.aligned_entities)

# Test filtering in isolamento
filtering_stage = FilteringStage("synthetic_only")
result = filtering_stage.execute(original, augmented)
assert all(pair not in original.aligned_entities for pair in result.aligned_entities)

# Test baseline (nessun augmenter chiamato)
runner = ExperimentRunner(config_without_augmentation)
runner.run()
# Verifica che augmenter non venga mai chiamato
```

## 🔄 Evoluzione del Design

### Iterazione 1: Modulo Separato ❌

```python
# PROBLEMA: API awkward, due step manuali
augmented = augmenter.augment(dataset)
filtered = filter_dataset_by_mode(dataset, augmented, "synthetic_only")
```

**Problema**: Utente deve manualmente chiamare due funzioni separate.

### Iterazione 2: Integrato con Early-Exit per Baseline ⚠️

```python
# PROBLEMA: Augmenter gestisce "baseline" (semantica sbagliata)
augmenter = PLMAugmenter({"training_mode": "baseline"})
result = augmenter.augment(dataset)  # → ritorna originale senza augmentare!
```

**Problema**: Viola SRP - augmenter che non augmenta quando `training_mode="baseline"`.

### Iterazione 3: Baseline nel Runner, Filtering nell'Augmenter ⚠️

```python
# MEGLIO: Baseline gestito dal runner (non istanzia augmenter)
# MA: Augmenter ancora responsabile di filtering
if config.has_augmentation:
    augmenter = PLMAugmenter({"training_mode": "synthetic_only"})
    result = augmenter.augment(dataset)  # → augmenta + filtra
else:
    result = dataset  # baseline
```

**Problema**: Augmenter ancora viola SRP - fa sia augmentation CHE filtering.

### Iterazione 4: Stage-Based Architecture ✅

```python
# PERFETTO: Responsabilità completamente separate
# Augmenter: SOLO augmentation
augmented = augmenter.augment(dataset)  # → SEMPRE original + synthetic

# FilteringStage: SOLO filtering
filtering_stage = FilteringStage("synthetic_only")
filtered = filtering_stage.execute(dataset, augmented)  # → SOLO synthetic

# Runner: SOLO orchestrazione
if config.has_augmentation:
    dataset_aug = augmentation_stage.execute(...)
    dataset_final = filtering_stage.execute(...)
else:
    dataset_final = dataset  # baseline
```

**Vantaggi**:
- ✅ Ogni classe ha UNA sola responsabilità
- ✅ Filtering è riusabile con altri augmenters
- ✅ Facile testare ogni componente in isolamento
- ✅ Semantica chiara: `augment()` SEMPRE augmenta

## 📊 Training Modes Table

| Mode | Config | Runner | Augmenter | FilteringStage | Output |
|------|--------|--------|-----------|----------------|--------|
| **Baseline** | No "augmentation" | Non istanzia augmenter/filtering | - | - | Dataset ridotto |
| **Synthetic-only** | `training_mode: synthetic_only` | Istanzia entrambi | Genera (original + synthetic) | Rimuove original | Solo synthetic |
| **Augmented** | `training_mode: augmented` | Istanzia entrambi | Genera (original + synthetic) | Passa tutto | Original + synthetic |

## 🎯 Best Practices

### DO ✅

```python
# Config per baseline: ometti "augmentation"
experiment:
  dataset: ...
  reduction: ...
  # NO augmentation section
  model: ...

# Config per synthetic_only: specifica training_mode
experiment:
  dataset: ...
  reduction: ...
  augmentation:
    training_mode: synthetic_only
  model: ...
```

### DON'T ❌

```python
# ❌ Non mettere training_mode: "baseline" nella config
# (non ha senso, baseline = niente augmentation)
augmentation:
  training_mode: baseline  # SBAGLIATO!

# ❌ Non mettere filtering logic nell'augmenter
class PLMAugmenter:
    def augment(self, dataset):
        augmented = self._do_augmentation(dataset)
        if self.training_mode == "synthetic_only":
            return self._filter(augmented)  # SBAGLIATO! Usa FilteringStage

# ❌ Non filtrare manualmente nel runner
dataset_aug = augmentation_stage.execute(...)
if training_mode == "synthetic_only":
    # filtra manualmente...  # SBAGLIATO! Usa FilteringStage
```

## 🔍 Code Locations

### PLMAugmenter

- **File**: `src/augmentation/methods/plm/plm_augmenter.py`
- **Key method**: `augment()` - Ritorna SEMPRE original + synthetic (no filtering)
- **Lines**: ~193-310 (augment method)

### FilteringStage

- **File**: `experiments/runner/stages.py`
- **Class**: `FilteringStage`
- **Lines**: ~383-467 (FilteringStage implementation)
- **Key method**: `execute(dataset_original, dataset_augmented) -> Dataset`

### ExperimentRunner

- **File**: `experiments/runner/runner.py`
- **Lines**:
  - 39: FilteringStage import
  - 78-79: augmentations list
  - 1074-1077: FilteringStage instantiation and execution (standard path)
  - 1234-1237: FilteringStage instantiation and execution (retry path)

### Configs

- **Baseline**: `config/experiments/massive/bert_int_baseline/*.yaml` (no augmentation section)
- **Synthetic-only**: `config/experiments/massive/bert_int_synthetic_only/*.yaml` (training_mode: synthetic_only)

## 📚 References

- `FAIR_COMPARISON_GUIDE.md` - Perché serve aug_ratio=1.0 per fair comparison
- `config/experiments/massive/README_QUALITY_EVALUATION.md` - Come eseguire esperimenti
- `examples/synthetic_comparison_example.py` - Esempio di uso

---

## 📝 Changelog

### 2025-12-15: Stage-Based Architecture (Iterazione 4)

**Cambiamenti**:
- ✅ Rimosso `training_mode` da `PLMAugmenter.__init__()`
- ✅ Rimosso `_apply_training_mode_filter()` da `PLMAugmenter`
- ✅ `PLMAugmenter.augment()` ora ritorna SEMPRE original + synthetic
- ✅ Creato `FilteringStage` separato in `experiments/runner/stages.py`
- ✅ Integrato `FilteringStage` nel runner (sia standard che retry path)
- ✅ Aggiornata documentazione

**Motivazione**:
- Pieno rispetto del Single Responsibility Principle
- Augmenter responsabile SOLO di augmentation
- FilteringStage responsabile SOLO di filtering
- Runner responsabile SOLO di orchestrazione
- Filtering riusabile con qualsiasi augmenter

---

**Design Status**: ✅ **Stage-Based Architecture** - Clean, modular, fully respects SOLID principles
