# Synthetic Data Quality Evaluation

Valutazione della qualità dei dati sintetici confrontando performance con solo dati sintetici vs solo dati reali.

## 🎯 Obiettivo

**Domanda centrale**: I dati sintetici possono sostituire i dati reali? Quanto è grande il quality gap?

Confrontiamo:
- **Baseline (Mode 1)**: Training con SOLO dati reali/originali
- **Synthetic-only (Mode 2)**: Training con SOLO dati sintetici/generati

## 📊 Setup Sperimentale

### Due Modalità di Training

1. **Baseline** (Mode 1 - Reference)
   - **Directory**: `bert_int_baseline/`, `rrea_baseline/`
   - **Training**: Solo coppie allineate originali (reali)
   - **Augmentation**: Disabilitato
   - **Config files**: 700 per model type
   - **Scopo**: Performance di riferimento con dati ground-truth

2. **Synthetic-only** (Mode 2 - Quality Test)
   - **Directory**: `bert_int_synthetic_only/`, `rrea_synthetic_only/`
   - **Training**: SOLO coppie generate (sintetiche), rimuove originali
   - **Augmentation**: Abilitato con `training_mode: synthetic_only`
   - **Config files**: 700 per model type
   - **Scopo**: Valutare se sintetici possono sostituire reali

## 📁 Struttura Directory

```
config/experiments/massive/
├── bert_int_baseline/          # 700 configs - Baseline (solo reali)
│   ├── BBC_DB_01_01.yaml
│   ├── D_W_15K_V1_01_01.yaml
│   └── ...
├── bert_int_synthetic_only/    # 700 configs - Synthetic-only (solo sintetici)
│   ├── BBC_DB_01_01.yaml
│   ├── D_W_15K_V1_01_01.yaml
│   └── ...
├── rrea_baseline/              # 700 configs - Baseline (solo reali)
│   └── ...
└── rrea_synthetic_only/        # 700 configs - Synthetic-only (solo sintetici)
    └── ...
```

**TOTALE**: 2800 configurazioni (1400 baseline + 1400 synthetic_only)

## 🎯 Metriche di Valutazione Qualità

### 1. Quality Gap (Primary Metric)

```
Quality Gap = Performance(Baseline) - Performance(Synthetic-only)
```

**Interpretazione**:
- **Gap < 5%**: Qualità ECCELLENTE - Sintetici possono sostituire i reali
- **Gap 5-15%**: Qualità BUONA - Sintetici mantengono pattern generali
- **Gap > 15%**: Qualità SCARSA - Problemi significativi nei sintetici
- **Gap < 0%**: ⚠️ ATTENZIONE - Possibili artifacts/shortcuts nei sintetici

### 2. Transferability Score

```
Transferability = Performance(Synthetic-only) / Performance(Baseline)
```

**Interpretazione**:
- **> 1.0**: ⚠️ Possibili artifacts (sintetici "troppo buoni")
- **0.95-1.0**: ✓ Eccellente trasferibilità
- **0.85-0.95**: ✓ Buona trasferibilità
- **< 0.85**: Problemi di trasferibilità

### 3. Relative Performance

```
Relative = (Synthetic-only / Baseline) × 100%
```

Percentuale di performance mantenuta dai sintetici rispetto ai reali.

## 🚀 Come Eseguire

### ⭐ Opzione Raccomandata: Fair Comparison (aug_ratio=1.0)

**Per valutare la qualità dei sintetici**, usa il flag `--fair-comparison`:

```bash
# Run FAIR comparison for bert_int (70 experiments: 7 datasets × 10 reduction ratios)
bash scripts/run_quality_evaluation.sh --model bert_int --fair-comparison --jobs 4

# Run FAIR comparison for rrea
bash scripts/run_quality_evaluation.sh --model rrea --fair-comparison --jobs 4
```

**Cosa fa `--fair-comparison`**:
- Esegue **SOLO** config con `aug_ratio=1.0` (pattern `*_*_10.yaml`)
- Garantisce stesso N di coppie: N reali vs N sintetici
- Confronto **non biased** dalla dimensione del dataset
- **70 esperimenti** per model invece di 700

**Esempio**:
```
reduction=0.3, aug_ratio=1.0
→ Baseline: 300 coppie reali
→ Synthetic-only: 300 coppie sintetiche (stesso N!)
→ Fair comparison ✓
```

### Opzione 2: Grid Completo (tutti aug_ratio)

```bash
# Run all baseline + synthetic_only experiments for bert_int (700 exp)
bash scripts/run_quality_evaluation.sh --model bert_int --jobs 4

# Run all baseline + synthetic_only experiments for rrea (700 exp)
bash scripts/run_quality_evaluation.sh --model rrea --jobs 4
```

**Quando usarlo**:
- Per esplorare l'effetto di diversi `aug_ratio`
- Per trovare l'`aug_ratio` ottimale
- Attenzione: confronto non fair se aug_ratio ≠ 1.0

### Opzione 3: Solo baseline o solo synthetic_only

```bash
# Run only baseline experiments
bash scripts/run_quality_evaluation.sh --model bert_int --baseline-only --jobs 4

# Run only synthetic_only experiments
bash scripts/run_quality_evaluation.sh --model bert_int --synthetic-only --jobs 4
```

### Opzione 4: Subset di esperimenti (pattern matching)

```bash
# Run only D_W_15K_V1 experiments
bash scripts/run_quality_evaluation.sh --model bert_int --pattern "D_W_15K_V1_*.yaml" --jobs 4

# Run only reduction ratio 0.5
bash scripts/run_quality_evaluation.sh --model bert_int --pattern "*_05_*.yaml" --jobs 4

# Run only specific augmentation ratios
bash scripts/run_quality_evaluation.sh --model bert_int --pattern "*_*_05.yaml" --jobs 4
```

### Opzione 5: Dry run (preview)

```bash
# Preview what will be executed without running
bash scripts/run_quality_evaluation.sh --model bert_int --dry-run

# Dry run with fair comparison
bash scripts/run_quality_evaluation.sh --model bert_int --fair-comparison --dry-run
```

### Opzione 6: Resume interrupted run

```bash
# Resume if experiments were interrupted
bash scripts/run_quality_evaluation.sh --model bert_int --resume --jobs 4
```

## 📊 Analisi Risultati

### Dopo il completamento

```bash
# Analyze quality gap and transferability
python experiments/statistics/compare_quality.py --model bert_int

# Generate LaTeX tables
python experiments/statistics/compare_quality.py --model bert_int --latex

# Detailed per-dataset breakdown
python experiments/statistics/compare_quality.py --model bert_int --detailed
```

### Output Atteso

Lo script di analisi calcolerà:

1. **Quality Gap per dataset**:
   ```
   Dataset          Baseline  Synthetic  Gap      Interpretation
   ================================================================
   D_W_15K_V1       0.750     0.735      -2.0%    EXCELLENT
   BBC_DB           0.680     0.620      -8.8%    GOOD
   ICEW_WIKI        0.590     0.450     -23.7%    POOR
   ```

2. **Transferability Score**:
   ```
   Dataset          Transfer  Interpretation
   ==========================================
   D_W_15K_V1       0.980     Excellent
   BBC_DB           0.912     Good
   ICEW_WIKI        0.763     Poor
   ```

3. **Quality vs Augmentation Ratio**:
   - Come varia il quality gap con aug_ratio (0.1 → 1.0)?
   - Esiste un ratio ottimale per massimizzare qualità?

## ⚖️ Fair Comparison: Perché aug_ratio=1.0?

### Il Problema del Confronto Unfair

```python
# Scenario SBAGLIATO (aug_ratio ≠ 1.0):
reduction_ratio = 0.3
aug_ratio = 0.5

Dataset originale: 1000 coppie
→ Dopo reduction: 300 coppie reali
→ Dopo augmentation: 150 coppie sintetiche

Baseline: 300 coppie reali
Synthetic-only: 150 coppie sintetiche  # ← METÀ del baseline!

# Se Synthetic-only performa peggio, è perché:
# - I sintetici sono di bassa qualità? ❌ Non possiamo dirlo
# - Oppure semplicemente hai MENO dati? ✅ Probabilmente questo!
```

**Problema**: Il confronto è **invalido scientificamente** - stai confrontando dataset di dimensioni diverse!

### La Soluzione: aug_ratio = 1.0

```python
# Scenario CORRETTO (aug_ratio = 1.0):
reduction_ratio = 0.3
aug_ratio = 1.0  # ← FISSO a 1.0

Dataset originale: 1000 coppie
→ Dopo reduction: 300 coppie reali
→ Dopo augmentation: 300 coppie sintetiche (1.0 × 300)

Baseline: 300 coppie reali
Synthetic-only: 300 coppie sintetiche  # ← STESSO numero!
Augmented: 600 coppie (300 + 300)

# Ora il confronto è FAIR:
# - Se Baseline > Synthetic-only → i sintetici sono di qualità inferiore
# - Se Baseline ≈ Synthetic-only → i sintetici sono di qualità simile
# - La differenza riflette QUALITÀ, non QUANTITÀ
```

### Quando NON usare Fair Comparison

Il flag `--fair-comparison` è specifico per valutare **qualità** dei sintetici.

Per altre domande di ricerca, usa il grid completo:

| Domanda | aug_ratio | Fair? | Scopo |
|---------|-----------|-------|-------|
| **Qualità sintetici** | **1.0 fisso** | ✅ | Valutare qualità (questa è la tua domanda!) |
| Aug_ratio ottimale | 0.1-1.0 variabile | ❌ | Trovare aug_ratio che massimizza performance |
| Data efficiency | Variabile | ❌ | Quanto valgono N sintetici vs M reali? |

### Riassunto

```bash
# Per valutare QUALITÀ dei sintetici (raccomandato):
bash scripts/run_quality_evaluation.sh --model bert_int --fair-comparison --jobs 4

# Per esplorare aug_ratio ottimale (opzionale):
bash scripts/run_quality_evaluation.sh --model bert_int --jobs 4
```

## 🔬 Domande di Ricerca

### Q1: I sintetici possono sostituire i reali?

**Metrica**: Quality Gap medio aggregato

```python
avg_gap = mean(Baseline - Synthetic_only) across all datasets
```

- Se `avg_gap < 5%`: SÌ, possono sostituire
- Se `avg_gap > 15%`: NO, troppo quality loss

### Q2: La qualità dipende dal dataset?

**Analisi**: Quality Gap per dataset

- Alcuni datasets hanno sintetici migliori?
- Patterns nei dataset che funzionano bene/male?

### Q3: La qualità scala con aug ratio?

**Analisi**: Quality Gap vs aug_ratio

```
Hypothesis: Più augmentation → più dati sintetici → migliore coverage?
```

Test:
- Confronta gap a aug_ratio=0.1 vs 0.5 vs 1.0
- Trova sweet spot

### Q4: La qualità dipende dalla quantità di dati reali?

**Analisi**: Quality Gap vs reduction_ratio

```
Hypothesis: Con pochi reali (reduction=0.1), i sintetici sono più importanti?
```

Test:
- Confronta gap a reduction_ratio basso vs alto
- I sintetici aiutano di più in low-data regime?

## 📈 Scenari di Interpretazione

### Scenario 1: Qualità Eccellente

```
Dataset: D_W_15K_V1
Baseline:       0.750 (Hits@1)
Synthetic-only: 0.745 (Hits@1)
Quality Gap:    -0.7% (< 5%)
Transferability: 0.993
```

**Interpretazione**:
- Dati sintetici di ECCELLENTE qualità
- Possono sostituire i reali senza loss significativo
- BART genera valori molto realistici

**Azione**: Usare synthetic_only in production per data augmentation

### Scenario 2: Qualità Buona

```
Dataset: BBC_DB
Baseline:       0.680
Synthetic-only: 0.620
Quality Gap:    -8.8% (5-15%)
Transferability: 0.912
```

**Interpretazione**:
- Dati sintetici di qualità BUONA
- Mantengono pattern generali ma perdono dettagli
- Ancora utili per augmentation (non replacement)

**Azione**: Usare augmented mode (reali + sintetici) invece di synthetic_only

### Scenario 3: Qualità Scarsa

```
Dataset: ICEW_WIKI
Baseline:       0.590
Synthetic-only: 0.450
Quality Gap:    -23.7% (> 15%)
Transferability: 0.763
```

**Interpretazione**:
- Dati sintetici di qualità SCARSA
- Loss significativo, problemi con BART
- Possibili cause: dataset complesso, multi-lingue, literals rari

**Azione**: Investigare cause, migliorare augmentation strategy

### Scenario 4: Sintetici Sospettosamente Buoni

```
Dataset: SUSPICIOUS_DATASET
Baseline:       0.700
Synthetic-only: 0.750
Quality Gap:    +7.1% (sintetici MEGLIO dei reali!)
Transferability: 1.071
```

**Interpretazione**:
- ⚠️ ATTENZIONE: Possibili artifacts o data leakage
- Sintetici "troppo buoni" suggeriscono shortcuts
- Possibile overfitting al test set

**Azione**:
- Verifica train/test split
- Analizza valori generati per patterns sospetti
- Verifica che BART non memorizzi test set

## 🎓 Struttura File Config

### Baseline Config Example

```yaml
experiment:
  name: D_W_15K_V1_05_05
  dataset:
    name: openea/D_W_15K_V1
    writer: bert_int
  reduction:
    method: random_entities
    ratio: 0.5
    writer: bert_int
    eval: true
    save_dataset: false
    save_model: false
  # NO augmentation section
  model: bert_int
  seed: 11037
  clear: true
  overwrite_existing: false
```

### Synthetic-only Config Example

```yaml
experiment:
  name: D_W_15K_V1_05_05
  dataset:
    name: openea/D_W_15K_V1
    writer: bert_int
  reduction:
    method: random_entities
    ratio: 0.5
    writer: bert_int
    eval: true
    save_dataset: false
    save_model: false
  augmentation:  # ← Augmentation enabled
    method: plm
    ratio: 0.5
    training_mode: synthetic_only  # ← KEY: Use ONLY synthetic
    writer:
      type: bert_int
      augmented_only_train: true
    eval: true
    save_dataset: false
    save_model: false
  model: bert_int
  seed: 11037
  clear: true
  overwrite_existing: false
```

**Differenza chiave**: `training_mode: synthetic_only` → rimuove coppie originali, tiene solo generate

## 📝 Note Importanti

1. **Seed fisso (11037)**: Garantisce riproducibilità
2. **Same test set**: Baseline e Synthetic-only usano STESSO test set
3. **Fairness**: Stesso reduction_ratio, stesso aug_ratio per confronto equo
4. **Training set diverso**:
   - Baseline: N coppie reali
   - Synthetic-only: M coppie sintetiche (M = N × aug_ratio)
5. **Early completion**: Esperimenti già completati vengono skippati

## ⏱️ Stima Tempi

### Per Esperimento
- **Fast datasets** (BBC_DB): ~15 min
- **Medium datasets** (D_W_15K): ~30 min
- **Large datasets** (SRPRS): ~60 min
- **Average**: ~30 min per experiment

### Totale

```
Single Model (700 baseline + 700 synthetic_only = 1400 experiments):
  Sequential: 1400 × 30min = 700 hours = 29 days
  Parallel (4 jobs): 700 hours ÷ 4 = 175 hours = 7.3 days

Both Models (bert_int + rrea):
  Parallel (4 jobs): ~15 days total
```

## 🐛 Troubleshooting

**Problema**: Synthetic-only ha 0 coppie di training
- **Causa**: Augmentation ratio troppo basso o augmentation fallita
- **Soluzione**: Verifica `ratio >= 0.1`, controlla log BART

**Problema**: Quality gap troppo grande (> 30%)
- **Causa**: BART non genera valori sensati per questo dataset
- **Soluzione**: Verifica literals nel dataset, considera fine-tuning BART

**Problema**: Synthetic > Baseline (sospetto)
- **Causa**: Possibili artifacts o data leakage
- **Soluzione**: Verifica train/test split, analizza valori generati

**Problema**: Esperimenti troppo lenti
- **Soluzione**: Aumenta `--jobs` se hai GPU memory disponibile

## 🚦 Workflow Consigliato

1. **Test setup** (1 config):
   ```bash
   bash scripts/run_quality_evaluation.sh --model bert_int \
     --pattern "D_W_15K_V1_05_05.yaml" --dry-run
   ```

2. **Pilot run** (1 dataset, tutte le combinazioni):
   ```bash
   bash scripts/run_quality_evaluation.sh --model bert_int \
     --pattern "D_W_15K_V1_*.yaml" --jobs 4
   ```

3. **Analisi pilot**:
   ```bash
   python experiments/statistics/compare_quality.py --model bert_int \
     --dataset D_W_15K_V1
   ```

4. **Se qualità OK, full run**:
   ```bash
   bash scripts/run_quality_evaluation.sh --model bert_int --jobs 4
   ```

5. **Analisi completa**:
   ```bash
   python experiments/statistics/compare_quality.py --model bert_int --latex
   ```

## 📊 Visualizzazioni Attese

Dopo l'analisi, potrai generare:

1. **Quality Gap Heatmap**:
   - X-axis: aug_ratio (0.1 → 1.0)
   - Y-axis: reduction_ratio (0.1 → 1.0)
   - Color: Quality Gap
   - Per dataset

2. **Transferability Plot**:
   - Scatter plot: Baseline (x) vs Synthetic-only (y)
   - Ideale: punti sulla linea y=x
   - Gap: distanza dalla linea

3. **Quality vs Aug Ratio**:
   - Line plot: aug_ratio (x) vs Quality Gap (y)
   - Per dataset
   - Trova optimal ratio

4. **Per-dataset Summary Table**:
   ```
   Dataset    | Baseline | Synthetic | Gap    | Transfer | Quality
   ================================================================
   D_W_15K_V1 | 0.750    | 0.745     | -0.7%  | 0.993    | EXCELLENT
   BBC_DB     | 0.680    | 0.620     | -8.8%  | 0.912    | GOOD
   ICEW_WIKI  | 0.590    | 0.450     |-23.7%  | 0.763    | POOR
   ```

## 🎯 Contributo Scientifico

Questo experimental setup permette di rispondere empiricamente a:

1. **Can synthetic data replace real data in Entity Alignment?**
   - Prima valutazione sistematica su larga scala
   - 7 datasets, 100 configurazioni per dataset

2. **How does synthetic data quality vary across datasets?**
   - Identificare caratteristiche dataset che beneficiano
   - Guidelines per quando usare synthetic data

3. **Optimal augmentation ratio for quality?**
   - Trade-off quantità vs qualità
   - Data efficiency analysis

## ✅ Checklist

- [x] Baseline configs generated (1400 files)
- [x] Synthetic-only configs generated (1400 files)
- [x] Execution script created
- [x] Documentation written
- [ ] Dry-run test passed
- [ ] Pilot run successful
- [ ] Full run initiated
- [ ] Quality analysis completed
- [ ] LaTeX tables generated
- [ ] Visualizations created
