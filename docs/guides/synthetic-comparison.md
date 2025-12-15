# Massive Synthetic vs Real Data Comparison

Experimental setup per confrontare l'efficacia di dati sintetici vs reali su larga scala.

## 📊 Setup Sperimentale

### Due Modalità di Training (Massive)

1. **Baseline** (Mode 1)
   - **Directory**: `bert_int_baseline/`, `rrea_baseline/`
   - **Training**: Solo coppie allineate originali (reali)
   - **Augmentation**: Disabilitato
   - **Config files**: 700 per model type
   - **Scopo**: Performance baseline con dati reali

2. **Augmented** (Mode 3)
   - **Directory**: `bert_int_aug_red/`, `rrea_aug_red/`
   - **Training**: Coppie originali + generate (mix)
   - **Augmentation**: Abilitato con augmented_only_train=true
   - **Config files**: 700 per model type
   - **Scopo**: Performance standard augmentation (use case reale)

**Note**: Mode 2 (Synthetic-only) non è incluso in questo setup massive.

## 📁 Struttura Directory

```
config/experiments/massive/
├── bert_int_baseline/      # 700 configs - Baseline mode (bert_int)
│   ├── BBC_DB_01_01.yaml
│   ├── D_W_15K_V1_01_01.yaml
│   └── ...
├── bert_int_aug_red/       # 700 configs - Augmented mode (bert_int)
│   ├── BBC_DB_01_01.yaml
│   ├── D_W_15K_V1_01_01.yaml
│   └── ...
├── rrea_baseline/          # 700 configs - Baseline mode (rrea)
│   ├── BBC_DB_01_01.yaml
│   ├── D_W_15K_V1_01_01.yaml
│   └── ...
└── rrea_aug_red/           # 700 configs - Augmented mode (rrea)
    ├── BBC_DB_01_01.yaml
    ├── D_W_15K_V1_01_01.yaml
    └── ...
```

## 🎯 Parametri Sperimentali

### Datasets (7 total)
- BBC_DB
- D_W_15K_V1
- D_W_15K_V2
- ICEW_WIKI
- ICEW_YAGO
- SRPRS_D_W_15K_V1
- SRPRS_D_W_15K_V2

### Reduction Ratios (10 levels)
- 01 → 0.1 (10% dei dati)
- 02 → 0.2 (20% dei dati)
- ...
- 10 → 1.0 (100% dei dati)

### Augmentation Ratios (10 levels)
- 01 → 0.1 (10% augmentation)
- 02 → 0.2 (20% augmentation)
- ...
- 10 → 1.0 (100% augmentation)

### Models (2 types)
- bert_int
- rrea

### Total Configurations
- 7 datasets × 10 reduction ratios × 10 aug ratios = 700 configs per mode
- 700 baseline + 700 augmented = 1400 configs per model
- 2 models × 1400 = **2800 total configurations**

## 🚀 Come Generare i Config

### Generazione Baseline Configs

I config baseline sono già generati. Per rigenerarli:

```bash
python scripts/generate_massive_baseline_configs.py
```

Questo crea:
- `config/experiments/massive/bert_int_baseline/` (700 files)
- `config/experiments/massive/rrea_baseline/` (700 files)

I config augmented (`bert_int_aug_red/`, `rrea_aug_red/`) esistono già.

## 🏃 Come Eseguire gli Esperimenti

### Opzione 1: Tutti gli esperimenti per un modello

```bash
# Run all baseline + augmented experiments for bert_int
bash scripts/run_massive_synthetic_comparison.sh --model bert_int --jobs 4

# Run all baseline + augmented experiments for rrea
bash scripts/run_massive_synthetic_comparison.sh --model rrea --jobs 4
```

### Opzione 2: Solo baseline o solo augmented

```bash
# Run only baseline experiments
bash scripts/run_massive_synthetic_comparison.sh --model bert_int --baseline-only --jobs 4

# Run only augmented experiments
bash scripts/run_massive_synthetic_comparison.sh --model bert_int --augmented-only --jobs 4
```

### Opzione 3: Subset di esperimenti (pattern matching)

```bash
# Run only D_W_15K_V1 experiments
bash scripts/run_massive_synthetic_comparison.sh --model bert_int --pattern "D_W_15K_V1_*.yaml"

# Run only experiments with reduction ratio 0.5
bash scripts/run_massive_synthetic_comparison.sh --model bert_int --pattern "*_05_*.yaml"

# Run only specific augmentation ratios
bash scripts/run_massive_synthetic_comparison.sh --model bert_int --pattern "*_*_10.yaml"
```

### Opzione 4: Dry run (preview)

```bash
# Preview what will be executed without running
bash scripts/run_massive_synthetic_comparison.sh --model bert_int --dry-run
```

### Opzione 5: Resume interrupted run

```bash
# Resume if experiments were interrupted
bash scripts/run_massive_synthetic_comparison.sh --model bert_int --resume
```

## 📊 Analisi Risultati

### Dopo il completamento

Dopo che tutti gli esperimenti sono completati, esegui l'analisi comparativa:

```bash
# Compare baseline vs augmented for bert_int
python experiments/statistics/compare_baseline_augmented.py --model bert_int

# Generate LaTeX tables
python experiments/statistics/compare_baseline_augmented.py --model bert_int --latex
```

### Metriche Calcolate

Per ogni coppia (dataset, reduction_ratio, aug_ratio):

1. **Quality Gap**: `Performance(Baseline) - Performance(Augmented)`
   - Positivo: Baseline migliore (augmentation dannosa)
   - Negativo: Augmented migliore (augmentation utile)
   - Vicino a 0: Nessun effetto

2. **Augmentation Benefit**: `Performance(Augmented) - Performance(Baseline)`
   - > 10%: Beneficio FORTE
   - 5-10%: Beneficio MODERATO
   - < 5%: Beneficio DEBOLE
   - < 0%: Augmentation dannosa

3. **Relative Improvement**: `(Augmented - Baseline) / Baseline × 100%`

## 🔬 Domande di Ricerca

### Q1: L'augmentation migliora sempre?
Confronta performance baseline vs augmented per ogni dataset/ratio.

### Q2: Quanto beneficio porta l'augmentation?
Calcola improvement medio aggregato per dataset.

### Q3: L'augmentation scala con la quantità di dati?
Analizza trend al variare di reduction_ratio (0.1 → 1.0).

### Q4: L'augmentation è più utile con pochi dati?
Confronta beneficio a ratio basso (0.1-0.3) vs alto (0.7-1.0).

### Q5: Quanto augmentation è ottimale?
Analizza performance al variare di aug_ratio (0.1 → 1.0).

## 📈 Interpretazione Risultati

### Scenario 1: Augmentation sempre benefica
```
Dataset: D_W_15K_V1
All ratios: Augmented > Baseline
```
**Interpretazione**: Augmentation consistentemente utile, usala sempre.

### Scenario 2: Augmentation solo con pochi dati
```
Dataset: BBC_DB
Ratio 0.1-0.3: Augmented >> Baseline (+15%)
Ratio 0.7-1.0: Augmented ≈ Baseline (+2%)
```
**Interpretazione**: Augmentation più utile in low-data regime.

### Scenario 3: Augmentation dannosa
```
Dataset: ICEW_WIKI
Most ratios: Baseline > Augmented (-5%)
```
**Interpretazione**: Dati sintetici di bassa qualità o introducono noise.

### Scenario 4: Augmentation ratio ottimale
```
Dataset: D_W_15K_V2, Reduction 0.5
Aug 0.3: +12%
Aug 0.6: +15% ← PEAK
Aug 1.0: +10%
```
**Interpretazione**: Troppa augmentation può degradare, trovare sweet spot.

## 🔧 Parametri di Esecuzione

### Jobs (parallelismo)
```bash
--jobs 2  # Default: 2 jobs in parallel (safe for single GPU)
--jobs 4  # 4 jobs in parallel (requires more GPU memory)
```

**Raccomandazione**:
- Single GPU (12-16GB): `--jobs 2`
- Single GPU (24GB+): `--jobs 4`
- Multi-GPU: Usa GPU diversi per baseline e augmented

### Timeout
```bash
--timeout 7200  # Default: 2 hours per experiment
--timeout 14400 # 4 hours for larger datasets
```

### GPU
```bash
--gpu-id 0  # Use GPU 0 (default)
--gpu-id 1  # Use GPU 1
```

## 🎓 Struttura File Config

### Baseline Config Example
```yaml
experiment:
  name: D_W_15K_V1_01_01
  dataset:
    name: openea/D_W_15K_V1
    writer: bert_int
  reduction:
    method: random_entities
    ratio: 0.1
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

### Augmented Config Example
```yaml
experiment:
  name: D_W_15K_V1_01_01
  dataset:
    name: openea/D_W_15K_V1
    writer: bert_int
  reduction:
    method: random_entities
    ratio: 0.1
    writer: bert_int
    eval: true
    save_dataset: false
    save_model: false
  augmentation:  # ← Key difference
    method: plm
    ratio: 0.1
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

## 📝 Note Importanti

1. **Seed fisso (11037)**: Garantisce riproducibilità tra baseline e augmented
2. **Same test set**: Tutti gli esperimenti usano stesso test set per fairness
3. **Early completion**: Gli esperimenti già completati vengono skippati automaticamente
4. **Disk space**: 2800 experiments → planifica ~500GB di storage per risultati
5. **Time estimate**:
   - ~30min per experiment (medio)
   - 700 experiments ÷ 4 parallel jobs = 175 batches
   - 175 × 30min = ~88 hours = 3.5 giorni per model
   - Totale (2 models): ~7 giorni

## 🐛 Troubleshooting

**Problema**: Esperimenti troppo lenti
- **Soluzione**: Aumenta `--jobs` se hai GPU memory disponibile

**Problema**: GPU out of memory
- **Soluzione**: Riduci `--jobs` a 1 o 2

**Problema**: Esperimenti falliscono
- **Soluzione**: Verifica logs in `results/logs/`, usa `--resume` per riprendere

**Problema**: Voglio fermare e riprendere
- **Soluzione**: CTRL+C per fermare, poi usa `--resume` per riprendere

**Problema**: Troppo tempo per tutti gli esperimenti
- **Soluzione**: Usa `--pattern` per eseguire subset (es. solo alcuni datasets)

## 🚦 Workflow Consigliato

1. **Test rapido** (verifica setup):
   ```bash
   bash scripts/run_massive_synthetic_comparison.sh --model bert_int \
     --pattern "D_W_15K_V1_01_01.yaml" --dry-run
   ```

2. **Pilot run** (1 dataset, tutte le ratios):
   ```bash
   bash scripts/run_massive_synthetic_comparison.sh --model bert_int \
     --pattern "D_W_15K_V1_*.yaml" --jobs 4
   ```

3. **Full run** (tutti i datasets):
   ```bash
   bash scripts/run_massive_synthetic_comparison.sh --model bert_int --jobs 4
   ```

4. **Analisi risultati**:
   ```bash
   python experiments/statistics/compare_baseline_augmented.py --model bert_int --latex
   ```

5. **Ripeti per altro modello**:
   ```bash
   bash scripts/run_massive_synthetic_comparison.sh --model rrea --jobs 4
   ```
