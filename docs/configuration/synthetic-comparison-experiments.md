# Synthetic vs Real Data Comparison Experiments

Experimental setup per confrontare l'efficacia di dati sintetici vs reali nell'addestramento di modelli di Entity Alignment.

## 📊 Setup Sperimentale

### Tre Modalità di Training

1. **Baseline** (`baseline.yaml`)
   - **Training**: Solo coppie allineate originali (reali)
   - **Augmentation**: Disabilitato
   - **Scopo**: Performance baseline con dati reali

2. **Synthetic-only** (`synthetic_only.yaml`)
   - **Training**: SOLO coppie allineate generate (sintetiche)
   - **Augmentation**: Abilitato, rimuove coppie originali
   - **Scopo**: Valutare qualità assoluta dei dati sintetici

3. **Augmented** (`augmented.yaml`)
   - **Training**: Coppie originali + generate (mix)
   - **Augmentation**: Abilitato, mantiene tutto
   - **Scopo**: Performance standard augmentation (use case reale)

## 🎯 Domande di Ricerca

### Q1: Qualità dei Dati Sintetici
**Metrica**: Quality Gap = Performance(Baseline) - Performance(Synthetic-only)

- **Gap < 5%**: Sintetici di qualità ECCELLENTE
- **Gap 5-15%**: Sintetici di qualità BUONA
- **Gap > 15%**: Sintetici hanno problemi significativi

### Q2: Beneficio dell'Augmentation
**Metrica**: Aug Benefit = Performance(Augmented) - Performance(Baseline)

- **> 10%**: Beneficio FORTE
- **5-10%**: Beneficio MODERATO
- **< 5%**: Beneficio DEBOLE
- **< 0%**: Augmentation dannosa

### Q3: Transferability
**Metrica**: Transfer Score = Performance(Synthetic-only) / Performance(Baseline)

- **> 1.0**: ⚠️ ATTENZIONE: Possibili artifacts/shortcuts
- **0.95-1.0**: ✓ Sintetici si trasferiscono bene
- **< 0.95**: Gap di trasferibilità

## 🚀 Come Eseguire

### Opzione 1: Singolo Esperimento

```bash
# Baseline
python -m src.main --config config/experiments/synthetic_comparison/baseline.yaml

# Synthetic-only
python -m src.main --config config/experiments/synthetic_comparison/synthetic_only.yaml

# Augmented
python -m src.main --config config/experiments/synthetic_comparison/augmented.yaml
```

### Opzione 2: Tutti gli Esperimenti + Confronto

```bash
# Esegui tutti e tre gli esperimenti e confronta risultati
python experiments/run_synthetic_comparison.py

# Solo dry-run (mostra comandi senza eseguire)
python experiments/run_synthetic_comparison.py --dry-run

# Solo alcuni esperimenti
python experiments/run_synthetic_comparison.py --experiments baseline synthetic_only

# Solo confronto (assume esperimenti già eseguiti)
python experiments/run_synthetic_comparison.py --compare-only
```

## 📁 Struttura File

```
config/experiments/synthetic_comparison/
├── README.md           # Questo file
├── baseline.yaml       # Config baseline (solo reali)
├── synthetic_only.yaml # Config synthetic-only (solo sintetici)
└── augmented.yaml      # Config augmented (reali + sintetici)

src/augmentation/
└── training_mode_filter.py  # Logica per filtrare dataset

experiments/
└── run_synthetic_comparison.py  # Script per eseguire confronto

results/synthetic_comparison/
├── baseline/           # Risultati baseline
├── synthetic_only/     # Risultati synthetic-only
└── augmented/          # Risultati augmented
```

## 🔧 Integrazione nel Codice

**Nota**: Il `training_mode` è ora integrato direttamente in `PLMAugmenter`. Non serve più un filtro separato!

### Modo Semplice (Raccomandato)

```python
from src.core.dataset import Dataset
from src.augmentation.methods.plm.plm_augmenter import PLMAugmenter

# 1. Load dataset
dataset = Dataset.load("D_W_15K_V1")

# 2. Configure augmenter with training_mode
config = {
    "augmentation": {
        "method": "plm",
        "ratio": 1.0,
        "training_mode": "synthetic_only"  # ← KEY: baseline, synthetic_only, or augmented
    }
}

# 3. Augment (filtering happens automatically)
augmenter = PLMAugmenter(config)
dataset_final = augmenter.augment(dataset)

# 4. Train model
model.train(dataset_final, config["training"])
```

### Training Modes

- **`baseline`**: Returns only original pairs (no augmentation applied)
- **`synthetic_only`**: Returns only synthetic pairs (removes originals)
- **`augmented`**: Returns all pairs (original + synthetic) [default]

## 📈 Interpretazione Risultati

### Scenario 1: Synthetic-only ≈ Baseline
```
Baseline: 0.75
Synthetic: 0.74
Gap: -1.3%
```
**Interpretazione**: Sintetici di ECCELLENTE qualità, possono sostituire i reali!

### Scenario 2: Synthetic-only < Baseline
```
Baseline: 0.75
Synthetic: 0.65
Gap: -13.3%
```
**Interpretazione**: Sintetici di qualità inferiore, mantengono pattern generali ma perdono dettagli.

### Scenario 3: Synthetic-only > Baseline
```
Baseline: 0.75
Synthetic: 0.80
Gap: +6.7%
```
**Interpretazione**: ⚠️ ATTENZIONE! Possibili artifacts nei sintetici (shortcuts che funzionano solo su test set simile).

### Scenario 4: Augmented > Baseline significativamente
```
Baseline: 0.75
Augmented: 0.85
Benefit: +13.3%
```
**Interpretazione**: Augmentation molto efficace, i sintetici aggiungono valore ai reali.

## 🔬 Analisi Avanzate

### Curva di Apprendimento
Esegui esperimenti con diverse `reduction_ratio`:

```yaml
# baseline_10.yaml
reduction_ratio: 0.1  # 10% dei dati

# baseline_50.yaml
reduction_ratio: 0.5  # 50% dei dati

# baseline_100.yaml
reduction_ratio: 1.0  # 100% dei dati
```

Confronta: Come scala la performance con quantità di dati (reali vs sintetici)?

### Data Efficiency
Confronta:
- Baseline con N esempi reali
- Augmented con N reali + N sintetici
- Baseline con 2N esempi reali

Domanda: È meglio augmentare o raccogliere più dati reali?

## ⚙️ Parametri Chiave

### augmentation.ratio
```yaml
ratio: 0.5  # Genera 50% di coppie in più
ratio: 1.0  # Genera stesso numero di coppie (raddoppia dataset)
ratio: 2.0  # Genera doppio delle coppie originali (triplica dataset)
```

### augmentation.training_mode
```yaml
training_mode: "baseline"        # Solo originali
training_mode: "synthetic_only"  # Solo sintetici
training_mode: "augmented"       # Originali + sintetici (default)
```

## 📝 Note Importanti

1. **Seed fisso**: Usa sempre lo stesso seed per riproducibilità
2. **Test set identico**: Tutti gli esperimenti usano STESSO test set
3. **Metriche consistenti**: Valuta sempre su stesse metriche (Hits@1, Hits@10, MRR)
4. **Multiple runs**: Esegui con seed diversi e fai media per robustezza

## 🎓 Paper di Riferimento

L'approccio è ispirato a:
- **Fagin et al. (2023)**: Value sets per entity resolution
- **Data augmentation literature**: Valutare qualità dati sintetici
- **Transfer learning**: Transferability from synthetic to real

## 🐛 Troubleshooting

**Problema**: `synthetic_only` ha 0 coppie!
- **Causa**: Augmentation fallita o ratio troppo basso
- **Soluzione**: Verifica `ratio >= 0.1` e controlla log BART

**Problema**: Performance identiche tra modi
- **Causa**: Augmentation non ha effetto
- **Soluzione**: Aumenta `ratio`, verifica che BART generi valori diversi

**Problema**: Synthetic >> Baseline (sospetto)
- **Causa**: Possibili artifacts o data leakage
- **Soluzione**: Verifica train/test split, analizza valori generati
