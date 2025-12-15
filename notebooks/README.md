# DAKGEA Tutorial Notebooks

Questa directory contiene notebook Jupyter interattivi per imparare ad usare DAKGEA.

## 📚 Notebooks Disponibili

### 1. Introduction (`01_introduction.ipynb`)
- Introduzione a DAKGEA
- Setup e verifica installazione
- Panoramica del framework
- Struttura del progetto

**Tempo stimato**: 5 minuti

### 2. Simple Experiment (`02_simple_experiment.ipynb`)
- Primo esperimento baseline
- Configurazione YAML
- Esecuzione e analisi risultati
- Comprensione delle metriche

**Tempo stimato**: 10-15 minuti (+ tempo esecuzione)

### 3. Data Augmentation (`03_data_augmentation.ipynb`)
- Aggiungere augmentation con PLM
- Confronto baseline vs augmented
- Interpretazione dei miglioramenti
- Tuning dei parametri

**Tempo stimato**: 15-20 minuti (+ tempo esecuzione)

### 4. Analyze Results (`04_analyze_results.ipynb`) [TODO]
- Analisi dettagliata dei risultati
- Generazione grafici
- Statistiche avanzate
- Confronto multi-configurazione

### 5. Advanced Configuration (`05_advanced_configuration.ipynb`) [TODO]
- Suite di esperimenti
- Configurazioni massive
- Esecuzione parallela
- Best practices

## 🚀 Come Iniziare

### Prerequisiti

1. **Installare Jupyter**:
```bash
pip install jupyter notebook
# oppure
pip install jupyterlab
```

2. **Verificare installazione DAKGEA**:
```bash
python -c "from experiments.runner.runner import ExperimentRunner; print('✓ OK')"
```

### Avviare i Notebook

#### Opzione A: Jupyter Notebook
```bash
cd notebooks
jupyter notebook
```

#### Opzione B: JupyterLab
```bash
cd notebooks
jupyter lab
```

#### Opzione C: VSCode
1. Installa l'estensione "Jupyter" in VSCode
2. Apri un file `.ipynb`
3. Seleziona il kernel Python del virtualenv

### Kernel Python

Assicurati di usare il Python del virtualenv di DAKGEA:

```bash
# Attiva il virtualenv
source ../.venv/bin/activate  # Linux/Mac
# oppure
..\.venv\Scripts\activate     # Windows

# Avvia Jupyter dal virtualenv
jupyter notebook
```

## 📖 Ordine Consigliato

Segui i notebook in ordine numerato:

```
01_introduction.ipynb         ← Inizia qui
    ↓
02_simple_experiment.ipynb    ← Primo esperimento
    ↓
03_data_augmentation.ipynb    ← Aggiungi augmentation
    ↓
04_analyze_results.ipynb      ← Analisi avanzata [TODO]
    ↓
05_advanced_configuration.ipynb  ← Configurazioni massive [TODO]
```

## 💡 Tips

### Esecuzione più Veloce

Per ridurre i tempi di esecuzione durante il tutorial:

1. **Usa dataset piccoli**: `D_W_15K_V1` è una buona scelta
2. **Riduci ratio**: Usa `reduction.ratio: 0.1` invece di 0.3
3. **GPU**: Assicurati di avere CUDA disponibile per BERT-INT

### Modificare i Notebook

I notebook sono interattivi - sentiti libero di:
- Modificare i parametri di configurazione
- Sperimentare con dataset diversi
- Aggiungere celle per analisi personalizzate
- Salvare le tue versioni modificate

### Problemi Comuni

#### "Module not found"
```python
# Aggiungi questa cella all'inizio del notebook
import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))
```

#### "CUDA out of memory"
```yaml
# Riduci il batch size nel config del modello
# o usa CPU (più lento)
```

#### Esperimento già esiste
```yaml
# Nel config, cambia:
overwrite_existing: true
# oppure cambia il nome dell'esperimento
```

## 📚 Risorse Aggiuntive

- **Documentazione**: `../docs/index.md`
- **Getting Started Guide**: `../docs/guides/getting-started.md`
- **Config Examples**: `../config/experiments/`
- **Scripts**: `../scripts/`

## 🆘 Supporto

Se hai problemi:

1. Verifica i prerequisiti (Python 3.11+, dipendenze installate)
2. Consulta la documentazione in `../docs/`
3. Controlla gli esempi in `../config/experiments/`
4. Apri un issue su GitHub

## ✅ Completamento

Dopo aver completato i notebook, sarai in grado di:

- ✓ Configurare ed eseguire esperimenti DAKGEA
- ✓ Usare data reduction e augmentation
- ✓ Analizzare e interpretare i risultati
- ✓ Ottimizzare i parametri per il tuo caso d'uso
- ✓ Eseguire esperimenti massive in parallelo

Buon apprendimento! 🚀
