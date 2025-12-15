# LaTeX Export e Surface Plot 3D - Guida Rapida

## Panoramica

Il modulo statistics ora supporta:
- ✅ **Esportazione tabelle LaTeX** con formattazione condizionale (colori per delta positivo/negativo)
- ✅ **Generazione documenti LaTeX completi** con tabelle e figure
- ✅ **Surface plot 3D** (statici e interattivi) per visualizzare metriche

## Uso Rapido

### 1. Esportare solo tabelle LaTeX

```bash
python3 experiments/statistics/analyze_results.py \
  --export-formats latex
```

**Output**: `results/statistics/latex/dataset_summary.tex`

### 2. Generare documento LaTeX completo

```bash
python3 experiments/statistics/analyze_results.py \
  --export-formats latex-doc \
  --enable-advanced-plots
```

**Output**:
- `results/statistics/latex/dataset_summary.tex` (tabella)
- `results/statistics/latex/results_document.tex` (documento completo)

### 3. Generare plot 3D

```bash
python3 experiments/statistics/analyze_results.py \
  --enable-advanced-plots
```

**Output**:
- `results/statistics/surface_3d/*.png` (plot statici)
- `results/statistics/surface_3d_interactive/*.html` (plot interattivi, se plotly è installato)

## Esempi Completi

### Esempio 1: Esportazione completa con tutti i grafici

```bash
python3 experiments/statistics/analyze_results.py \
  --export-formats tsv latex latex-doc \
  --enable-advanced-plots \
  --advanced-stats \
  --dpi 400 \
  --metrics hits@1 hits@5 mrr
```

Questo comando genera:
- File TSV per analisi con Excel/Python
- Tabelle LaTeX formattate
- Documento LaTeX compilabile
- Tutti i grafici avanzati (boxplot, violin, heatmap, 3D surface, ecc.)

### Esempio 2: Solo un dataset specifico

```bash
python3 experiments/statistics/analyze_results.py \
  --datasets BBC_DB \
  --export-formats latex-doc \
  --enable-advanced-plots
```

### Esempio 3: Filtrare per ratio

```bash
python3 experiments/statistics/analyze_results.py \
  --reduction-ratios 0.5 0.7 \
  --augmentation-ratios 1.0 1.5 2.0 \
  --export-formats latex-doc
```

## Compilare i documenti LaTeX

### Prerequisiti

Assicurati di avere LaTeX installato:

```bash
# Ubuntu/Debian
sudo apt-get install texlive-latex-extra texlive-fonts-recommended

# Fedora
sudo dnf install texlive-scheme-medium

# macOS
brew install --cask mactex
```

### Compilazione

```bash
cd results/statistics/latex
pdflatex results_document.tex
pdflatex results_document.tex  # Eseguilo due volte per i riferimenti
```

Oppure usa `latexmk` per compilazione automatica:

```bash
latexmk -pdf results_document.tex
```

## Caratteristiche delle Tabelle LaTeX

### Formattazione Condizionale

Le tabelle LaTeX con colori automatici evidenziano:
- 🟢 **Verde**: Delta positivo (miglioramento)
- 🔴 **Rosso**: Delta negativo (peggioramento)

Esempio di output:

```latex
\textcolor{green!70!black}{+0.0379}  % Miglioramento
\textcolor{red!70!black}{-0.0200}     % Peggioramento
```

### Personalizzazione

Puoi personalizzare:
- **Caption**: `caption="Mia Tabella"`
- **Label**: `label="tab:my_table"`
- **Allineamento colonne**: `col_spec="lrrcc"`
- **Posizione**: `position="htbp"`
- **Font size**: `small=True`

## Surface Plot 3D

I plot 3D mostrano come variano le metriche al variare dei ratio di reduction e augmentation.

### Caratteristiche

**Plot Statici (Matplotlib)**:
- PNG ad alta risoluzione (configurabile con `--dpi`)
- Contour projection sul piano base
- Colorbar per interpretazione valori
- Angolo di visione ottimizzato

**Plot Interattivi (Plotly)**:
- File HTML con rotazione/zoom interattivi
- Ideali per presentazioni
- Richiede: `pip install plotly`

### Interpretazione

- **Asse X**: Reduction Ratio
- **Asse Y**: Augmentation Ratio
- **Asse Z**: Valore metrica (es. hits@1, MRR)
- **Colore**: Intensità del valore

I picchi indicano le combinazioni ottimali di reduction/augmentation ratio.

## Struttura File Generati

```
results/statistics/
├── latex/
│   ├── dataset_summary.tex          # Tabella LaTeX
│   └── results_document.tex         # Documento completo
├── surface_3d/
│   ├── BBC_DB_hitsat1_reduction_surface3d.png
│   ├── BBC_DB_hitsat1_augmentation_surface3d.png
│   └── ...
├── surface_3d_interactive/
│   ├── BBC_DB_hitsat1_augmentation_surface3d.html
│   └── ...
├── heatmaps/
├── boxplots/
├── violins/
└── ... (altri tipi di grafici)
```

## API Python

### Usare le funzioni direttamente

```python
from pathlib import Path
from experiments.statistics.exporters import write_latex_table_colored
from experiments.statistics.latex_document import LaTeXDocument

# Creare una tabella con colori
headers = ["Dataset", "Metric", "Delta", "Δ%"]
rows = [
    ["BBC_DB", "hits@1", "+0.0379", "+4.44%"],
    ["Test", "hits@1", "-0.0200", "-2.22%"],
]

write_latex_table_colored(
    path=Path("output.tex"),
    headers=headers,
    rows=rows,
    caption="Risultati Esperimenti",
    label="tab:results",
)

# Creare un documento completo
doc = LaTeXDocument(title="I Miei Risultati")
doc.add_section("Introduzione", level=1)
doc.add_text("Questo documento presenta...")
doc.add_table_inline(headers, rows, caption="Sommario")
doc.write(Path("documento.tex"))
```

## Test

Esegui i test per verificare l'installazione:

```bash
python3 experiments/statistics/test_latex_export.py
```

Questo creerà file di esempio in `/tmp` che puoi compilare per verificare che tutto funzioni.

## Risoluzione Problemi

### Errore: `! LaTeX Error: File 'booktabs.sty' not found`

**Soluzione**: Installa i pacchetti LaTeX mancanti:
```bash
sudo apt-get install texlive-latex-extra
```

### Nessun plot 3D generato

**Soluzione**: Assicurati di usare `--enable-advanced-plots` e che esistano dati con vari ratio.

### Plot interattivi non generati

**Soluzione**: Installa plotly:
```bash
pip install plotly
# oppure
.venv/bin/pip install plotly
```

## Documentazione Completa

Per maggiori dettagli, consulta:
- `docs/LATEX_EXPORT.md` - Documentazione completa
- `docs/METRICS.md` - Metriche disponibili
- `config/statistics.yaml` - Configurazione

## Changelog

### v1.0.0 (2025-12-01)
- ✨ Aggiunta esportazione tabelle LaTeX con colori condizionali
- ✨ Generatore documenti LaTeX completi
- ✨ Surface plot 3D statici (matplotlib)
- ✨ Surface plot 3D interattivi (plotly)
- ✨ Funzione `save_figure()` per multi-formato export
- 📝 Documentazione completa
- ✅ Test suite per LaTeX export
