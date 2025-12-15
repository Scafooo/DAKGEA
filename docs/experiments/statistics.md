# Experiment Statistics

This directory contains scripts for analyzing experiment results and generating statistics in LaTeX format.

## Analysis Tools

### Full Statistics Analysis (`analyze_results.sh`)

Comprehensive analysis with visualizations, statistics, and exports including LaTeX comparison tables.

```bash
# Run with default settings (generates TSV + LaTeX exports)
bash scripts/analyze_results.sh

# Basic mode (TSV only, no advanced features)
bash scripts/analyze_results.sh --basic

# Full mode (includes complete LaTeX document with figures)
bash scripts/analyze_results.sh --full
```

**Generated Outputs:**

By default (or with `--export-formats tsv latex`):
- **TSV exports**: `dataset_summary.tsv`, `ratio_summary.tsv`
- **LaTeX exports**:
  - `latex/dataset_summary.tex`: Single table with all datasets
  - `latex/comparison_tables/`: **Aggregated tables** (one per dataset)
    - Side-by-side baseline vs augmented comparison
    - Values shown as **mean±std** across multiple seeds
    - Color-coded improvements (green) and degradations (red)
    - All reduction ratios 0.1-1.0 (10 rows per table)
  - `latex/comparison_tables_detailed/`: **Detailed tables** (one per dataset per ratio)
    - Individual experiment values (not aggregated)
    - One row per seed/experiment
    - Direct comparison of baseline vs augmented for each seed
    - **Gap column (Δ)** showing the difference between augmented and baseline
    - Combined files `{dataset}_all_detailed.tex` for easy inclusion of all ratios

With `--full` mode:
- All of the above, plus:
  - `latex/results_document.tex`: Complete LaTeX document ready to compile

### Output Format

Two types of comparison tables are generated:

#### 1. Aggregated Tables (`comparison_tables/`)

One table per dataset (e.g., `BBC_DB.tex`, `D_W_15K_V1.tex`):

**Table structure:**
- **Rows**: All reduction ratios from 0.1 to 1.0 (always shown, even if data is missing)
- **Columns**: Side-by-side Base/Aug for each metric
  - **Baseline (Reduction)**: H@1, H@5, H@10, MRR, MR, P, R, F1
  - **Augmented (PLM)**: H@1, H@5, H@10, MRR, MR, P, R, F1

**Statistics:**
- Values shown as **mean ± std** across multiple seeds
- **Best values in bold** for easy comparison (baseline vs augmented)
- Percentages for Hits@K, Precision, Recall, F1 (multiplied by 100)
- Decimals for MRR (4 decimals)
- Float for MR (1 decimal)
- **Missing data**: N/A in gray for ratios without experiments

#### 2. Detailed Tables (`comparison_tables_detailed/`)

One table per dataset per reduction ratio (e.g., `BBC_DB_ratio0.1_detailed.tex`, `BBC_DB_ratio0.5_detailed.tex`):

**Table structure:**
- **Rows**: Individual experiments/seeds (e.g., BBC_DB_01_01, BBC_DB_01_02, ...)
- **Columns**: Three columns per metric (Base / Aug / Δ)
  - **Base**: Baseline value for this experiment
  - **Aug**: Augmented value for this experiment
  - **Δ (Gap)**: Difference (Aug - Base) with color coding (green if improvement, red if degradation)
- **Values**: Individual experiment values (not aggregated)
- **Best values in bold** for easy comparison (baseline vs augmented)
- Useful for inspecting variance and outliers across seeds

### Example Output

#### Aggregated Table (mean±std across seeds)

```latex
\begin{table}[H]
\centering
\scriptsize
\caption{Results for D\_W\_15K\_V1. Values shown as mean$\pm$std across multiple seeds.}
\label{tab:d_w_15k_v1}
\resizebox{1.1\textwidth}{!}{%  ← Automatically scales to page width
\begin{tabular}{c|cc|cc|...}
\hline
\multirow{2}{*}{\textbf{Ratio}} & \multicolumn{2}{c|}{H@1} & \multicolumn{2}{c|}{H@5} & ... \\
 & \textit{Base} & \textit{Aug} & \textit{Base} & \textit{Aug} & ... \\
\hline
\textbf{0.1} & 45.23$\pm$1.12 & \cellcolor{green!15}\textbf{48.34$\pm$1.05} & 72.45$\pm$0.89 & ... \\
\textbf{0.2} & 52.34$\pm$0.98 & \cellcolor{green!15}\textbf{54.89$\pm$0.91} & 78.12$\pm$0.67 & ... \\
\textbf{0.3} & \textcolor{gray}{N/A} & \textcolor{gray}{N/A} & \textcolor{gray}{N/A} & ... \\
\textbf{0.4} & \textbf{58.45$\pm$1.05} & \cellcolor{red!15}57.12$\pm$1.23 & 82.34$\pm$0.78 & ... \\
...
\textbf{1.0} & 78.90$\pm$0.45 & \cellcolor{green!15}\textbf{82.34$\pm$0.52} & 91.23$\pm$0.34 & ... \\
\hline
\end{tabular}
}% End resizebox
\end{table}
```

#### Detailed Table (individual experiments)

```latex
\begin{table}[H]
\centering
\scriptsize
\caption{Detailed results for D\_W\_15K\_V1 at reduction ratio 0.1. Individual experiment values.}
\label{tab:d_w_15k_v1_ratio0.1_detailed}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l|ccc|ccc|...}
\hline
\multirow{2}{*}{\textbf{Experiment}} & \multicolumn{3}{c|}{H@1} & \multicolumn{3}{c|}{H@5} & ... \\
 & \textit{Base} & \textit{Aug} & \textit{$\Delta$} & \textit{Base} & \textit{Aug} & \textit{$\Delta$} & ... \\
\hline
\texttt{D\_W\_15K\_V1\_01\_01} & 44.50 & \cellcolor{green!15}\textbf{47.80} & \textcolor{green!70!black}{+3.30} & 71.20 & \cellcolor{green!15}\textbf{74.50} & \textcolor{green!70!black}{+3.30} & ... \\
\texttt{D\_W\_15K\_V1\_01\_02} & 45.80 & \cellcolor{green!15}\textbf{48.90} & \textcolor{green!70!black}{+3.10} & 73.10 & \cellcolor{green!15}\textbf{76.20} & \textcolor{green!70!black}{+3.10} & ... \\
\texttt{D\_W\_15K\_V1\_01\_03} & 44.90 & \cellcolor{green!15}\textbf{48.20} & \textcolor{green!70!black}{+3.30} & 72.80 & \cellcolor{green!15}\textbf{76.10} & \textcolor{green!70!black}{+3.30} & ... \\
...
\texttt{D\_W\_15K\_V1\_01\_10} & 46.10 & \cellcolor{green!15}\textbf{49.30} & \textcolor{green!70!black}{+3.20} & 72.90 & \cellcolor{green!15}\textbf{76.10} & \textcolor{green!70!black}{+3.20} & ... \\
\hline
\end{tabular}
}% End resizebox
\end{table}
```

**Key features:**

*Aggregated tables:*
- **All ratios 0.1-1.0** always shown (10 rows per table)
- Values as **mean±std** across seeds
- **Best values in bold** (baseline or augmented, whichever is better)
- **Green** background when augmented > baseline (improvement)
- **Red** background when augmented < baseline (degradation)
- **Gray N/A** for missing experiments

*Detailed tables:*
- One row per individual experiment/seed
- No aggregation - raw experiment values
- **Gap column (Δ)** showing difference between augmented and baseline
- **Best values in bold** for each experiment
- **Gap color-coded**: green text for improvements, red for degradations
- Same color coding as aggregated tables for augmented values
- Useful for identifying outliers and variance patterns

*Both:*
- Light colors (15% opacity) for readability
- Side-by-side baseline vs augmented comparison
- **Forced positioning with [H]** to prevent LaTeX from moving tables

### Including in LaTeX Document

**IMPORTANT**: Add these packages to your LaTeX preamble (all are required):
```latex
\usepackage{multirow}  % For multi-row headers
\usepackage{xcolor}    % For cell colors (REQUIRED for color coding)
\usepackage{colortbl}  % For \cellcolor command (REQUIRED for highlighting)
\usepackage{graphicx}  % For \resizebox (REQUIRED to fit wide tables)
\usepackage{float}     % For [H] placement (REQUIRED to force table position)
\usepackage{booktabs}  % For dataset_summary.tex (professional tables)
```

**What the ± symbol means**:
- Values are shown as **mean ± std** (standard deviation)
- Example: `45.50±0.50` means average value is 45.50 with std deviation of 0.50
- Calculated across multiple experiment seeds (typically 10 seeds per ratio)

Include tables in your document:
```latex
% Complete summary table (all datasets)
\input{results_analysis/latex/dataset_summary.tex}

% Aggregated comparison tables (mean±std per dataset)
\input{results_analysis/latex/comparison_tables/D_W_15K_V1.tex}
\input{results_analysis/latex/comparison_tables/BBC_DB.tex}

% Detailed tables - Option 1: Include all ratios for a dataset at once (RECOMMENDED)
\input{results_analysis/latex/comparison_tables_detailed/D_W_15K_V1_all_detailed.tex}
\input{results_analysis/latex/comparison_tables_detailed/BBC_DB_all_detailed.tex}

% Detailed tables - Option 2: Include individual ratios separately
\input{results_analysis/latex/comparison_tables_detailed/D_W_15K_V1_ratio0.1_detailed.tex}
\input{results_analysis/latex/comparison_tables_detailed/D_W_15K_V1_ratio0.5_detailed.tex}
\input{results_analysis/latex/comparison_tables_detailed/BBC_DB_ratio0.1_detailed.tex}
```

**Note on combined files**: Each `{dataset}_all_detailed.tex` file includes all 10 reduction ratios (0.1-1.0) for that dataset, making it easy to include all detailed tables with a single `\input{}` command.

**Recommendation**: Use aggregated tables in the main paper, and detailed tables in the appendix for full transparency.

**Testing the tables**:
- See `docs/DAKGEA_report/latex_table_example.tex` for a complete working example
- Compile with: `pdflatex latex_table_example.tex`
- This example includes all required packages and shows how to use the tables

**How wide tables are handled**:
- All tables use `\resizebox{\textwidth}{!}{...}` to automatically scale to page width
- This prevents tables from overflowing page margins
- Requires `\usepackage{graphicx}` in your preamble
- The `!` maintains aspect ratio while scaling

**Common errors**:
- ❌ **Error: Undefined control sequence \cellcolor** → Missing `\usepackage{colortbl}`
- ❌ **Error: Undefined control sequence \textcolor** → Missing `\usepackage{xcolor}`
- ❌ **Error: Undefined control sequence \multirow** → Missing `\usepackage{multirow}`
- ❌ **Error: Undefined control sequence \resizebox** → Missing `\usepackage{graphicx}`
- ❌ **Error: Unknown float option 'H'** → Missing `\usepackage{float}`

**Note:** Augmented results are colored:
- **Light green** background: Improvement over baseline
- **Light red** background: Degradation from baseline

## Metrics Explanation

- **H@1, H@5, H@10**: Hits@1, Hits@5, Hits@10 (percentage of correct predictions in top-1, top-5, top-10)
- **MRR**: Mean Reciprocal Rank (average of 1/rank for correct predictions)
- **MR**: Mean Rank (average rank of correct prediction)
- **P**: Precision (true positives / predicted positives)
- **R**: Recall (true positives / actual positives)
- **F1**: F-measure (harmonic mean of precision and recall)

## Directory Structure

```
experiments/statistics/
├── README.md                          # This file
├── analyze_results.py                 # Main analysis script
├── comparison_tables.py               # Standalone LaTeX table generator (legacy)
├── latex_document.py                  # LaTeX document builder
├── exporters.py                       # Export utilities (includes comparison tables)
├── visualizations.py                  # Plotting functions
├── advanced_stats.py                  # Statistical analysis
└── config.py                          # Configuration

results_analysis/                      # Generated by analyze_results.sh
├── dataset_summary.tsv                # TSV export
├── ratio_summary.tsv                  # Ratio-based TSV export
└── latex/                             # LaTeX exports
    ├── dataset_summary.tex            # Single summary table
    ├── results_document.tex           # Complete LaTeX document (with --export-formats latex-doc)
    ├── comparison_tables/             # Aggregated tables (mean±std)
    │   ├── BBC_DB.tex
    │   ├── D_W_15K_V1.tex
    │   ├── D_W_15K_V2.tex
    │   ├── ICEW_WIKI.tex
    │   ├── ICEW_YAGO.tex
    │   ├── SRPRS_D_W_15K_V1.tex
    │   └── SRPRS_D_W_15K_V2.tex
    └── comparison_tables_detailed/    # Detailed tables (individual values)
        ├── BBC_DB_all_detailed.tex        # Combined file (all ratios)
        ├── BBC_DB_ratio0.1_detailed.tex
        ├── BBC_DB_ratio0.2_detailed.tex
        ├── ...
        ├── BBC_DB_ratio1.0_detailed.tex
        ├── D_W_15K_V1_all_detailed.tex    # Combined file (all ratios)
        ├── D_W_15K_V1_ratio0.1_detailed.tex
        ├── D_W_15K_V1_ratio0.2_detailed.tex
        ├── ...
        └── D_W_15K_V1_ratio1.0_detailed.tex
```

## Requirements

- Python 3.7+
- NumPy (for statistics computation)
- Completed experiment runs with results.json files

## Experiment Name Format

The script expects experiment names in format: `{DATASET}_{RATIO_IDX}_{SEED_IDX}`

Examples:
- `BBC_DB_01_01` → BBC_DB dataset, ratio=0.1, seed=1
- `D_W_15K_V1_05_07` → D_W_15K_V1 dataset, ratio=0.5, seed=7
- `ICEW_WIKI_10_10` → ICEW_WIKI dataset, ratio=1.0, seed=10

**Ratio mapping**: 01→0.1, 02→0.2, ..., 10→1.0

## Troubleshooting

**No results found:**
- Ensure experiments have been run and completed
- Check that `results.json` files exist in evaluation directories
- Verify experiment directory path

**Missing metrics:**
- Some experiments may not have all metrics
- Missing metrics default to 0.0±0.0
- Check experiment logs for evaluation errors

**Large tables:**
- Tables with all metrics are wide and may need `\resizebox` or `\scriptsize`
- Consider splitting into separate tables for different metric groups
- Use landscape orientation with `\usepackage{pdflscape}` and `\begin{landscape}`
