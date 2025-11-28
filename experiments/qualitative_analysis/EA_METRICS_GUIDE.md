# Entity Alignment-Specific Metrics Guide

Queste sono le 6 metriche **specifiche per Entity Alignment** che abbiamo aggiunto per valutare la qualità delle entità sintetiche generate da PLM augmentation.

## 📊 Le 6 Metriche EA-Specific

### 1. **Alignment Preservation Score** 🎯
**Cosa misura**: Le coppie sintetiche rimangono allineabili?

**Idea**: Se (A, B) sono entità allineate, e generiamo (A', B'), allora A' dovrebbe essere più simile a B' che a qualsiasi altra entità random.

**Come funziona**:
```python
Per ogni coppia sintetica (A', B'):
1. Calcola similarità sim(A', B') - dovrebbe essere alta
2. Calcola similarità sim(A', random_entity) - dovrebbe essere bassa
3. Se sim(A', B') > sim(A', random) → PRESERVED ✓
```

**Interpretazione**:
- **>70%**: Ottimo! Le coppie sintetiche mantengono l'allineabilità
- **50-70%**: Accettabile, ma potrebbe migliorare
- **<50%**: ⚠️ Problema! Le entità sintetiche perdono l'allineamento

**Esempio concreto**:
```
Originale:
  Paris_DBP ←→ Q90_Wikidata

Sintetico:
  Berlin_DBP ←→ Q64_Wikidata (Berlin)

✓ PRESERVED: sim(Berlin_DBP, Q64) = 0.82 > sim(Berlin_DBP, random_Q123) = 0.31
```

---

### 2. **Structural Consistency Score** 🏗️
**Cosa misura**: La struttura del KG è preservata?

**Idea**: Le entità sintetiche devono avere la stessa "forma" delle originali (stessi tipi di attributi, stesse frequenze).

**Come funziona**:
```python
Originali:                    Sintetici:
- name: 100%                  - name: 95%       ✓
- birthDate: 80%              - birthDate: 75%  ✓
- occupation: 60%             - occupation: 5%  ✗ (PROBLEMA!)
```

Calcola:
1. **Jaccard similarity** tra set di predicati
2. **KL divergence** tra distribuzioni di frequenza
3. **Differenza** nel numero medio di attributi

**Interpretazione**:
- **>70%**: Struttura ben preservata
- **50-70%**: Struttura parzialmente preservata
- **<50%**: ⚠️ Struttura KG non mantenuta

**Perché è importante**:
Se le entità originali hanno sempre `birthDate + birthPlace` insieme, anche le sintetiche dovrebbero averli!

---

### 3. **Predicate Co-occurrence Preservation** 🔗
**Cosa misura**: I pattern di co-occorrenza degli attributi sono mantenuti?

**Idea**: Se nel dataset originale `name` e `birthDate` appaiono insieme nel 95% delle entità, nei sintetici dovrebbe essere simile.

**Come funziona**:
```python
Originali:
- (name, birthDate): co-occur in 95% entities
- (name, occupation): co-occur in 80% entities
- (birthDate, deathDate): co-occur in 30% entities

Sintetici:
- (name, birthDate): co-occur in 92% entities  ✓ Simile
- (name, occupation): co-occur in 85% entities ✓ Simile
- (birthDate, deathDate): co-occur in 5% entities ✗ Molto diverso!
```

Calcola **cosine similarity** tra matrici di co-occorrenza.

**Interpretazione**:
- **>60%**: Pattern preservati bene
- **40-60%**: Pattern parzialmente preservati
- **<40%**: ⚠️ Pattern persi

---

### 4. **Cross-KG Style Consistency** 🎨
**Cosa misura**: Ogni KG mantiene il suo "stile" distintivo?

**Idea**: DBpedia e Wikidata hanno stili diversi. Le entità sintetiche devono mantenere questi stili.

**Esempio**:
```
DBpedia style:
- "Paris, France"
- "Albert Einstein, physicist"

Wikidata style:
- "Paris (city in France)"
- "Albert Einstein (theoretical physicist)"

✓ Le entità sintetiche di DBpedia devono somigliare allo stile DBpedia
✓ Le entità sintetiche di Wikidata devono somigliare allo stile Wikidata
```

**Come funziona**:
```python
1. Similarità within-KG (DBpedia_synth vs DBpedia_synth) = 0.75  ← Alto
2. Similarità cross-KG (DBpedia_synth vs Wikidata_synth) = 0.48  ← Più basso
3. Style Score = within-KG - cross-KG = 0.75 - 0.48 = 0.27
```

**Interpretazione**:
- **>0.2**: Stili ben distinti ✓
- **0.1-0.2**: Stili parzialmente distinti
- **<0.1**: ⚠️ Stili collassati (troppo simili)

---

### 5. **Nearest Neighbor Distance Ratio (NNDR)** 📏
**Cosa misura**: Bilancio tra diversità e realismo.

**Idea**: Le entità sintetiche dovrebbero essere "tra" le originali:
- Non troppo vicine (altrimenti sono solo copie → bassa diversity)
- Non troppo lontane (altrimenti sono hallucinations → basso realism)

**Come funziona**:
```python
Per ogni entità sintetica S:

1. Distanza al vicino originale più vicino: d_orig = 0.25
2. Distanza al vicino sintetico più vicino: d_synth = 0.30

3. NNDR = d_orig / d_synth = 0.25 / 0.30 = 0.83
```

**Interpretazione**:
- **0.8-1.2**: Perfetto! Bilancio ideale
- **0.5-0.8 o 1.2-1.5**: Accettabile
- **<0.5**: ⚠️ Troppo simili agli originali (bassa diversity)
- **>1.5**: ⚠️ Troppo lontane dagli originali (possibili hallucinations)

**Visualizzazione**:
```
Originali:        O    O    O    O    O
                   |    |    |    |    |
Sintetici:          S    S    S    S      ← Perfetto! Intercalati

Troppo vicini:    O S  O S  O S  O S      ← Bassa diversity
Troppo lontani:   O    O    O    O    SSSS ← Possibili hallucinations
```

---

### 6. **Alignment Model Performance Gain** 🏆
**Cosa misura**: Le entità sintetiche migliorano le performance del modello EA?

**Idea**: Questa è la **metrica finale definitiva**. Se l'augmentation funziona, il modello EA addestrato su `Original + Synthetic` deve performare meglio di quello addestrato solo su `Original`.

**Come funziona**:
```python
# Baseline (senza augmentation)
hits@1_baseline = train_EA_model(original_only)  # Es: 0.75

# Con augmentation
hits@1_augmented = train_EA_model(original + synthetic)  # Es: 0.82

# Performance gain
gain = (hits@1_augmented - hits@1_baseline) / hits@1_baseline
     = (0.82 - 0.75) / 0.75
     = 0.093  (9.3% improvement!)
```

**Interpretazione**:
- **>5%**: 🎉 Augmentation funziona! Metodo valido!
- **0-5%**: 👍 Piccolo beneficio, potrebbe migliorare
- **Negativo**: ⚠️ Augmentation danneggia! Dati sintetici di bassa qualità

**Questa è la metrica CHE CONTA per il paper!**

---

## 🎯 Riassunto: Cosa Significano le Metriche

| Metrica | Risponde alla domanda | Target | Critico? |
|---------|----------------------|--------|----------|
| **Alignment Preservation** | Le coppie sintetiche sono ancora allineate? | >70% | ⭐⭐⭐ |
| **Structural Consistency** | La struttura KG è preservata? | >70% | ⭐⭐ |
| **Co-occurrence Preservation** | I pattern di attributi sono mantenuti? | >60% | ⭐⭐ |
| **Style Consistency** | Ogni KG mantiene il suo stile? | >0.2 | ⭐ |
| **NNDR** | C'è bilancio diversity/realism? | 0.8-1.2 | ⭐⭐ |
| **Performance Gain** | Il modello EA migliora? | >5% | ⭐⭐⭐⭐⭐ |

---

## 🚀 Come Usarle

### Analisi Completa
```bash
python -m experiments.qualitative_analysis.ea_specific_metrics \
    --original data/raw/openea/BBC_DB \
    --augmented results/BBC_DB_01_05/augmentation \
    --output ea_metrics.json
```

### Report Integrato (include tutte le metriche)
```bash
bash scripts/analyze_quality.sh \
    data/raw/openea/BBC_DB \
    results/BBC_DB_01_05/augmentation \
    results/quality_analysis
```

Il report markdown includerà automaticamente tutte le 6 metriche EA!

### Da Python
```python
from experiments.qualitative_analysis import analyze_ea_metrics

metrics = analyze_ea_metrics(
    original_path="data/raw/openea/BBC_DB",
    augmented_path="results/BBC_DB_01_05/augmentation"
)

print(f"Alignment Preservation: {metrics['alignment_preservation_score']:.2%}")
print(f"Structural Consistency: {metrics['structural_consistency_score']:.2%}")
print(f"NNDR: {metrics['nndr_mean']:.2f}")
```

---

## 📚 Confronto con Letteratura

| Paper | Metriche Usate | DAKGEA Coverage |
|-------|---------------|----------------|
| **GAN** (Goodfellow 2014) | Inception Score | ❌ Solo immagini |
| **BootEA** (Sun 2018) | Hits@1 on pseudo-labels | ✅ Nostra metrica #6 |
| **MixGCF** (Huang 2021) | Link prediction accuracy | ⚠️ Diverso contesto (KG completion) |
| **TMix** (Chen 2020) | Text classification F1 | ⚠️ Adattabile come #6 |
| **DAKGEA** (nostro) | 6 EA-specific + downstream | ✅ **Più completo!** |

**Contributo unico**: Siamo i primi a proporre metriche **specifiche per Entity Alignment augmentation**!

---

## 🎓 Per il Paper

Nel paper, dovresti includere:

1. ✅ **Sezione "Evaluation Metrics"**:
   - Subsection: "General Quality Metrics" (diversity + realism)
   - Subsection: "EA-Specific Metrics" (le nostre 6!)
   - Subsection: "Downstream Task Evaluation" (performance gain)

2. ✅ **Tabella comparativa**:
```latex
\begin{table}
\caption{Quality Metrics for Synthetic Entities}
\begin{tabular}{lll}
\toprule
Category & Metric & Target \\
\midrule
Diversity & Novelty Ratio & 40-70\% \\
          & Embedding Diversity & 0.3-0.6 \\
Realism & Fluency Rate & >80\% \\
        & Attribute Validity & >90\% \\
EA-Specific & Alignment Preservation & >70\% \\
            & Structural Consistency & >70\% \\
            & NNDR & 0.8-1.2 \\
Downstream & Performance Gain (Hits@1) & >5\% \\
\bottomrule
\end{tabular}
\end{table}
```

3. ✅ **Sezione "Results"**:
   - Report tutti i valori
   - Confronta con baseline (no augmentation)
   - Mostra che metriche EA sono soddisfatte
   - **Performance gain è positivo!**

---

## 💡 Tips per Interpretare i Risultati

**Scenario 1: Tutte metriche alte + Performance gain positivo**
→ 🎉 Augmentation eccellente! Pubblica!

**Scenario 2: Metriche alte MA Performance gain negativo**
→ ⚠️ Dati sintetici sono "di qualità" ma non utili per EA. Indaga!

**Scenario 3: Alignment Preservation basso**
→ 💡 Aumenta peso predicate matching, controlla `match_attr` files

**Scenario 4: NNDR troppo basso (<0.5)**
→ 💡 Aumenta `base_alpha` e `temperature` per più diversity

**Scenario 5: NNDR troppo alto (>1.5)**
→ 💡 Riduci `base_alpha` e `temperature` per più conservatività

---

Ora hai un framework completo per valutare la qualità delle entità sintetiche in modo scientifico! 🚀
