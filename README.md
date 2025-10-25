# Data Augmentation for Knowledge Graph Entity Resolution

This project provides a **modular pipeline** for **data augmentation** and **dataset reduction** to support **Entity Alignment (EA)** tasks on Knowledge Graphs. It allows experimenting with multiple EA models and evaluating the impact of reduction and augmentation on alignment performance.

---

## ⚙️ Configuration

Configurations are managed with **YAML files**:

- `config/global.yaml` → global settings (paths, seed, logging)  
- `config/models/*.yaml` → model-specific parameters (e.g., HybEA, KnowFormer, BERT-INT)  
- `config/experiments/*.yaml` → experiment-specific settings (dataset subsets, augmentation/reduction parameters)  

All relative paths are automatically resolved by the Python loader (`src/config/loader.py`).

---

## 🛠️ Installation

Clone the repository:

```bash
git clone <PROJECT_URL>
cd <PROJECT_NAME>
```

Create a virtual environment and install dependencies:

```bash
conda env create -f install/HybEA_env.yml
# or
pip install -r install/requirements.txt
```

---

## 🚀 Running Experiments

The full pipeline follows this flow:

1. **Reduction** → shrink dataset size  
2. **Augmentation** → apply data augmentation on reduced datasets  
3. **Train & Evaluate EA models** → HybEA, KnowFormer, BERT-INT  
4. **Metrics & Analysis** → evaluate performance and gap between reduced, and augmented datasets  

Example command:

```bash
python experiments/run.py --config_exp config/experiments/exp_1.yaml --model hybea
```

---

## 📊 Results

All metrics, analyses, and experiment outcomes are saved in:

```
experiments/results/
```

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🔗 References

- [HybEA GitHub](https://github.com/fanourakis/HybEA)  
- [KnowFormer](https://arxiv.org/abs/XXXX.XXXX)  
- [BERT-INT](https://arxiv.org/abs/XXXX.XXXX)
