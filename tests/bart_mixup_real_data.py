import torch
import random
import yaml
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from transformers import (
    BartForConditionalGeneration, 
    BartTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from transformers.modeling_outputs import BaseModelOutput
from datasets import Dataset as HFDataset
from rdflib import URIRef, Literal

# Aggiunta del progetto al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.core.dataset import Dataset
from src.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# 1. CARICAMENTO CONFIGURAZIONE E DATASET
# ------------------------------------------------------------------

def load_project_config():
    config_path = PROJECT_ROOT / "config" / "augmentation" / "plm.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_real_dataset():
    print("Caricamento dataset BBC_DB...")
    reader = OpeneaDatasetReader()
    # Usiamo il path relativo alla root del progetto
    dataset = reader.read("data/raw/openea/BBC_DB")
    return dataset

# ------------------------------------------------------------------
# 2. SETUP MODELLO CON SPECIAL TOKENS
# ------------------------------------------------------------------

class BartMixupEngine:
    def __init__(self, model_name: str, predicates: List[str], device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        
        # Registrazione predicati come Special Tokens
        self.special_tokens = [f"<{p.split('/')[-1]}>") for p in predicates]
        self.tokenizer.add_tokens(self.special_tokens)
        
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Mappa per recupero rapido
        self.pred_to_token = dict(zip(predicates, self.special_tokens))

    def get_token(self, pred_uri: str) -> str:
        return self.pred_to_token.get(pred_uri, "<attribute>")

# ------------------------------------------------------------------
# 3. COSTRUZIONE DATASET DAE + IDENTITY (Dati Reali)
# ------------------------------------------------------------------

from difflib import SequenceMatcher

def string_sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

def prepare_training_data(dataset: Dataset, engine: BartMixupEngine, max_pairs=2000):
    """
    Estrae coppie reali usando:
    1. Attribute Matching (corrispondenze tra predicati diversi)
    2. Proximity Matching (accoppiamento dei valori più simili per evitare rumore)
    """
    kg_src = dataset.knowledge_graph_source
    kg_tgt = dataset.knowledge_graph_target
    aligned = dataset.aligned_entities
    attr_matches = dataset.attribute_matches  # Mappa src_pred -> [tgt_pred1, ...]
    
    raw_data = []
    print(f"Estrazione attributi usando {len(attr_matches)} regole di matching...")
    
    sampled_aligned = list(aligned)
    random.shuffle(sampled_aligned)
    
    for s_uri, t_uri in sampled_aligned:
        # 1. Raccogliamo tutti i letterali per questa coppia di entità
        s_vals_by_pred = defaultdict(list)
        for _, p, o in kg_src.triples((s_uri, None, None)):
            if isinstance(o, Literal): s_vals_by_pred[str(p)].append(str(o))
            
        t_vals_by_pred = defaultdict(list)
        for _, p, o in kg_tgt.triples((t_uri, None, None)):
            if isinstance(o, Literal): t_vals_by_pred[str(p)].append(str(o))

        # 2. Applichiamo le regole di Attribute Matching
        for s_pred, t_preds in attr_matches.items():
            if s_pred in s_vals_by_pred:
                for t_pred in t_preds:
                    if t_pred in t_vals_by_pred:
                        # Abbiamo trovato predicati che matchano!
                        v_sources = s_vals_by_pred[s_pred]
                        v_targets = t_vals_by_pred[t_pred]
                        
                        # 3. Proximity Matching (Accoppiamo i valori più simili)
                        # Implementazione della logica formalizzata: per ogni target, trova il sorgente più vicino
                        for vt in v_targets:
                            best_vs = max(v_sources, key=lambda vs: string_sim(vs, vt))
                            raw_data.append((s_pred, best_vs, vt))
                            
        if len(raw_data) >= max_pairs: break

    print(f"Recuperate {len(raw_data)} coppie di valori cross-schema.")

    # Creazione dataset HF (DAE + Identity)
    train_rows = []
    def noise(text):
        if len(text) < 5 or random.random() < 0.15: return text
        l = list(text); i = random.randint(0, len(l)-2)
        l[i], l[i+1] = l[i+1], l[i]
        return "".join(l)

    for p_uri, v_s, v_t in raw_data:
        p_tok = engine.get_token(p_uri)
        # Il modello deve imparare a ricostruire SRC da NOISE(SRC) e TGT da NOISE(TGT)
        # sempre condizionato dal token del predicato
        train_rows.append({"input": f"{p_tok} {noise(v_s)}", "target": f"{p_tok} {v_s}"})
        train_rows.append({"input": f"{p_tok} {noise(v_t)}", "target": f"{p_tok} {v_t}"})
        # Aggiungiamo anche la coppia cross-lingua come identity per "fondere" i concetti
        train_rows.append({"input": f"{p_tok} {v_s}", "target": f"{p_tok} {v_t}"})

    def tokenize(batch):
        inputs = engine.tokenizer(batch["input"], max_length=96, truncation=True, padding="max_length")
        labels = engine.tokenizer(batch["target"], max_length=96, truncation=True, padding="max_length")
        inputs["labels"] = labels["input_ids"]
        return inputs

    return HFDataset.from_list(train_rows).map(tokenize, batched=True, remove_columns=["input", "target"])

# ------------------------------------------------------------------
# 4. INFERENZA CON PARAMETRI PLM.YAML
# ------------------------------------------------------------------

def augment_mixup(engine: BartMixupEngine, v_a: str, v_b: str, pred_uri: str, gen_cfg: dict, alpha=0.5):
    engine.model.eval()
    p_tok = engine.get_token(pred_uri)
    
    t_a = f"{p_tok} {v_a}"
    t_b = f"{p_tok} {v_b}"
    
    inputs = engine.tokenizer([t_a, t_b], return_tensors="pt", padding="max_length", max_length=96, truncation=True).to(engine.device)
    
    with torch.no_grad():
        encoder_outputs = engine.model.get_encoder()(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        H_mix = alpha * encoder_outputs.last_hidden_state[0:1] + (1.0 - alpha) * encoder_outputs.last_hidden_state[1:2]
        
        # Parametri da plm.yaml
        generated_ids = engine.model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=H_mix),
            attention_mask=inputs.attention_mask[0:1],
            max_new_tokens=gen_cfg.get("max_new_tokens", 32),
            do_sample=gen_cfg.get("do_sample", True),
            temperature=gen_cfg.get("temperature", 0.85),
            top_p=gen_cfg.get("top_p", 0.9),
            num_beams=gen_cfg.get("num_beams", 5),
            repetition_penalty=gen_cfg.get("repetition_penalty", 1.7),
            no_repeat_ngram_size=gen_cfg.get("no_repeat_ngram_size", 3)
        )
        
    decoded = engine.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded.replace(p_tok, "").strip()

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Carica Risorse
    full_config = load_project_config()
    gen_config = full_config["augmentation"]["bart"]["generation"]
    dataset = load_real_dataset()
    
    # Estrarre lista predicati reali dal dataset
    all_preds = set(str(p) for _, p, o in dataset.knowledge_graph_source.triples((None, None, None)))
    
    # 2. Setup Modello
    engine = BartMixupEngine("facebook/bart-base", list(all_preds))
    
    # 3. Training (Reale)
    train_ds = prepare_training_data(dataset, engine, max_pairs=1000)
    
    print("\nInizio fine-tuning DAE su dati reali (3 epoche per test)...")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results/bart_mixup_real_test",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_steps=20,
        report_to="none"
    )
    
    trainer = Seq2SeqTrainer(
        model=engine.model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=engine.tokenizer,
        data_collator=DataCollatorForSeq2Seq(engine.tokenizer, model=engine.model)
    )
    
    trainer.train()
    
    # 4. Test Augmentation su coppie reali
    print("\n" + "="*80)
    print("RISULTATI AUGMENTATION (MIX-UP SU DATI REALI)")
    print("="*80)
    
    # Test cases reali da BBC_DB
    test_samples = [
        ("Judas Priest", "Priest, Judas", "http://purl.org/ontology/mo/name"),
        ("1969-04-10", "10 April 1969", "http://purl.org/dc/terms/date"),
        ("London", "Londres", "http://xmlns.com/foaf/0.1/based_near"),
        ("Heavy Metal", "Hard Rock", "http://purl.org/ontology/mo/genre")
    ]
    
    for v_s, v_t, p_uri in test_samples:
        res = augment_mixup(engine, v_s, v_t, p_uri, gen_config, alpha=0.5)
        print(f"Predicato: {p_uri.split('/')[-1]}")
        print(f"  In: '{v_s}' + '{v_t}'")
        print(f"  Out: '{res}'\n")
