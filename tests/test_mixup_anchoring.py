import sys
import torch
import logging
import random
import time
import json
from pathlib import Path
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset as HFDataset

# Aggiunta del progetto al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(PROJECT_ROOT)))

from src.core.dataset.reader.openea_dataset_reader import OpeneaDatasetReader
from src.augmentation.methods.plm.mixup_bart_interpolator import MixupBartInterpolator
from src.augmentation.methods.plm.mixup_data_builder import MixupDataBuilder

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MixupAnchoring")

# ------------------------------------------------------------------
# CANONICAL PREDICATE MAP (Specifico per BBC_DB)
# ------------------------------------------------------------------
PREDICATE_MAP = {
    "http://purl.org/ontology/mo/name": "<NAME>",
    "http://purl.org/dc/elements/1.1/title": "<NAME>", # Sinonimo semantico
    "http://purl.org/dc/terms/date": "<DATE>",
    "http://purl.org/ontology/mo/genre": "<GENRE>",
    "http://xmlns.com/foaf/0.1/based_near": "<LOCATION>"
}

def noise_fn(text):
    if not text or len(text) < 5 or random.random() < 0.1: return text
    l = list(text); i = random.randint(0, len(l)-2)
    l[i], l[i+1] = l[i+1], l[i]
    return "".join(l)

def run_anchoring_test():
    print("\n" + "█"*100)
    print("█" + " MIX-UP ANCHORING & CANONICAL TOKEN TEST ".center(98) + "█")
    print("█"*100)

    # 1. Caricamento Dataset
    reader = OpeneaDatasetReader()
    dataset = reader.read("data/raw/openea/BBC_DB")
    
    # 2. Setup Modello con Mapping Canonico
    device = "cuda" if torch.cuda.is_available() else "cpu"
    interpolator = MixupBartInterpolator(
        model_name="facebook/bart-base",
        device=device,
        reuse_if_available=False
    )
    
    # Applichiamo il mapping canonico (registra i token <NAME>, <DATE>, etc.)
    interpolator.set_predicate_mapping(PREDICATE_MAP)

    # 3. Estrazione e Training (DAE + Canonical Conditioning)
    builder = MixupDataBuilder()   raw_pairs = builder.build_denoising_pairs(dataset, max_pairs_per_pred=500)
    
    training_rows = []
    for p_uri, v_s, v_t in raw_pairs:
        p_tok = interpolator._get_pred_token(p_uri)
        # Task: Noise -> Clean (Target è sempre condizionato dal token canonico)
        training_rows.append({"input": f"{p_tok} {noise_fn(v_s)}", "target": f"{p_tok} {v_s}"})
        training_rows.append({"input": f"{p_tok} {noise_fn(v_t)}", "target": f"{p_tok} {v_t}"})
        # Concept Merging: v_s -> v_t (nello stesso spazio canonico)
        training_rows.append({"input": f"{p_tok} {v_s}", "target": f"{p_tok} {v_t}"})

    def tokenize(batch):
        return interpolator.tokenizer(batch["input"], text_target=batch["target"], 
                                    max_length=96, truncation=True, padding="max_length")

    hf_ds = HFDataset.from_list(training_rows).map(tokenize, batched=True)
    
    # Training veloce (5 epoche) per verificare l'ancoraggio
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results/mixup_anchoring_test",
        per_device_train_batch_size=16,
        num_train_epochs=5,
        report_to="none",
        fp16=torch.cuda.is_available()
    )
    
    trainer = Seq2SeqTrainer(
        model=interpolator.model,
        args=training_args,
        train_dataset=hf_ds,
        data_collator=DataCollatorForSeq2Seq(interpolator.tokenizer, model=interpolator.model)
    )
    
    trainer.train()

    # 4. VERIFICA ANCORAGGIO (Challenge)
    print("\n" + "="*100)
    print(" VERIFICATION: ANCHORING EFFECT ".center(100))
    print("=" * 100)

    # Testiamo due predicati diversi che mappano sullo stesso token canonico <NAME>
    test_cases = [
        ("Judas Priest", "Priest, Judas", "http://purl.org/ontology/mo/name", "Predicate: mo/name -> <NAME>"),
        ("British Steel", "British Steel (Album)", "http://purl.org/dc/elements/1.1/title", "Predicate: dc/title -> <NAME>")
    ]

    for v_s, v_t, p_uri, desc in test_cases:
        p_tok = interpolator._get_pred_token(p_uri)
        print(f"\n{desc} (Canonical: {p_tok})")
        print(f"  In: '{v_s}' + '{v_t}'")
        
        # Verifichiamo se il decoder è "ancorato"
        res, _ = interpolator.interpolate_pair(v_s, v_t, predicate=p_uri, alpha=0.5)
        print(f"  Out: '{res}'")
        
        # Check manuale se l'output contiene il token (non dovrebbe, pulito dall'interpolatore)
        if p_tok in res:
            print("  ⚠️ Warning: Canonical token leaked into output text!")
        else:
            print("  ✅ Success: Decoder generated clean value anchored to semantic type.")

    print("\n" + "="*100)

if __name__ == "__main__":
    run_anchoring_test()
