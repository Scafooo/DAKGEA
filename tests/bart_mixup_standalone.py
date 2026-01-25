import torch
import random
import re
import os
from typing import List, Tuple, Dict, Optional
from transformers import (
    BartForConditionalGeneration, 
    BartTokenizer, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from transformers.modeling_outputs import BaseModelOutput
from datasets import Dataset as HFDataset

# ------------------------------------------------------------------
# 1. SETUP MODELLO E TOKENIZER (STRATEGIA SPECIAL TOKENS)
# ------------------------------------------------------------------

def setup_bart_mixup(model_name: str, predicates: List[str]):
    print(f"Initializing model {model_name}...")
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    # Trasformiamo i predicati in Special Tokens univoci
    # Esempio: "name" -> "<name>", "birthDate" -> "<birthDate>"
    special_tokens = [f"<{p}>" for p in predicates]
    tokenizer.add_tokens(special_tokens)
    
    model = BartForConditionalGeneration.from_pretrained(model_name)
    # Resize obbligatorio per includere i nuovi token nella matrice dei pesi
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer, special_tokens

# ------------------------------------------------------------------
# 2. DATASET PER DENOISING AUTOENCODER (+ Identity Mapping)
# ------------------------------------------------------------------

class MixupTrainingDataBuilder:
    def __init__(self, tokenizer, max_length=96):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _apply_noise(self, text: str) -> str:
        """Applica rumore per il task di denoising."""
        if not text or len(text) < 4 or random.random() < 0.15:
            return text
        
        chars = list(text)
        # 1. Swap casuale
        if random.random() < 0.5:
            idx = random.randint(0, len(chars) - 2)
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        
        # 2. Drop casuale
        if random.random() < 0.3:
            chars.pop(random.randint(0, len(chars) - 1))
            
        return "".join(chars)

    def build_dataset(self, raw_pairs: List[Tuple[str, str, str]]):
        """
        raw_pairs: List di (pred_token, val_src, val_tgt)
        Genera campioni: 
        - <pred> noise(val) -> <pred> val  (Denoising)
        - <pred> val -> <pred> val         (Identity)
        """
        data = []
        for p_tok, v_s, v_t in raw_pairs:
            # Esempi per Source
            data.append({"input": f"{p_tok} {self._apply_noise(v_s)}", "target": f"{p_tok} {v_s}"})
            data.append({"input": f"{p_tok} {v_s}", "target": f"{p_tok} {v_s}"}) # Identity
            
            # Esempi per Target
            data.append({"input": f"{p_tok} {self._apply_noise(v_t)}", "target": f"{p_tok} {v_t}"})
            data.append({"input": f"{p_tok} {v_t}", "target": f"{p_tok} {v_t}"}) # Identity

        def tokenize(batch):
            model_inputs = self.tokenizer(batch["input"], max_length=self.max_length, truncation=True, padding="max_length")
            labels = self.tokenizer(batch["target"], max_length=self.max_length, truncation=True, padding="max_length")
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return HFDataset.from_list(data).map(tokenize, batched=True, remove_columns=["input", "target"])

# ------------------------------------------------------------------
# 3. LOGICA DI INFERENZA (MIX-UP SEPARATO)
# ------------------------------------------------------------------

class MixupInference:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def augment_with_mixup(self, val_a: str, val_b: str, pred_token: str, alpha: float = 0.5):
        self.model.eval()
        
        # Costruzione stringhe condizionate
        text_a = f"{pred_token} {val_a}"
        text_b = f"{pred_token} {val_b}"
        
        # Tokenizzazione con padding fisso per mix-up dimensionale
        inputs = self.tokenizer(
            [text_a, text_b], 
            return_tensors="pt", 
            padding="max_length", 
            max_length=96, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            # 1. Encoding separato
            encoder_outputs = self.model.get_encoder()(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )
            
            # Hidden states: H_A e H_B sono (1, L, D)
            H_A = encoder_outputs.last_hidden_state[0:1]
            H_B = encoder_outputs.last_hidden_state[1:2]
            
            # 2. Mix-up Lineare negli stati latenti
            H_mix = alpha * H_A + (1.0 - alpha) * H_B
            
            # 3. Decodifica dal vettore interpolato
            mixed_enc_out = BaseModelOutput(last_hidden_state=H_mix)
            
            generated_ids = self.model.generate(
                encoder_outputs=mixed_enc_out,
                attention_mask=inputs.attention_mask[0:1], # Usiamo maschera di A (identica a B grazie a padding fisso)
                max_new_tokens=48,
                do_sample=True,
                temperature=0.9
            )
            
        decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # Pulizia dal token del predicato
        return decoded.replace(pred_token, "").strip()

# ------------------------------------------------------------------
# ESECUZIONE TEST
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Predicati reali BBC_DB
    predicates = ["name", "birthDate", "birthPlace", "genre"]
    model, tokenizer, tokens = setup_bart_mixup("facebook/bart-base", predicates)
    
    # Esempi di dati (Simulando allineamenti BBC_DB)
    raw_examples = [
        (tokens[0], "Judas Priest", "Priest, Judas"),
        (tokens[1], "1969-04-10", "10 April 1969"),
        (tokens[2], "Birmingham, England", "Birmingham, UK"),
        (tokens[3], "Heavy Metal", "Hard Rock / Metal")
    ]
    
    print("\n--- Training Step ---")
    builder = MixupTrainingDataBuilder(tokenizer)
    train_ds = builder.build_dataset(raw_examples)
    print(f"Dataset pronto: {len(train_ds)} campioni generati (DAE + Identity).")
    
    # Setup Trainer rapido per il test
    training_args = Seq2SeqTrainingArguments(
        output_dir="./tmp_mixup_checkpoints",
        per_device_train_batch_size=4,
        num_train_epochs=1, # Solo per verifica
        report_to="none"
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )
    
    print("Eseguo un'epoca di training di prova...")
    trainer.train()
    
    print("\n--- Inference Step (Mix-up) ---")
    inf = MixupInference(model, tokenizer, device="cpu") # Test in CPU
    
    for p_tok, v_s, v_t in raw_examples:
        result = inf.augment_with_mixup(v_s, v_t, p_tok, alpha=0.5)
        print(f"Predicate {p_tok}: '{v_s}' + '{v_t}' -> Interpolato: '{result}'")
