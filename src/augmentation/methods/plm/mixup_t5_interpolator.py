"""T5 Mix-up Interpolator for Semantic Data Augmentation."""

from __future__ import annotations

import os
import random
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.modeling_outputs import BaseModelOutput
from datasets import Dataset as HFDataset

logger = logging.getLogger(__name__)

class MixupT5Interpolator:
    """T5 Interpolator con Mix-up nello spazio latente dell'Encoder."""

    def __init__(
        self,
        model_name: str = "google/t5-v1_1-base",
        out_dir: str = "./t5_mixup_model",
        device: Optional[str] = None,
        max_len_in: int = 128,
        reuse_if_available: bool = True,
    ):
        self.model_name = model_name
        self.out_dir = out_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len_in = max_len_in
        self.reuse_if_available = reuse_if_available

        # Parametri di default (sovrascritti dallo sweep)
        self.latent_noise_std = 0.05
        self.gen_temperature = 1.0
        self.gen_num_beams = 5
        self.gen_repetition_penalty = 1.5
        self.gen_top_k = 50
        self.gen_top_p = 0.95
        
        # Creative Mode params
        self.creative_temp = 1.5
        self.creative_noise = 0.15
        self.similarity_threshold = 0.85

        self.model, self.tokenizer = self._load_or_init_model()

    def _load_or_init_model(self):
        if self.reuse_if_available and os.path.isdir(self.out_dir) and os.path.exists(os.path.join(self.out_dir, "config.json")):
            logger.info(f"[MIXUP-T5] Loading fine-tuned model from {self.out_dir}")
            tok = T5Tokenizer.from_pretrained(self.out_dir)
            mdl = T5ForConditionalGeneration.from_pretrained(self.out_dir).to(self.device)
        else:
            logger.info(f"[MIXUP-T5] Initializing from pretrained {self.model_name}")
            tok = T5Tokenizer.from_pretrained(self.model_name)
            mdl = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        return mdl, tok

    def _clean_output(self, text: str) -> str:
        # T5 a volte genera tag <extra_id_0>, li rimuoviamo
        text = re.sub(r'<extra_id_\d+>', '', text)
        text = re.sub(r'<[^>]*>', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def fine_tune(self, training_rows: List[Dict[str, str]], epochs: int = 5, batch_size: int = 16, lr: float = 1e-4, force_retrain: bool = False):
        def tokenize_fn(batch):
            mi = self.tokenizer(batch["input"], max_length=self.max_len_in, truncation=True, padding="max_length")
            lb = self.tokenizer(text_target=batch["target"], max_length=self.max_len_in, truncation=True, padding="max_length")
            mi["labels"] = lb["input_ids"]; return mi
            
        hf_dataset = HFDataset.from_list(training_rows).map(tokenize_fn, batched=True, remove_columns=["input", "target"])
        args = Seq2SeqTrainingArguments(
            output_dir=self.out_dir, 
            per_device_train_batch_size=batch_size, 
            num_train_epochs=epochs, 
            learning_rate=lr, 
            weight_decay=0.01, 
            fp16=torch.cuda.is_available(), 
            gradient_checkpointing=True, # SALVA-VRAM
            save_strategy="no", 
            report_to="none"
        )
        trainer = Seq2SeqTrainer(model=self.model, args=args, train_dataset=hf_dataset, data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=self.model))
        trainer.train(); self.model.save_pretrained(self.out_dir); self.tokenizer.save_pretrained(self.out_dir)

    def interpolate_pair(self, val_src: str, val_tgt: str, predicate: str = "", alpha: float = 0.5) -> Tuple[str, str]:
        # T5 Prompting: "rewrite [predicate]: [value]"
        p_name = predicate.replace("<", "").replace(">", "").lower()
        inputs = self.tokenizer([f"rewrite {p_name}: {val_src}", f"rewrite {p_name}: {val_tgt}"], return_tensors="pt", padding=True, truncation=True, max_length=self.max_len_in).to(self.device)

        # Dual Mode Logic
        w1, w2 = set(val_src.lower().split()), set(val_tgt.lower().split())
        jaccard = len(w1 & w2) / len(w1 | w2) if (w1 | w2) else 1.0
        use_creative = (jaccard > self.similarity_threshold)
        
        curr_noise = self.creative_noise if use_creative else self.latent_noise_std
        curr_temp = self.creative_temp if use_creative else self.gen_temperature

        self.model.eval()
        with torch.no_grad():
            # T5 Encoder
            enc_out = self.model.encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            H_A, H_B = enc_out.last_hidden_state[0:1], enc_out.last_hidden_state[1:2]
            
            # Mixup
            H_mix_A = (1.0 - alpha) * H_A + alpha * H_B
            H_mix_B = alpha * H_A + (1.0 - alpha) * H_B
            
            if curr_noise > 0:
                n_a = torch.randn_like(H_mix_A) * curr_noise
                n_b = torch.randn_like(H_mix_B) * curr_noise
                H_mix_A += n_a; H_mix_B += n_b

            H_f = torch.cat([H_mix_A, H_mix_B], dim=0)
            m_f = torch.cat([inputs.attention_mask[0:1], inputs.attention_mask[1:2]], dim=0)
            
            out_ids = self.model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=H_f), 
                attention_mask=m_f, 
                max_new_tokens=64,
                do_sample=True,
                temperature=curr_temp,
                top_p=self.gen_top_p,
                top_k=self.gen_top_k,
                num_beams=self.gen_num_beams,
                repetition_penalty=self.gen_repetition_penalty
            )

        res_a = self._clean_output(self.tokenizer.decode(out_ids[0], skip_special_tokens=True)) or val_src
        res_b = self._clean_output(self.tokenizer.decode(out_ids[1], skip_special_tokens=True)) or val_tgt
        
        return res_a, res_b
