"BART Mix-up Interpolator con Dual-Mode Inference (Standard vs Creative)."

from __future__ import annotations

import os
import random
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.modeling_outputs import BaseModelOutput
from datasets import Dataset as HFDataset

logger = logging.getLogger(__name__)

class MixupBartInterpolator:
    """BART Interpolator che adatta i parametri basandosi sulla somiglianza della coppia."""

    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        out_dir: str = "./bart_mixup_model",
        device: Optional[str] = None,
        seed: int = 42,
        max_len_in: int = 96,
        generation_config: Optional[Dict] = None,
        reuse_if_available: bool = True,
    ):
        self.model_name = model_name
        self.out_dir = out_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.max_len_in = max_len_in
        self.reuse_if_available = reuse_if_available

        self.predicate_mapping: Dict[str, str] = {}
        self.canonical_tokens: List[str] = []

        # Configurazione Generazione
        gen_cfg = generation_config or {}
        self.gen_max_new_tokens = int(gen_cfg.get("max_new_tokens", 32))
        self.gen_do_sample = bool(gen_cfg.get("do_sample", True))
        self.gen_top_p = float(gen_cfg.get("top_p", 0.9))
        self.gen_top_k = int(gen_cfg.get("top_k", 50))
        
        # --- PARAMETRI STANDARD ---
        self.gen_temperature = float(gen_cfg.get("temperature", 1.0))
        self.gen_repetition_penalty = float(gen_cfg.get("repetition_penalty", 1.5))
        self.latent_noise_std = float(gen_cfg.get("latent_noise_std", 0.05))
        
        # --- PARAMETRI CREATIVE (per casi di uguaglianza) ---
        self.creative_temp = 1.5
        self.creative_noise = 0.15
        self.creative_penalty = 2.5
        self.similarity_threshold = 0.85 

        self.gen_num_beams = int(gen_cfg.get("num_beams", 5)) 
        self.gen_no_repeat_ngram_size = int(gen_cfg.get("no_repeat_ngram_size", 3)) 
        self.gen_length_penalty = float(gen_cfg.get("length_penalty", 1.0))

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.model, self.tokenizer = self._load_or_init_model()

    def _load_or_init_model(self):
        if self.reuse_if_available and os.path.isdir(self.out_dir) and os.path.exists(os.path.join(self.out_dir, "config.json")):
            tok = BartTokenizer.from_pretrained(self.out_dir)
            mdl = BartForConditionalGeneration.from_pretrained(self.out_dir).to(self.device)
        else:
            tok = BartTokenizer.from_pretrained(self.model_name)
            mdl = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        return mdl, tok

    def set_predicate_mapping(self, mapping: Dict[str, str]) -> None:
        self.predicate_mapping = mapping
        self.canonical_tokens = sorted(list(set(mapping.values())))
        if self.canonical_tokens:
            self.tokenizer.add_tokens(self.canonical_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _get_pred_token(self, predicate_uri: str) -> str:
        return self.predicate_mapping.get(predicate_uri, f"<{str(predicate_uri).split('/')[-1].upper()}>")

    def _clean_output(self, text: str) -> str:
        try:
            if "u00" in text:
                text = re.sub(r'(?<!\\)u([0-9a-fA-F]{4})', r'\\u\1', text)
                text = text.encode('utf-8').decode('unicode_escape')
        except: pass
        for tok in self.canonical_tokens: text = text.replace(tok, "")
        text = re.sub(r'<[^>]*>', '', text)
        text = text.replace("<", "").replace(">", "")
        return re.sub(r'\s+', ' ', text).strip()

    def fine_tune(self, training_rows: List[Dict[str, str]], epochs: int = 5, batch_size: int = 16, lr: float = 5e-5, force_retrain: bool = False):
        def tokenize_fn(batch):
            mi = self.tokenizer(batch["input"], max_length=self.max_len_in, truncation=True, padding="max_length")
            lb = self.tokenizer(text_target=batch["target"], max_length=self.max_len_in, truncation=True, padding="max_length")
            mi["labels"] = lb["input_ids"]; return mi
        hf_dataset = HFDataset.from_list(training_rows).map(tokenize_fn, batched=True, remove_columns=["input", "target"])
        args = Seq2SeqTrainingArguments(output_dir=self.out_dir, per_device_train_batch_size=batch_size, num_train_epochs=epochs, 
                                        learning_rate=lr, weight_decay=0.05, label_smoothing_factor=0.1, lr_scheduler_type="cosine", warmup_ratio=0.1,
                                        fp16=torch.cuda.is_available(), save_strategy="no", report_to="none")
        trainer = Seq2SeqTrainer(model=self.model, args=args, train_dataset=hf_dataset, data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=self.model))
        trainer.train(); self.model.save_pretrained(self.out_dir); self.tokenizer.save_pretrained(self.out_dir)

    def interpolate_pair(self, val_src: str, val_tgt: str, predicate: str = "", alpha: float = 0.5) -> Tuple[str, str]:
        if not val_src and not val_tgt: return "", ""
        p_tok = self._get_pred_token(predicate)
        inputs = self.tokenizer([f"{p_tok} {val_src}", f"{p_tok} {val_tgt}"], return_tensors="pt", padding=True, truncation=True, max_length=self.max_len_in).to(self.device)

        # Calcolo somiglianza per decidere la modalità
        w1, w2 = set(val_src.lower().split()), set(val_tgt.lower().split())
        jaccard = len(w1 & w2) / len(w1 | w2) if (w1 | w2) else 1.0
        use_creative = (jaccard > self.similarity_threshold)

        self.model.eval()
        with torch.no_grad():
            enc_out = self.model.get_encoder()(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            H_A, H_B = enc_out.last_hidden_state[0:1], enc_out.last_hidden_state[1:2]
            
            curr_noise = self.creative_noise if use_creative else self.latent_noise_std
            curr_temp = self.creative_temp if use_creative else self.gen_temperature
            curr_pen = self.creative_penalty if use_creative else self.gen_repetition_penalty

            H_mix_A = (1.0 - alpha) * H_A + alpha * H_B
            H_mix_B = alpha * H_A + (1.0 - alpha) * H_B
            
            if curr_noise > 0:
                n_a, n_b = torch.randn_like(H_mix_A)*curr_noise, torch.randn_like(H_mix_B)*curr_noise
                n_a[:,0,:], n_b[:,0,:] = 0, 0
                H_mix_A, H_mix_B = H_mix_A+n_a, H_mix_B+n_b

            H_f = torch.cat([H_mix_A, H_mix_B], dim=0)
            m_f = torch.cat([inputs.attention_mask[0:1], inputs.attention_mask[1:2]], dim=0)
            
            out_ids = self.model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=H_f), attention_mask=m_f, 
                                        max_new_tokens=self.gen_max_new_tokens, do_sample=True, temperature=curr_temp,
                                        top_p=self.gen_top_p, top_k=self.gen_top_k, num_beams=self.gen_num_beams,
                                        no_repeat_ngram_size=self.gen_no_repeat_ngram_size, repetition_penalty=curr_pen,
                                        length_penalty=self.gen_length_penalty, early_stopping=True if self.gen_num_beams > 1 else False)

        return self._clean_output(self.tokenizer.decode(out_ids[0], skip_special_tokens=True)) or val_src, \
               self._clean_output(self.tokenizer.decode(out_ids[1], skip_special_tokens=True)) or val_tgt
