"""BART Mix-up Interpolator con Predicate Conditioning via Special Tokens.

Pipeline:
1. Condizionamento: predicati registrati come token speciali (<hasTitle>, <hasName>, ...)
2. Training: DAE + Identity Mapping via Seq2SeqTrainer
3. Inference: Mix-up asimmetrico dei Last Hidden States (una encode, due decode)
"""

from __future__ import annotations

import os
import random
import logging
import re
from collections import defaultdict
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
    """BART Interpolator con Mix-up dei Last Hidden States e Predicate Conditioning."""

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

        # Predicate mapping: URI -> special token
        self.predicate_mapping: Dict[str, str] = {}
        self.canonical_tokens: List[str] = []

        # Generation config
        gen_cfg = generation_config or {}
        self.gen_max_new_tokens = int(gen_cfg.get("max_new_tokens", 32))
        self.gen_do_sample = bool(gen_cfg.get("do_sample", True))
        self.gen_top_p = float(gen_cfg.get("top_p", 0.9))
        self.gen_top_k = int(gen_cfg.get("top_k", 50)) # Filtro anti-garbage
        self.gen_temperature = float(gen_cfg.get("temperature", 1.0))
        self.gen_repetition_penalty = float(gen_cfg.get("repetition_penalty", 1.5))
        self.gen_length_penalty = float(gen_cfg.get("length_penalty", 1.0))
        self.gen_num_beams = int(gen_cfg.get("num_beams", 5)) 
        self.gen_no_repeat_ngram_size = int(gen_cfg.get("no_repeat_ngram_size", 3)) 
        self.latent_noise_std = float(gen_cfg.get("latent_noise_std", 0.05)) # Molto basso per integrità

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Load model and tokenizer
        self.model, self.tokenizer = self._load_or_init_model()

    def _load_or_init_model(self):
        """Load fine-tuned model if available, otherwise init from pretrained."""
        if (
            self.reuse_if_available
            and os.path.isdir(self.out_dir)
            and os.path.exists(os.path.join(self.out_dir, "config.json"))
        ):
            logger.info(f"[MIXUP-BART] Loading fine-tuned model from {self.out_dir}")
            tok = BartTokenizer.from_pretrained(self.out_dir)
            mdl = BartForConditionalGeneration.from_pretrained(self.out_dir).to(self.device)
        else:
            logger.info(f"[MIXUP-BART] Initializing from pretrained {self.model_name}")
            tok = BartTokenizer.from_pretrained(self.model_name)
            mdl = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        return mdl, tok

    def set_predicate_mapping(self, mapping: Dict[str, str]) -> None:
        """Register predicate special tokens in tokenizer and resize embeddings."""
        self.predicate_mapping = mapping
        self.canonical_tokens = sorted(list(set(mapping.values())))

        if self.canonical_tokens:
            num_added = self.tokenizer.add_tokens(self.canonical_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"[MIXUP-BART] Registered {num_added} canonical tokens.")

    def _get_pred_token(self, predicate_uri: str) -> str:
        if not predicate_uri: return "<ATTR>"
        return self.predicate_mapping.get(predicate_uri, f"<{str(predicate_uri).split('/')[-1].upper()}>")

    def _clean_output(self, text: str) -> str:
        """Rimuove aggressivamente i token speciali e i residui, inclusi i codici unicode."""
        # 1. Decodifica i residui unicode letterali (es: u00e1 -> á)
        try:
            if "u00" in text:
                text = re.sub(r'(?<!\\)u([0-9a-fA-F]{4})', r'\\u\1', text)
                text = text.encode('utf-8').decode('unicode_escape')
        except Exception:
            pass

        # 2. Rimuove i token canonici
        for tok in self.canonical_tokens:
            text = text.replace(tok, "")
        # 3. Rimuove qualsiasi residuo di tag o frammenti di parentesi angolari
        text = re.sub(r'<[^>]*>', '', text)
        text = text.replace("<", "").replace(">", "")
        # 4. Rimuove spazi multipli
        text = re.sub(r'\s+', ' ', text).strip()
        # 5. Fallback se l'output è troppo corto o solo punteggiatura
        if len(text) < 2: return ""
        return text

    def fine_tune(self, training_rows: List[Dict[str, str]], epochs: int = 5, batch_size: int = 16, lr: float = 5e-5, force_retrain: bool = False):
        if not force_retrain and self._model_exists(): return
        
        logger.info(f"[MIXUP-BART] Fine-tuning on {len(training_rows)} samples...")
        def tokenize_fn(batch):
            model_inputs = self.tokenizer(batch["input"], max_length=self.max_len_in, truncation=True, padding="max_length")
            labels = self.tokenizer(text_target=batch["target"], max_length=self.max_len_in, truncation=True, padding="max_length")
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        hf_dataset = HFDataset.from_list(training_rows).map(tokenize_fn, batched=True, remove_columns=["input", "target"])
        args = Seq2SeqTrainingArguments(
            output_dir=self.out_dir, 
            per_device_train_batch_size=batch_size, 
            num_train_epochs=epochs, 
            learning_rate=lr, 
            weight_decay=0.05,        # Aumentato per regolarizzazione
            label_smoothing_factor=0.1, # Evita overfitting e aiuta la creatività
            lr_scheduler_type="cosine", # Discesa dolce
            warmup_ratio=0.1,          # Inizio graduale
            fp16=torch.cuda.is_available(), 
            save_strategy="no", 
            report_to="none"
        )
        trainer = Seq2SeqTrainer(model=self.model, args=args, train_dataset=hf_dataset, data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=self.model))
        trainer.train()
        self.model.save_pretrained(self.out_dir); self.tokenizer.save_pretrained(self.out_dir)

    def interpolate_pair(self, val_src: str, val_tgt: str, predicate: str = "", alpha: float = 0.5) -> Tuple[str, str]:
        if not val_src and not val_tgt: return "", ""
        if not val_src: return val_tgt, val_tgt
        if not val_tgt: return val_src, val_src

        p_tok = self._get_pred_token(predicate)
        text_a, text_b = f"{p_tok} {val_src}", f"{p_tok} {val_tgt}"

        # Dynamic padding to ensure better alignment in latent space
        inputs = self.tokenizer([text_a, text_b], return_tensors="pt", padding=True, truncation=True, max_length=self.max_len_in).to(self.device)

        self.model.eval()
        with torch.no_grad():
            enc_out = self.model.get_encoder()(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            H_A = enc_out.last_hidden_state[0:1]
            H_B = enc_out.last_hidden_state[1:2]
            
            # Generiamo due punti distinti nello spazio latente
            # H_mix_A è vicino ad A, H_mix_B è vicino a B
            H_mix_A = (1.0 - alpha) * H_A + alpha * H_B
            H_mix_B = alpha * H_A + (1.0 - alpha) * H_B
            
            # Aggiungiamo rumore ad entrambi se richiesto
            if self.latent_noise_std > 0:
                noise_a = torch.randn_like(H_mix_A) * self.latent_noise_std
                noise_b = torch.randn_like(H_mix_B) * self.latent_noise_std
                noise_a[:, 0, :], noise_b[:, 0, :] = 0, 0 # Anchoring
                H_mix_A += noise_a
                H_mix_B += noise_b

            # Doppia generazione (Batch size = 2 per efficienza)
            H_final = torch.cat([H_mix_A, H_mix_B], dim=0)
            mask_final = torch.cat([inputs.attention_mask[0:1], inputs.attention_mask[1:2]], dim=0)
            
            out_ids = self.model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=H_final),
                attention_mask=mask_final,
                max_new_tokens=self.gen_max_new_tokens,
                do_sample=self.gen_do_sample,  # Rispetta sempre la config (permette temperature con beam search)
                top_p=self.gen_top_p,
                top_k=self.gen_top_k,
                temperature=self.gen_temperature,
                num_beams=self.gen_num_beams,
                no_repeat_ngram_size=self.gen_no_repeat_ngram_size,
                repetition_penalty=self.gen_repetition_penalty,
                length_penalty=self.gen_length_penalty,
                early_stopping=True if self.gen_num_beams > 1 else False
            )

        res_a = self._clean_output(self.tokenizer.decode(out_ids[0], skip_special_tokens=True))
        res_b = self._clean_output(self.tokenizer.decode(out_ids[1], skip_special_tokens=True))
        
        return (res_a or val_src), (res_b or val_tgt)

    def _model_exists(self) -> bool:
        return os.path.isdir(self.out_dir) and os.path.exists(os.path.join(self.out_dir, "config.json"))