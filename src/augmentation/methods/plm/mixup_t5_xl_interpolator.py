"""T5 XL Mix-up Interpolator using LoRA in BF16 for RTX 4090."""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union

import torch
import torch.nn as nn
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.modeling_outputs import BaseModelOutput
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

logger = logging.getLogger(__name__)

class MixupT5XLInterpolator:
    def __init__(
        self,
        model_name: str = "google/flan-t5-xl",
        out_dir: str = "./t5_xl_lora_model",
        device: str = "cuda",
        max_len_in: int = 128,
        pretrained_path: str | None = None,
        generation_config: dict | None = None,
    ):
        self.model_name = model_name
        self.out_dir = out_dir
        self.device = device
        self.max_len_in = max_len_in
        self.pretrained_path = pretrained_path

        # Generation parameters (from config or defaults)
        gen_cfg = generation_config or {}
        self.gen_temperature = float(gen_cfg.get("temperature", 1.0))
        self.gen_num_beams = int(gen_cfg.get("num_beams", 4))
        self.gen_repetition_penalty = float(gen_cfg.get("repetition_penalty", 2.0))
        self.gen_top_p = float(gen_cfg.get("top_p", 0.9))
        self.latent_noise_std = float(gen_cfg.get("latent_noise_std", 0.05))
        # Default alpha for interpolation (can be overridden from best_config.json)
        self.default_alpha = float(gen_cfg.get("alpha", 0.5))

        # Load tokenizer and model
        if pretrained_path and Path(pretrained_path).exists():
            logger.info(f"[T5-XL] Loading pre-trained model from {pretrained_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)
            self.model = self._load_pretrained_model(pretrained_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            self.model = self._load_model()

    def _load_model(self):
        logger.info(f"[T5-XL] Loading {self.model_name} in BF16 Precision...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.resize_token_embeddings(len(self.tokenizer))
        
        # Fix tecnici necessari per BF16 + LoRA
        model.enable_input_require_grads()
        model.config.use_cache = False 
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q", "v"]
        )
        model = get_peft_model(model, peft_config)
        return model

    def _load_pretrained_model(self, pretrained_path: str):
        """Load a pre-trained LoRA model from disk."""
        from peft import PeftModel

        logger.info(f"[T5-XL] Loading base model {self.model_name} in BF16...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        base_model.resize_token_embeddings(len(self.tokenizer))

        logger.info(f"[T5-XL] Loading LoRA adapters from {pretrained_path}...")
        model = PeftModel.from_pretrained(base_model, pretrained_path)
        model.config.use_cache = False

        logger.info("[T5-XL] Pre-trained model loaded successfully.")
        return model

    def set_predicate_mapping(self, mapping: Dict[str, str]) -> None:
        self.predicate_mapping = mapping
        self.canonical_tokens = sorted(list(set(mapping.values())))
        if self.canonical_tokens:
            self.tokenizer.add_tokens(self.canonical_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"[T5-XL] Added {len(self.canonical_tokens)} canonical predicate tokens to tokenizer")

    def _clean_output(self, text: str) -> str:
        text = re.sub(r'<extra_id_\d+>', '', text)
        text = re.sub(r'^.*:\s*', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # v14: Fix unicode escapes (u00e9 → é, u00f3 → ó)
        def replace_u00(match):
            try:
                hex_val = match.group(0)[1:]  # rimuovi 'u'
                return chr(int(hex_val, 16))
            except:
                return match.group(0)
        text = re.sub(r'u00[a-f0-9]{2}', replace_u00, text, flags=re.IGNORECASE)

        return text

    def fine_tune(self, training_rows: List[Dict[str, str]], epochs: int = 3, batch_size: int = 8, lr: float = 1e-3):
        def tokenize_fn(batch):
            mi = self.tokenizer(batch["input"], max_length=self.max_len_in, truncation=True, padding="max_length")
            lb = self.tokenizer(text_target=batch["target"], max_length=self.max_len_in, truncation=True, padding="max_length")
            labels = [[(l if l != self.tokenizer.pad_token_id else -100) for l in label_seq] for label_seq in lb["input_ids"]]
            mi["labels"] = labels
            return mi
            
        hf_dataset = HFDataset.from_list(training_rows).map(tokenize_fn, batched=True, remove_columns=["input", "target"])
        
        args = Seq2SeqTrainingArguments(
            output_dir=self.out_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,  # Effective batch = batch_size × 2
            num_train_epochs=epochs,
            learning_rate=lr,
            bf16=True,
            logging_steps=20,
            save_strategy="no",
            report_to="none",
            gradient_checkpointing=False,  # Disabled for RTX 4090 24GB - faster training
        )
        
        trainer = Seq2SeqTrainer(model=self.model, args=args, train_dataset=hf_dataset, data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=self.model))
        trainer.train()
        self.model.save_pretrained(self.out_dir)
        self.tokenizer.save_pretrained(self.out_dir)

    def interpolate_pair(self, val_src: str, val_tgt: str, predicate: str = "", alpha: float = None) -> Tuple[str, str]:
        # Use default_alpha if not specified (loaded from best_config.json)
        if alpha is None:
            alpha = self.default_alpha
        # Look up canonical token from mapping (same as training rows); fall back to uppercase local name
        pred_mapping = getattr(self, 'predicate_mapping', {})
        p_tok = pred_mapping.get(predicate, f"<{predicate.split('/')[-1].split('#')[-1].upper()}>")
        # IMPORTANTE: prompt deve matchare ESATTAMENTE il training (NO "synthetic"!)
        prompt_src = f"generate variation {p_tok}: {val_src}"
        prompt_tgt = f"generate variation {p_tok}: {val_tgt}"
        
        inputs = self.tokenizer([prompt_src, prompt_tgt], return_tensors="pt", padding=True, truncation=True, max_length=self.max_len_in).to(self.device)

        self.model.eval()
        with torch.no_grad():
            encoder = self.model.base_model.model.get_encoder()
            enc_out = encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            H_A, H_B = enc_out.last_hidden_state[0:1], enc_out.last_hidden_state[1:2]
            
            H_mix_A = (1.0 - alpha) * H_A + alpha * H_B
            H_mix_B = alpha * H_A + (1.0 - alpha) * H_B
            
            if self.latent_noise_std > 0:
                H_mix_A += torch.randn_like(H_mix_A) * self.latent_noise_std
                H_mix_B += torch.randn_like(H_mix_B) * self.latent_noise_std

            m_f = (inputs.attention_mask[0:1] | inputs.attention_mask[1:2]).repeat(2, 1)
            H_f = torch.cat([H_mix_A, H_mix_B], dim=0)
            
            out_ids = self.model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=H_f),
                attention_mask=m_f,
                max_new_tokens=64,
                do_sample=True,
                temperature=self.gen_temperature,
                num_beams=self.gen_num_beams,
                repetition_penalty=self.gen_repetition_penalty,
                top_p=self.gen_top_p
            )

        res_a = self._clean_output(self.tokenizer.decode(out_ids[0], skip_special_tokens=True)) or val_src
        res_b = self._clean_output(self.tokenizer.decode(out_ids[1], skip_special_tokens=True)) or val_tgt
        
        return res_a, res_b