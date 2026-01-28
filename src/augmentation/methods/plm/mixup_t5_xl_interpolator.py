"""T5 XL Mix-up Interpolator using LoRA in BF16 for RTX 4090."""

from __future__ import annotations

import os
import re
import logging
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
    prepare_model_for_kbit_training,
    TaskType
)

logger = logging.getLogger(__name__)

class MixupT5XLInterpolator:
    """
    Flan-T5-XL Interpolator ottimizzato per RTX 4090 usando LoRA in BF16 (No Quantization).
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-xl",
        out_dir: str = "./t5_xl_lora_model",
        device: str = "cuda",
        max_len_in: int = 128,
    ):
        self.model_name = model_name
        self.out_dir = out_dir
        self.device = device
        self.max_len_in = max_len_in
        
        # Configurazione Generazione
        self.gen_temperature = 0.8
        self.gen_num_beams = 5
        self.gen_repetition_penalty = 2.5
        self.gen_top_p = 0.9
        self.latent_noise_std = 0.03

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self._load_model()

    def _load_model(self):
        """Carica il modello in BF16 e prepara LoRA."""
        logger.info(f"[T5-XL] Loading {self.model_name} in BF16 Precision...")
        
        # Carichiamo il modello direttamente in BF16 (senza quantizzazione 8-bit)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Ridimensiona per sicurezza
        model.resize_token_embeddings(len(self.tokenizer))
        
        # Configura LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"] 
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    def _clean_output(self, text: str) -> str:
        text = re.sub(r'<extra_id_\d+>', '', text)
        text = re.sub(r'^.*: ', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def fine_tune(self, training_rows: List[Dict[str, str]], epochs: int = 3, batch_size: int = 8, lr: float = 5e-4):
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
            gradient_accumulation_steps=4,
            num_train_epochs=epochs,
            learning_rate=lr,
            bf16=True,
            logging_steps=20,
            save_strategy="no",
            report_to="none",
            label_smoothing_factor=0.1,
            gradient_checkpointing=True
        )
        
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=hf_dataset,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        )
        
        logger.info("[T5-XL] Starting BF16 LoRA Fine-tuning...")
        trainer.train()
        self.model.save_pretrained(self.out_dir)
        self.tokenizer.save_pretrained(self.out_dir)

    def interpolate_pair(self, val_src: str, val_tgt: str, predicate: str = "", alpha: float = 0.5) -> Tuple[str, str]:
        p_name = predicate.replace("<", "").replace(">", "").lower().replace('_', ' ')
        prompt_src = f"generate synthetic variation <{p_name}>: {val_src}"
        prompt_tgt = f"generate synthetic variation <{p_name}>: {val_tgt}"
        
        inputs = self.tokenizer([prompt_src, prompt_tgt], return_tensors="pt", padding=True, truncation=True, max_length=self.max_len_in).to(self.device)

        self.model.eval()
        with torch.no_grad():
            # In PEFT, il modello base è accessibile via model.base_model.model
            encoder = self.model.base_model.model.get_encoder()
            
            enc_out = encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            H_A, H_B = enc_out.last_hidden_state[0:1], enc_out.last_hidden_state[1:2]
            
            H_mix_A = (1.0 - alpha) * H_A + alpha * H_B
            H_mix_B = alpha * H_A + (1.0 - alpha) * H_B
            
            if self.latent_noise_std > 0:
                H_mix_A += torch.randn_like(H_mix_A) * self.latent_noise_std
                H_mix_B += torch.randn_like(H_mix_B) * self.latent_noise_std

            # Unione maschere
            combined_mask = (inputs.attention_mask[0:1] | inputs.attention_mask[1:2])
            H_f = torch.cat([H_mix_A, H_mix_B], dim=0)
            m_f = torch.cat([combined_mask, combined_mask], dim=0)
            
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