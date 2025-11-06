"""
attribute_language_model.py

Train and use a lightweight language model (T5) that learns how to
format attribute values given a predicate, e.g.:
  foaf:name -> "John Doe"
  dbo:birthDate -> "1980-04-12"
"""

import os
import re
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments
)
from difflib import SequenceMatcher


class AttributeLanguageModel:
    def __init__(self,
                 model_name: str = "t5-small",
                 model_path: str = "./attribute_lm",
                 device: str = None):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.exists(model_path) and "pytorch_model.bin" in os.listdir(model_path):
            print(f"[AttributeLM] Loading fine-tuned model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        else:
            print(f"[AttributeLM] Loading base model {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    # ------------------------------------------------------------------
    # Dataset creation
    # ------------------------------------------------------------------
    def build_dataset(self, kg, min_len=2, max_len=100):
        """Extract predicate-value pairs from KG."""
        data = []
        for s, p, o in kg.triples((None, None, None)):
            if hasattr(o, "toPython"):
                o = o.toPython()
            if isinstance(o, str):
                val = o.strip()
                if min_len <= len(val) <= max_len:
                    pred = str(p)
                    clean_pred = re.sub(r".*[#/]", "", pred)
                    data.append({
                        "input": f"{clean_pred} | {val}",
                        "output": val
                    })
        df = pd.DataFrame(data)
        print(f"[AttributeLM] Created dataset with {len(df)} examples.")
        return df

    def split_dataset(self, df, test_size=0.1, seed=42):
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed)
        print(f"[AttributeLM] Split into {len(train_df)} train / {len(val_df)} val")
        return train_df, val_df

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fine_tune(self, train_df, val_df=None, epochs=3, batch_size=16, lr=5e-5):
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(val_df) if val_df is not None else None

        def preprocess(example):
            # Tokenizza input e output con padding e truncation
            model_inputs = self.tokenizer(
                example["input"],
                max_length=64,
                truncation=True,
                padding="max_length"
            )

            labels = self.tokenizer(
                example["output"],
                max_length=64,
                truncation=True,
                padding="max_length"
            )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_train = train_dataset.map(preprocess, batched=True)
        tokenized_eval = eval_dataset.map(preprocess, batched=True) if eval_dataset else None

        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            preds = np.argmax(preds, axis=-1)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            sims = [SequenceMatcher(None, p.strip(), l.strip()).ratio() for p, l in zip(decoded_preds, decoded_labels)]
            return {"similarity": np.mean(sims)}

        kwargs = dict(
            output_dir=self.model_path,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_steps=50
        )

        # version check: se "evaluation_strategy" esiste, lo aggiungiamo
        import inspect
        if "evaluation_strategy" in inspect.signature(TrainingArguments).parameters:
            kwargs.update({
                "evaluation_strategy": "epoch" if eval_dataset else "no",
                "save_strategy": "epoch",
                "report_to": "none",
                "load_best_model_at_end": bool(eval_dataset)
            })

        args = TrainingArguments(**kwargs)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics if eval_dataset else None
        )

        print(f"[AttributeLM] Fine-tuning on {len(train_df)} examples...")
        trainer.train()

        self.save(self.model_path)
        print(f"[AttributeLM] Model fine-tuned and saved to {self.model_path}")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, predicate: str, raw_value: str, max_tokens: int = 32):
        """Generate or normalize an attribute value given predicate + raw text."""
        if not raw_value or len(raw_value.strip()) < 2:
            return raw_value or ""

        input_text = f"{predicate} | {raw_value}"
        toks = self.tokenizer(input_text, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            ids = self.model.generate(
                **toks,
                max_new_tokens=max_tokens,
                num_beams=4,
                early_stopping=True
            )

        output = self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        output = self._clean_output(output)
        if not output or output.lower() in {"none", "null"}:
            return raw_value
        return output

    # ------------------------------------------------------------------
    # Cleanup utilities
    # ------------------------------------------------------------------
    def _clean_output(self, text):
        text = text.strip()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\.\-']", " ", text)
        return text.strip(" .-")

    def save(self, path=None):
        path = path or self.model_path
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path=None):
        path = path or self.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device)
