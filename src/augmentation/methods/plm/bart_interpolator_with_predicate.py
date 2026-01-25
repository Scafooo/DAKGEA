"""
BART Interpolator with Predicate Conditioning.

This variant adds predicate conditioning to the standard reconstruction format:
- Input: <predicate> <sep> noise(src_value) <sep> noise(tgt_value)
- Output: clean(src_value) <sep> clean(tgt_value)

This allows the model to learn predicate-specific value patterns while
maintaining the bidirectional reconstruction objective.
"""

from src.augmentation.methods.plm.bart_interpolator import BartInterpolatorPLM, PairExample
from datasets import Dataset as HFDataset
import random
import re
from collections import defaultdict

class BartInterpolatorWithPredicate(BartInterpolatorPLM):
    """BART Interpolator with predicate conditioning."""

    def _build_predicate_conditioned_dataset(self, pairs, balance_by_predicate=False):
        """Build dataset with predicate: pred <sep> noise(src) <sep> noise(tgt) → clean(src) <sep> clean(tgt)."""

        # Define local noise function
        def _noise_str(x: str) -> str:
            if not x:
                return x
            ops = [
                lambda s: s.lower(),
                lambda s: re.sub(r"\s+", " ", s),
                lambda s: re.sub(r"[^\w\s\.\-']", " ", s),
                lambda s: re.sub(r"\b(\w+)( \1\b)+", r"\1", s, flags=re.IGNORECASE),
                lambda s: s[: max(3, int(len(s) * random.uniform(0.7, 1.0)))],
            ]
            for f in random.sample(ops, random.randint(1, 3)):
                x = f(x)
            return re.sub(r"\s+", " ", x).strip(" .-")

        rows = []
        for ex in pairs:
            # Format: pred <sep> noise(src) <sep> noise(tgt) → clean(src) <sep> clean(tgt)
            inp = f"{ex.predicate} <sep> {_noise_str(ex.src_val)} <sep> {_noise_str(ex.tgt_val)}"
            out = f"{ex.out_src} <sep> {ex.out_tgt}"
            rows.append({
                "input_text": inp,
                "output_text": out,
                "predicate": ex.predicate
            })

        if balance_by_predicate:
            # Optional: balance by predicate using round-robin
            buckets = defaultdict(list)
            for row in rows:
                buckets[row["predicate"]].append(row)

            for k in buckets:
                random.shuffle(buckets[k])

            balanced_rows = []
            max_len = max(len(v) for v in buckets.values())
            keys = list(buckets.keys())
            for i in range(max_len):
                for k in keys:
                    if i < len(buckets[k]):
                        balanced_rows.append(buckets[k][i])
            rows = balanced_rows

        return HFDataset.from_list(rows)

    def fine_tune(self, pairs, epochs=20, batch_size=16, lr=5e-5,
                  max_train_samples=None, val_split=0.1, force_retrain=False,
                  num_proc=2, patience=3):
        """Fine-tune with predicate conditioning."""
        import os
        import inspect
        from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
        from src.logger import get_logger

        logger = get_logger(__name__)

        # Skip if model already trained
        if (self.reuse_if_available and not force_retrain and
            os.path.isdir(self.out_dir) and
            any(f in os.listdir(self.out_dir) for f in ["pytorch_model.bin", "config.json"])):
            logger.warning(f"[BART-WITH-PRED] Skipping fine-tuning — model already exists in {self.out_dir}.")
            return

        # Subsample if needed
        if max_train_samples and len(pairs) > max_train_samples:
            pairs = random.sample(pairs, max_train_samples)

        logger.info(f"[BART-WITH-PRED] Preparing fine-tuning with predicate conditioning ({len(pairs)} pairs)...")

        # Apply preprocessing (stratified sampling, etc)
        pairs = self._apply_advanced_preprocessing(pairs)

        logger.info(f"[BART-WITH-PRED] After preprocessing: {len(pairs)} pairs")

        # Split train/val
        n_val = max(1, int(len(pairs) * val_split))
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]

        # Build predicate-conditioned datasets
        train_ds = self._build_predicate_conditioned_dataset(train_pairs, balance_by_predicate=False)
        val_ds = self._build_predicate_conditioned_dataset(val_pairs, balance_by_predicate=False)

        logger.info(f"[BART-WITH-PRED] Training examples: {len(train_ds)}")
        logger.info(f"[BART-WITH-PRED] Validation examples: {len(val_ds)}")

        # Tokenization
        import torch
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        use_num_proc = None if torch.cuda.is_available() else num_proc

        def preprocess(batch):
            model_inputs = self.tokenizer(
                batch["input_text"],
                max_length=self.max_len_in,
                truncation=True,
                padding="max_length",
            )
            labels = self.tokenizer(
                batch["output_text"],
                max_length=self.max_len_out,
                truncation=True,
                padding="max_length",
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tok_train = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names, num_proc=use_num_proc)
        tok_val = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names, num_proc=use_num_proc)

        # Training arguments
        has_eval_strategy = "evaluation_strategy" in inspect.signature(TrainingArguments).parameters

        args_kwargs = {
            "output_dir": self.out_dir,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "learning_rate": lr,
            "save_strategy": "epoch" if has_eval_strategy else "steps",
            "save_total_limit": 2,
            "logging_dir": os.path.join(self.out_dir, "logs"),
            "logging_strategy": "steps",
            "logging_steps": 50,
            "report_to": "none",
        }

        if has_eval_strategy:
            args_kwargs.update({
                "evaluation_strategy": "epoch",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            })

        args = TrainingArguments(**args_kwargs)

        # Trainer
        callbacks = []
        if has_eval_strategy:
            callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tok_train,
            eval_dataset=tok_val if has_eval_strategy else None,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
        )

        logger.info(f"[BART-WITH-PRED] Starting fine-tuning with predicate conditioning...")
        trainer.train()

        # Save model
        self.model.save_pretrained(self.out_dir)
        self.tokenizer.save_pretrained(self.out_dir)
        logger.info(f"[BART-WITH-PRED] Fine-tuned model saved to {self.out_dir}")
