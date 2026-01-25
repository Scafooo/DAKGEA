"""
BART Interpolator with Translation-style Training Format.

This variant implements a cross-lingual translation approach:
- Input: <pred_src> <sep> noise(src_value)
- Output: clean(tgt_value)

And the reverse:
- Input: <pred_tgt> <sep> noise(tgt_value)
- Output: clean(src_value)

This format is more appropriate for Entity Alignment as it learns
to translate attribute values across knowledge graphs.
"""

from src.augmentation.methods.plm.bart_interpolator import BartInterpolatorPLM, PairExample
from datasets import Dataset as HFDataset
import random
import re
from collections import defaultdict

class BartInterpolatorTranslation(BartInterpolatorPLM):
    """BART Interpolator using translation-style training format."""

    def _build_translation_dataset(self, pairs, balance_by_predicate=False):
        """Build dataset with translation format: pred <sep> noise(src) → clean(tgt)."""

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
            # Direction 1: pred_src <sep> noise(src) → clean(tgt)
            inp_forward = f"{ex.predicate} <sep> {_noise_str(ex.src_val)}"
            out_forward = ex.out_tgt
            rows.append({
                "input_text": inp_forward,
                "output_text": out_forward,
                "predicate": ex.predicate,
                "direction": "src_to_tgt"
            })

            # Direction 2: pred_tgt <sep> noise(tgt) → clean(src)
            inp_reverse = f"{ex.predicate} <sep> {_noise_str(ex.tgt_val)}"
            out_reverse = ex.out_src
            rows.append({
                "input_text": inp_reverse,
                "output_text": out_reverse,
                "predicate": ex.predicate,
                "direction": "tgt_to_src"
            })

        # Shuffle to mix directions
        random.shuffle(rows)

        return HFDataset.from_list(rows)

    def fine_tune(self, pairs, epochs=20, batch_size=16, lr=5e-5,
                  max_train_samples=None, val_split=0.1, force_retrain=False,
                  num_proc=2, patience=3):
        """Fine-tune with translation format."""
        import os
        import inspect
        from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
        from src.logger import get_logger

        logger = get_logger(__name__)

        # Skip if model already trained
        if (self.reuse_if_available and not force_retrain and
            os.path.isdir(self.out_dir) and
            any(f in os.listdir(self.out_dir) for f in ["pytorch_model.bin", "config.json"])):
            logger.warning(f"[BART-TRANSLATION] Skipping fine-tuning — model already exists in {self.out_dir}.")
            return

        # Subsample if needed
        if max_train_samples and len(pairs) > max_train_samples:
            pairs = random.sample(pairs, max_train_samples)

        logger.info(f"[BART-TRANSLATION] Preparing fine-tuning with translation format ({len(pairs)} pairs → {len(pairs)*2} examples)...")

        # Apply preprocessing (stratified sampling, etc)
        pairs = self._apply_advanced_preprocessing(pairs)

        logger.info(f"[BART-TRANSLATION] After preprocessing: {len(pairs)} pairs")

        # Split train/val
        n_val = max(1, int(len(pairs) * val_split))
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]

        # Build translation-format datasets
        train_ds = self._build_translation_dataset(train_pairs, balance_by_predicate=False)
        val_ds = self._build_translation_dataset(val_pairs, balance_by_predicate=False)

        logger.info(f"[BART-TRANSLATION] Training examples: {len(train_ds)} (from {len(train_pairs)} pairs)")
        logger.info(f"[BART-TRANSLATION] Validation examples: {len(val_ds)} (from {len(val_pairs)} pairs)")

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

        logger.info(f"[BART-TRANSLATION] Starting fine-tuning with translation format...")
        trainer.train()

        # Save model
        self.model.save_pretrained(self.out_dir)
        self.tokenizer.save_pretrained(self.out_dir)
        logger.info(f"[BART-TRANSLATION] Fine-tuned model saved to {self.out_dir}")
