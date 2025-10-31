"""Latent interpolation utilities built on top of BART for PLM augmentation."""

from __future__ import annotations

import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from datasets import Dataset as HFDataset
from rdflib import Literal
from torch.nn import functional as F
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput

from src.logger import get_logger

logger = get_logger(__name__)

# Disable Flash SDP kernels that misbehave with custom attention masks
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:  # pragma: no cover - hardware dependent
    pass


def _noise_str(value: str) -> str:
    """Inject controlled noise into attribute values to improve robustness."""
    if not value:
        return value
    ops = [
        lambda s: s.lower(),
        lambda s: re.sub(r"\s+", " ", s),
        lambda s: re.sub(r"[^\w\s.\-']", " ", s),
        lambda s: re.sub(r"\b(\w+)( \1\b)+", r"\1", s, flags=re.IGNORECASE),
        lambda s: s[: max(3, int(len(s) * random.uniform(0.7, 1.0)))],
        lambda s: (" " + s) if random.random() < 0.3 else s,
        lambda s: s + (("." if not s.endswith(".") else "")) if random.random() < 0.2 else s,
    ]
    for func in random.sample(ops, random.randint(1, 3)):
        value = func(value)
    return re.sub(r"\s+", " ", value).strip(" .-")


def _clean_pred(predicate: str) -> str:
    """Return the local name for a predicate URI or CURIE."""
    if predicate is None:
        return ""
    tokens = re.split(r"[#/]", str(predicate))
    tail = tokens[-1] if tokens else str(predicate)
    return tail.split(":")[-1]


def _simple_clean(value: str) -> str:
    """Basic text normalization used before feeding data to the PLM."""
    if not value:
        return value
    value = re.sub(r"http\S+", "", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s.\-']", " ", value)
    value = re.sub(r"\b(\w+)( \1\b)+", r"\1", value, flags=re.IGNORECASE)
    value = re.sub(r"\b([A-Za-z])\b", "", value)
    return re.sub(r"\s+", " ", value).strip(" .-")


@dataclass
class PairExample:
    """Training pair used for denoising fine-tuning of the interpolator."""

    predicate: str
    src_val: str
    tgt_val: str
    out_src: str
    out_tgt: str


class BartInterpolatorPLM:
    """BART-based latent interpolator with light-weight fine-tuning support."""

    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        out_dir: str = "./bart_attribute_plm",
        device: Optional[str] = None,
        base_alpha: float = 0.35,
        alpha_spread: float = 0.25,
        max_len_in: int = 96,
        max_len_out: int = 48,
        seed: int = 42,
        reuse_if_available: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.out_dir = out_dir
        self.base_alpha = base_alpha
        self.alpha_spread = alpha_spread
        self.max_len_in = max_len_in
        self.max_len_out = max_len_out
        self.reuse_if_available = reuse_if_available
        random.seed(seed)

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.model, self.tokenizer = self._load_or_init_model()

    def _load_or_init_model(self):
        """Reuse an already fine-tuned model when possible, otherwise load pretrained weights."""
        if (
            self.reuse_if_available
            and os.path.isdir(self.out_dir)
            and any(fname in os.listdir(self.out_dir) for fname in ["pytorch_model.bin", "config.json"])
        ):
            logger.info("[BART-PLM] Reusing fine-tuned model from %s", self.out_dir)
            tokenizer = BartTokenizer.from_pretrained(self.out_dir)
            model = BartForConditionalGeneration.from_pretrained(self.out_dir).to(self.device)
        else:
            logger.info("[BART-PLM] Loading pretrained model %s", self.model_name)
            tokenizer = BartTokenizer.from_pretrained(self.model_name)
            model = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

        special_tokens = {"additional_special_tokens": ["<SRC>", "<TGT>", "<SEP>"]}
        tokenizer.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def _build_hf_dataset_from_pairs(self, pairs: List[PairExample], balance_by_predicate: bool = True):
        """Create a Hugging Face dataset, optionally balancing samples across predicates."""
        if not balance_by_predicate:
            rows = [
                {
                    "input_text": f"{example.predicate} <SEP> {_noise_str(example.src_val)} <SEP> {_noise_str(example.tgt_val)}",
                    "output_text": f"{example.out_src} | {example.out_tgt}",
                    "predicate": example.predicate,
                }
                for example in pairs
            ]
            return HFDataset.from_list(rows)

        buckets = defaultdict(list)
        for example in pairs:
            buckets[example.predicate].append(example)

        for predicate in buckets:
            random.shuffle(buckets[predicate])

        rows = []
        max_len = max(len(values) for values in buckets.values())
        ordered_predicates = list(buckets.keys())
        for idx in range(max_len):
            for predicate in ordered_predicates:
                if idx < len(buckets[predicate]):
                    example = buckets[predicate][idx]
                    rows.append(
                        {
                            "input_text": f"{example.predicate} <SEP> {_noise_str(example.src_val)} <SEP> {_noise_str(example.tgt_val)}",
                            "output_text": f"{example.out_src} | {example.out_tgt}",
                            "predicate": example.predicate,
                        }
                    )

        return HFDataset.from_list(rows)

    def fine_tune(
        self,
        pairs: List[PairExample],
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 5e-5,
        max_train_samples: Optional[int] = 4000,
        val_split: float = 0.1,
        force_retrain: bool = False,
        num_proc: int = 2,
        patience: int = 3,
    ):
        """Fine-tune the underlying BART model using synthetic training pairs."""
        from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

        if (
            self.reuse_if_available
            and not force_retrain
            and os.path.isdir(self.out_dir)
            and any(fname in os.listdir(self.out_dir) for fname in ["pytorch_model.bin", "config.json"])
        ):
            logger.info("[BART-PLM] Skipping fine-tuning; cached weights found in %s", self.out_dir)
            return

        if max_train_samples and len(pairs) > max_train_samples:
            logger.info(
                "[BART-PLM] Subsampling %d/%d training pairs for efficiency.",
                max_train_samples,
                len(pairs),
            )
            random.shuffle(pairs)
            pairs = pairs[:max_train_samples]

        dataset = self._build_hf_dataset_from_pairs(pairs)
        dataset = dataset.train_test_split(test_size=val_split)

        tokenized = dataset.map(
            lambda ex: self.tokenizer(
                ex["input_text"],
                max_length=self.max_len_in,
                truncation=True,
            ),
            batched=True,
            remove_columns=["input_text", "predicate"],
        )
        tokenized = tokenized.map(
            lambda ex: self.tokenizer(
                ex["output_text"],
                max_length=self.max_len_out,
                truncation=True,
            ),
            batched=True,
            remove_columns=["output_text"],
            fn_kwargs={"add_special_tokens": False},
        )

        training_args = TrainingArguments(
            output_dir=self.out_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            logging_steps=50,
            save_strategy="no",
            evaluation_strategy="epoch",
            learning_rate=lr,
            dataloader_num_workers=num_proc,
            report_to=[],
        )

        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["test"],
            callbacks=callbacks,
        )

        trainer.train()
        trainer.save_model(self.out_dir)
        self.tokenizer.save_pretrained(self.out_dir)

    def build_pairs_from_dataset(
        self,
        kg_source,
        kg_target,
        aligned_entities,
        max_pairs_per_predicate: int = 100,
    ) -> List[PairExample]:
        examples: List[PairExample] = []
        predicate_counts = defaultdict(int)

        for src, tgt in aligned_entities:
            src_literals = _collect_literals(kg_source, src)
            tgt_literals = _collect_literals(kg_target, tgt)

            common_preds = set(src_literals.keys()) & set(tgt_literals.keys())
            for pred in common_preds:
                if predicate_counts[pred] >= max_pairs_per_predicate:
                    continue
                src_val = src_literals[pred]
                tgt_val = tgt_literals[pred]
                if not src_val or not tgt_val:
                    continue
                predicate_counts[pred] += 1
                examples.append(
                    PairExample(
                        predicate=pred,
                        src_val=src_val,
                        tgt_val=tgt_val,
                        out_src=_simple_clean(src_val),
                        out_tgt=_simple_clean(tgt_val),
                    )
                )
        logger.info("[BART-PLM] Built %d training pairs from dataset.", len(examples))
        return examples

    def interpolate_pair(self, src_val: str, tgt_val: str, *, predicate: str, max_new_tokens: int = 32) -> Tuple[str, str]:
        prompt = f"{predicate} <SEP> {src_val} <SEP> {tgt_val}".strip()
        encoded = self.tokenizer( [prompt], return_tensors="pt" )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs: BaseModelOutput = self.model.encoder(**encoded, return_dict=True)
            hidden_states = outputs.last_hidden_state

        alpha = random.uniform(self.base_alpha - self.alpha_spread, self.base_alpha + self.alpha_spread)
        inv_alpha = 1 - alpha
        fused = alpha * hidden_states[:, :, :hidden_states.size(-1)] + inv_alpha * hidden_states

        decoder_input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=self.device)
        generated = self.model.generate(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=(fused,),
            max_new_tokens=max_new_tokens,
            num_beams=4,
            do_sample=False,
        )

        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        parts = [segment.strip() for segment in text.split("|")[:2]]
        out_src = _simple_clean(parts[0] if parts else src_val)
        out_tgt = _simple_clean(parts[1] if len(parts) > 1 else tgt_val)
        return out_src, out_tgt


def _collect_literals(graph, entity: Literal) -> dict[str, str]:
    values: dict[str, str] = {}
    for _, pred, obj in graph.triples((entity, None, None)):
        if isinstance(obj, Literal):
            key = str(pred)
            if key not in values:
                values[key] = str(obj)
    return values
