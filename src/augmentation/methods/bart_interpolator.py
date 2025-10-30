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
from rdflib import Literal, URIRef
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

    # ------------------------------------------------------------------
    # Model loading / initialization logic
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Fine-tuning utilities
    # ------------------------------------------------------------------
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
        import inspect
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
            pairs = random.sample(pairs, max_train_samples)

        if not pairs:
            logger.warning("[BART-PLM] No training pairs available; skipping fine-tuning.")
            return

        logger.info("[BART-PLM] Preparing %d fine-tuning examples.", len(pairs))

        def to_rows(examples: List[PairExample]):
            rows = []
            for example in examples:
                input_text = f"{example.predicate} <s> {_noise_str(example.src_val)} <t> {_noise_str(example.tgt_val)}"
                output_text = f"{example.out_src} <t> {example.out_tgt}"
                rows.append({"input_text": input_text, "output_text": output_text})
            return rows

        val_size = max(1, int(len(pairs) * val_split))
        val_pairs = pairs[:val_size]
        train_pairs = pairs[val_size:]

        train_ds = HFDataset.from_list(to_rows(train_pairs))
        val_ds = HFDataset.from_list(to_rows(val_pairs))

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        use_num_proc = None if torch.cuda.is_available() else num_proc
        logger.info("[BART-PLM] Tokenizing on %s.", "GPU" if torch.cuda.is_available() else "CPU")

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

        train_tokenized = train_ds.map(
            preprocess,
            batched=True,
            remove_columns=train_ds.column_names,
            num_proc=use_num_proc,
        )
        val_tokenized = val_ds.map(
            preprocess,
            batched=True,
            remove_columns=val_ds.column_names,
            num_proc=use_num_proc,
        )

        has_eval_strategy = "evaluation_strategy" in inspect.signature(TrainingArguments).parameters

        args_kwargs = dict(
            output_dir=self.out_dir,
            overwrite_output_dir=force_retrain,
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=2,
            logging_dir=os.path.join(self.out_dir, "logs"),
            logging_strategy="steps",
            logging_steps=50,
            report_to="none",
        )

        if has_eval_strategy:
            args_kwargs.update(
                {
                    "evaluation_strategy": "epoch",
                    "save_strategy": "epoch",
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "eval_loss",
                    "greater_is_better": False,
                }
            )
        else:
            args_kwargs["save_strategy"] = "steps"

        training_args = TrainingArguments(**args_kwargs)

        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)] if has_eval_strategy else []

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized if has_eval_strategy else None,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
        )

        logger.info(
            "[BART-PLM] Starting fine-tuning (%s early stopping).",
            "with" if has_eval_strategy else "without",
        )
        trainer.train()

        self.model.save_pretrained(self.out_dir)
        self.tokenizer.save_pretrained(self.out_dir)
        logger.info("[BART-PLM] Fine-tuned model stored in %s", self.out_dir)

    # ------------------------------------------------------------------
    # Dataset-driven pair generation
    # ------------------------------------------------------------------
    def build_pairs_from_dataset(
        self,
        knowledge_graph_source,
        knowledge_graph_target,
        aligned_entities: Iterable[Tuple[URIRef, URIRef]],
        max_per_predicate: int = 5000,
    ) -> List[PairExample]:
        """Extract literal pairs for aligned entities sharing predicates."""

        def collect_literals(graph):
            index = {}
            for subj, pred, obj in graph.triples((None, None, None)):
                if isinstance(obj, Literal):
                    pred_map = index.setdefault(subj, {})
                    pred_map.setdefault(pred, []).append(str(obj))
            return index

        idx_src = collect_literals(knowledge_graph_source)
        idx_tgt = collect_literals(knowledge_graph_target)

        per_predicate_counter = {}
        examples: List[PairExample] = []

        for entity_src, entity_tgt in aligned_entities:
            src_predicates = idx_src.get(entity_src, {})
            tgt_predicates = idx_tgt.get(entity_tgt, {})

            src_local = {_clean_pred(pred): pred for pred in src_predicates.keys()}
            tgt_local = {_clean_pred(pred): pred for pred in tgt_predicates.keys()}
            common_predicates = set(src_local.keys()) & set(tgt_local.keys())

            for local_name in common_predicates:
                pred_src = src_local[local_name]
                pred_tgt = tgt_local[local_name]

                values_src = src_predicates.get(pred_src, [])
                values_tgt = tgt_predicates.get(pred_tgt, [])
                if not values_src or not values_tgt:
                    continue

                for value_src in values_src:
                    for value_tgt in values_tgt:
                        count = per_predicate_counter.get(local_name, 0)
                        if count >= max_per_predicate:
                            break
                        per_predicate_counter[local_name] = count + 1

                        example = PairExample(
                            predicate=local_name,
                            src_val=_simple_clean(value_src),
                            tgt_val=_simple_clean(value_tgt),
                            out_src=self._canonicalize(local_name, _simple_clean(value_src)),
                            out_tgt=self._canonicalize(local_name, _simple_clean(value_tgt)),
                        )
                        examples.append(example)

        random.shuffle(examples)
        return examples

    def _canonicalize(self, predicate: str, value: str) -> str:
        """Lightweight canonicalisation for specific predicate families."""
        if not value:
            return value
        cleaned = value.strip()
        if any(key in predicate.lower() for key in ["name", "surname", "givenname", "birthname", "fullname"]):
            cleaned = " ".join(word.capitalize() for word in cleaned.split())
        return cleaned

    # ------------------------------------------------------------------
    # Latent interpolation
    # ------------------------------------------------------------------
    def _mean_pool(self, states: torch.Tensor, attention: Optional[torch.Tensor] = None) -> torch.Tensor:
        if states.dim() == 3:
            states = states.squeeze(0)
        if attention is None:
            return states.mean(0)
        mask = attention.squeeze(0).unsqueeze(-1).float()
        return (states * mask).sum(0) / mask.sum(0).clamp_min(1.0)

    def _adaptive_alpha(self, mean_src: torch.Tensor, mean_tgt: torch.Tensor) -> float:
        cosine = F.cosine_similarity(mean_src.unsqueeze(0), mean_tgt.unsqueeze(0)).item()
        similarity = (cosine + 1.0) / 2.0
        alpha = self.base_alpha + self.alpha_spread * (2 * similarity - 1)
        return max(0.05, min(0.95, alpha))

    def interpolate_pair(self, val_src: str, val_tgt: str, max_new_tokens: int = 32, predicate: str = "") -> Tuple[str, str]:
        """Interpolate two literal values into synthetic counterparts."""
        if not val_src and not val_tgt:
            return "", ""
        if not val_src:
            cleaned = _simple_clean(val_tgt)
            return cleaned, cleaned
        if not val_tgt:
            cleaned = _simple_clean(val_src)
            return cleaned, cleaned

        tokens = self.tokenizer(
            [val_src, val_tgt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len_in,
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            encoder_outputs = self.model.get_encoder()(
                tokens.input_ids,
                attention_mask=tokens.attention_mask,
            )

        hidden_src = encoder_outputs.last_hidden_state[0]
        hidden_tgt = encoder_outputs.last_hidden_state[1]
        mask_src = tokens.attention_mask[0]
        mask_tgt = tokens.attention_mask[1]

        mean_src = self._mean_pool(hidden_src, mask_src)
        mean_tgt = self._mean_pool(hidden_tgt, mask_tgt)
        alpha = self._adaptive_alpha(mean_src, mean_tgt)

        if any(
            keyword in (predicate or "").lower()
            for keyword in ["name", "givenname", "surname", "fullname", "birthname", "title"]
        ):
            alpha = max(0.10, min(alpha, 0.30))

        mixed_src = (1 - alpha) * hidden_src + alpha * hidden_tgt
        mixed_tgt = (1 - alpha) * hidden_tgt + alpha * hidden_src

        encoder_src = BaseModelOutput(last_hidden_state=mixed_src.unsqueeze(0))
        encoder_tgt = BaseModelOutput(last_hidden_state=mixed_tgt.unsqueeze(0))
        mask_src = mask_src.unsqueeze(0)
        mask_tgt = mask_tgt.unsqueeze(0)

        start_token = torch.tensor([[self.model.config.decoder_start_token_id]], device=self.device)
        bad_token_ids = self.tokenizer.convert_tokens_to_ids(["<SRC>", "<TGT>", "<SEP>"])
        generation_kwargs = dict(
            decoder_input_ids=start_token,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.2,
            num_beams=1,
            no_repeat_ngram_size=4,
            repetition_penalty=2.0,
            length_penalty=1.0,
            early_stopping=True,
            remove_invalid_values=True,
            bad_words_ids=[[token_id] for token_id in bad_token_ids],
        )

        with torch.no_grad():
            ids_src = self.model.generate(encoder_outputs=encoder_src, attention_mask=mask_src, **generation_kwargs)
            ids_tgt = self.model.generate(encoder_outputs=encoder_tgt, attention_mask=mask_tgt, **generation_kwargs)

        out_src = _simple_clean(self.tokenizer.decode(ids_src[0], skip_special_tokens=True))
        out_tgt = _simple_clean(self.tokenizer.decode(ids_tgt[0], skip_special_tokens=True))
        return out_src, out_tgt


__all__ = [
    "BartInterpolatorPLM",
    "PairExample",
    "_clean_pred",
]
