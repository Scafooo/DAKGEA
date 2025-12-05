import os
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional, Dict, Any
from src.logger import get_logger

logger = get_logger(__name__)
import numpy as np
import torch
from torch.nn import functional as F
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset as HFDataset
from rdflib import URIRef, Literal
from transformers.modeling_outputs import BaseModelOutput

from src.utils.reproducibility import set_random_seeds

from src.core.dataset import Dataset
from src.core.knowledge_graph import KnowledgeGraph

# Sentence-level interpolation for long texts
from .sentence_interpolator import (
    interpolate_long_text,
    is_long_text_predicate,
    count_tokens,
)

# Levenshtein distance for edit distance constraint
try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    # Fallback implementation if python-Levenshtein not installed
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

# Import advanced training modules
from src.augmentation.methods.plm.bart_training_modules import (
    StratifiedSampler,
    ContrastiveLoss,
    NegativeSampler,
    AttributeTypeClassifier,
    PredicateMatchClassifier,
    AttributeTypeInference,
    AdvancedNoiser,
    CurriculumScheduler,
    TrainingAugmenter,
    PairExample as AdvancedPairExample,
)

# Evita kernel SDPA in caso di maschere non standard/bug di broadcast
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass



# ----------------------------
# Utils: simple noise injector
# ----------------------------
def _noise_str(x: str) -> str:
    if not x:
        return x
    ops = []
    ops.append(lambda s: s.lower())
    ops.append(lambda s: re.sub(r"\s+", " ", s))
    ops.append(lambda s: re.sub(r"[^\w\s\.\-']", " ", s))
    ops.append(lambda s: re.sub(r"\b(\w+)( \1\b)+", r"\1", s, flags=re.IGNORECASE))
    ops.append(lambda s: s[: max(3, int(len(s) * random.uniform(0.7, 1.0)))])
    ops.append(lambda s: (" " + s) if random.random() < 0.3 else s)
    ops.append(lambda s: s + (("." if not s.endswith(".") else "")) if random.random() < 0.2 else s)
    # mescola 1–3 operazioni a caso
    k = random.randint(1, 3)
    for f in random.sample(ops, k):
        x = f(x)
    return re.sub(r"\s+", " ", x).strip(" .-")


def _clean_pred(p, kg=None) -> str:
    """Extract readable predicate name, preferring attr_to_name mapping when available.

    Args:
        p: Predicate URI (as URIRef or string)
        kg: Optional KnowledgeGraph with attr_to_name mapping

    Returns:
        Semantic predicate name if available (e.g., "date of birth"),
        otherwise local name (e.g., "P569")

    Examples:
        With attr_to_name: "http://www.wikidata.org/entity/P569" -> "date of birth"
        Without attr_to_name: "http://www.wikidata.org/entity/P569" -> "P569"
    """
    if p is None:
        return ""

    # Convert to string for lookup
    p_str = str(p)

    # Try attr_to_name mapping first (semantic names)
    if kg is not None and hasattr(kg, 'attr_to_name'):
        # Check if predicate URI is in the mapping
        if p_str in kg.attr_to_name:
            semantic_name = kg.attr_to_name[p_str]
            if semantic_name:
                return semantic_name

    # Fallback: extract local name (e.g., P569, birthPlace)
    m = re.split(r"[#/]", p_str)
    tail = m[-1] if m else p_str
    return tail.split(":")[-1]


# bart_interpolator.py
def _simple_clean(x: str) -> str:
    if not x:
        return x
    x = re.sub(r"http\S+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[^\w\s\.\-']", " ", x)
    x = re.sub(r"\b(\w+)( \1\b)+", r"\1", x, flags=re.IGNORECASE)  # dedup
    x = re.sub(r"\b([A-Za-z])\b", "", x)  # rimuovi parole di una sola lettera
    return re.sub(r"\s+", " ", x).strip(" .-")



# ------------------------------------------
# Dataset builder (da KG + entity alignments)
# ------------------------------------------
@dataclass
class PairExample:
    predicate: str
    src_val: str
    tgt_val: str
    # target desiderato per il fine-tuning (pulito)
    out_src: str
    out_tgt: str


class BartInterpolatorPLM:
    """
    BART-based latent interpolator + on-the-fly fine-tuner.

    Supports:
      - domain-aware fine-tuning with denoising
      - adaptive α based on similarity
      - automatic reuse of already fine-tuned model
    """

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
        advanced_training_config: Optional[Dict[str, Any]] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.out_dir = out_dir
        self.base_alpha = base_alpha
        self.alpha_spread = alpha_spread
        self.max_len_in = max_len_in
        self.max_len_out = max_len_out
        self.reuse_if_available = reuse_if_available
        self.training_config = training_config or {}

        # Set all random seeds for reproducibility
        set_random_seeds(seed)

        # Generation parameters (configurable)
        gen_cfg = generation_config or {}
        self.gen_max_new_tokens = int(gen_cfg.get("max_new_tokens", 32))
        self.gen_do_sample = bool(gen_cfg.get("do_sample", True))
        self.gen_top_k = int(gen_cfg.get("top_k", 50))
        self.gen_top_p = float(gen_cfg.get("top_p", 0.95))
        self.gen_temperature = float(gen_cfg.get("temperature", 1.2))
        self.gen_num_beams = int(gen_cfg.get("num_beams", 1))
        self.gen_repetition_penalty = float(gen_cfg.get("repetition_penalty", 2.0))
        self.gen_length_penalty = float(gen_cfg.get("length_penalty", 1.0))
        self.gen_no_repeat_ngram_size = int(gen_cfg.get("no_repeat_ngram_size", 4))

        # Edit distance constraint parameters
        self.enable_edit_distance_constraint = bool(gen_cfg.get("enable_edit_distance_constraint", False))
        self.max_edit_distance = int(gen_cfg.get("max_edit_distance", 5))
        self.num_candidates = int(gen_cfg.get("num_candidates", 10))  # For constraint generation

        # Token-level consistency (ensure shared tokens get same transformation)
        self.enable_token_consistency = bool(gen_cfg.get("enable_token_consistency", True))

        # Noise injection (force creativity when source=target)
        self.enable_noise_injection = bool(gen_cfg.get("enable_noise_injection", False))
        self.noise_std = float(gen_cfg.get("noise_std", 0.1))  # Standard deviation for gaussian noise
        self.noise_apply_when = str(gen_cfg.get("noise_apply_when", "identical_inputs"))  # "identical_inputs" or "always"

        # Retry mechanism for identical tokens
        self.enable_retry_on_identical_tokens = bool(gen_cfg.get("enable_retry_on_identical_tokens", False))
        self.max_retries = int(gen_cfg.get("max_retries", 3))
        self.noise_increment = float(gen_cfg.get("noise_increment", 0.05))  # Increase noise on each retry
        self.temperature_increment = float(gen_cfg.get("temperature_increment", 0.02))  # Increase temperature on each retry
        self.identical_tokens_threshold = float(gen_cfg.get("identical_tokens_threshold", 0.3))  # Overlap threshold for retry

        # Sentence-level interpolation for long texts
        self.enable_sentence_level = bool(gen_cfg.get("enable_sentence_level", True))
        self.sentence_chunk_max_tokens = int(gen_cfg.get("sentence_chunk_max_tokens", 80))
        self.sentence_min_length_for_chunking = int(gen_cfg.get("sentence_min_length_for_chunking", 60))

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Initialize advanced training modules
        self.advanced_config = advanced_training_config or {}
        self._init_advanced_modules()

        self.model, self.tokenizer = self._load_or_init_model()

    def _init_advanced_modules(self):
        """Initialize advanced training modules based on configuration."""

        # Stratified Sampling
        strat_cfg = self.advanced_config.get("stratified_sampling", {})
        if strat_cfg.get("enable", False):
            self.stratified_sampler = StratifiedSampler(
                min_samples=strat_cfg.get("min_samples_per_predicate", 10),
                max_samples=strat_cfg.get("max_samples_per_predicate", 1000),
                strategy=strat_cfg.get("balancing_strategy", "sqrt"),
            )
            logger.info("[BART] Stratified sampling enabled")
        else:
            self.stratified_sampler = None

        # Contrastive Learning
        contr_cfg = self.advanced_config.get("contrastive_learning", {})
        if contr_cfg.get("enable", False):
            self.contrastive_loss = ContrastiveLoss(temperature=contr_cfg.get("temperature", 0.07))
            self.negative_sampler = NegativeSampler(
                num_negatives=contr_cfg.get("num_negatives", 3),
                strategy=contr_cfg.get("negative_strategy", "same_predicate"),
            )
            self.contrastive_weight = contr_cfg.get("contrastive_weight", 0.3)
            logger.info("[BART] Contrastive learning enabled")
        else:
            self.contrastive_loss = None
            self.negative_sampler = None
            self.contrastive_weight = 0.0

        # Multi-task Learning
        mtl_cfg = self.advanced_config.get("multi_task_learning", {})
        if mtl_cfg.get("enable", False):
            self.use_multi_task = True
            self.predict_attr_type = mtl_cfg.get("predict_attribute_type", True)
            self.predict_pred_match = mtl_cfg.get("predict_predicate_match", True)
            self.auxiliary_weight = mtl_cfg.get("auxiliary_weight", 0.2)
            # Classifiers will be initialized after model is loaded
            self.attr_type_classifier = None
            self.pred_match_classifier = None
            logger.info("[BART] Multi-task learning enabled")
        else:
            self.use_multi_task = False
            self.predict_attr_type = False
            self.predict_pred_match = False
            self.auxiliary_weight = 0.0

        # Advanced Noising
        noise_cfg = self.advanced_config.get("advanced_noising", {})
        if noise_cfg.get("enable", False):
            self.advanced_noiser = AdvancedNoiser(
                span_corruption_ratio=noise_cfg.get("span_corruption_ratio", 0.3),
                mean_span_length=noise_cfg.get("mean_span_length", 3),
                entity_aware_masking=noise_cfg.get("entity_aware_masking", True),
                entity_mask_prob=noise_cfg.get("entity_mask_prob", 0.5),
            )
            logger.info("[BART] Advanced noising enabled")
        else:
            self.advanced_noiser = None

        # Curriculum Learning
        curr_cfg = self.advanced_config.get("curriculum_learning", {})
        if curr_cfg.get("enable", False):
            self.curriculum_scheduler = CurriculumScheduler(
                strategy=curr_cfg.get("strategy", "length"),
                num_phases=curr_cfg.get("num_phases", 3),
                phase_epochs=curr_cfg.get("phase_epochs", [3, 3, 4]),
            )
            logger.info("[BART] Curriculum learning enabled")
        else:
            self.curriculum_scheduler = None

        # Training Augmentation
        aug_cfg = self.advanced_config.get("training_augmentation", {})
        if aug_cfg.get("enable", False):
            self.training_augmenter = TrainingAugmenter(
                synonym_replacement=aug_cfg.get("synonym_replacement", False),
                back_translation=aug_cfg.get("back_translation", False),
                random_noise=aug_cfg.get("random_noise", False),
                augmentation_ratio=aug_cfg.get("augmentation_ratio", 0.3),
                noise_prob=aug_cfg.get("noise_prob", 0.1),
            )
            logger.info("[BART] Training augmentation enabled")
        else:
            self.training_augmenter = None

    # ------------------------------------------------------------------
    # Model loading / initialization logic
    # ------------------------------------------------------------------
    def _load_or_init_model(self):
        """
        If a fine-tuned model already exists in out_dir → load it.
        Otherwise initialize a pre-trained BART.
        """
        if (
            self.reuse_if_available
            and os.path.isdir(self.out_dir)
            and any(f in os.listdir(self.out_dir) for f in ["pytorch_model.bin", "config.json"])
        ):
            logger.warning(f"[BART-PLM] Found fine-tuned model in {self.out_dir}, reusing it.")
            tok = BartTokenizer.from_pretrained(self.out_dir)
            mdl = BartForConditionalGeneration.from_pretrained(self.out_dir).to(self.device)
            specials = {'additional_special_tokens': ['<SRC>', '<TGT>', '<SEP>']}
            tok.add_special_tokens(specials)
            mdl.resize_token_embeddings(len(tok))
        else:
            logger.info(f"[BART-PLM] Initializing from pretrained {self.model_name}.")
            tok = BartTokenizer.from_pretrained(self.model_name)
            mdl = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            specials = {'additional_special_tokens': ['<SRC>', '<TGT>', '<SEP>']}
            tok.add_special_tokens(specials)
            mdl.resize_token_embeddings(len(tok))
        return mdl, tok

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------
    from transformers import DataCollatorForSeq2Seq
    import math
    from collections import defaultdict

    def _build_input_with_context(self, ex: PairExample, noise_fn) -> str:
        """
        Build input text with context attributes if available.

        Format without context: "{predicate} <SEP> {noisy_src} <SEP> {noisy_tgt}"
        Format with context: "[attr1=val1|attr2=val2] {predicate} <SEP> {noisy_src} <SEP> {noisy_tgt}"

        Args:
            ex: PairExample (may have context_attrs attribute)
            noise_fn: Function to apply noise to values

        Returns:
            Formatted input string
        """
        # Check if context is available
        context_attrs = getattr(ex, 'context_attrs', None)

        if context_attrs and len(context_attrs) > 0:
            # Build context prefix: [attr1=val1|attr2=val2|...]
            # Apply noise to context values too for consistency
            context_pairs = [f"{k}={noise_fn(v)}" for k, v in sorted(context_attrs.items())]
            context_prefix = "[" + "|".join(context_pairs) + "] "
        else:
            context_prefix = ""

        # Build standard input with context prefix
        input_text = f"{context_prefix}{ex.predicate} <SEP> {noise_fn(ex.src_val)} <SEP> {noise_fn(ex.tgt_val)}"

        return input_text

    def _build_hf_dataset_from_pairs(self, pairs, balance_by_predicate: bool = True):
        """
        Costruisce un Dataset HuggingFace dai PairExample.
        Opzionale: bilancia per predicato (round-robin semplice).
        """
        if not balance_by_predicate:
            rows = []
            for ex in pairs:
                # Build input with context if available
                input_text = self._build_input_with_context(ex, _noise_str)
                rows.append({
                    "input_text": input_text,
                    "output_text": f"{ex.out_src} | {ex.out_tgt}",
                    "predicate": ex.predicate,
                })
            return HFDataset.from_list(rows)

        # bilanciamento: round-robin sui bucket per predicato
        buckets = defaultdict(list)
        for ex in pairs:
            buckets[ex.predicate].append(ex)

        # shuffle ogni bucket e interleava
        for k in buckets:
            random.shuffle(buckets[k])

        rows = []
        max_len = max(len(v) for v in buckets.values())
        keys = list(buckets.keys())
        for i in range(max_len):
            for k in keys:
                if i < len(buckets[k]):
                    ex = buckets[k][i]
                    # Build input with context if available
                    input_text = self._build_input_with_context(ex, _noise_str)
                    rows.append({
                        "input_text": input_text,
                        "output_text": f"{ex.out_src} | {ex.out_tgt}",
                        "predicate": ex.predicate,
                    })

        return HFDataset.from_list(rows)

    def _apply_advanced_preprocessing(self, pairs: List[PairExample]) -> List[PairExample]:
        """Apply advanced training preprocessing: stratified sampling, augmentation, etc."""
        import logging
        logger = get_logger(__name__)

        # 1. Training Data Augmentation
        if self.training_augmenter:
            pairs = self.training_augmenter.augment(pairs)

        # 2. Stratified Sampling
        if self.stratified_sampler:
            original_count = len(pairs)
            logger.info("[BART] Applying stratified sampling...")
            # Group by predicate
            by_predicate = defaultdict(list)
            for ex in pairs:
                by_predicate[ex.predicate].append(ex)
            pairs = self.stratified_sampler.sample(by_predicate)
            logger.info(f"[BART] Stratified sampling: {original_count} → {len(pairs)} examples")

        # 3. Curriculum Learning - assign difficulties
        if self.curriculum_scheduler:
            pairs = self.curriculum_scheduler.assign_difficulties(pairs)

        # 4. Index for contrastive learning
        if self.negative_sampler:
            self.negative_sampler.index_examples(pairs)
            logger.info(f"[BART] Indexed {len(pairs)} examples for contrastive learning")

        return pairs

    # ------------------------------------------------------------------
    # Fine-tuning con BART (GPU-safe + Early Stopping + Advanced Modules)
    # ------------------------------------------------------------------
    def fine_tune(
        self,
        pairs: List["PairExample"],
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 5e-5,
        max_train_samples: Optional[int] = 4000,
        val_split: float = 0.1,
        force_retrain: bool = False,
        num_proc: int = 2,
        patience: int = 3,   # numero di epoche senza miglioramento prima dello stop
    ):
        """
        Esegue fine-tuning di BART sul dataset di coppie (src,tgt) generate dai KGs,
        con early stopping automatico basato sulla loss di validazione.
        Supports advanced training modules (stratified sampling, contrastive learning, etc.)
        """
        import inspect
        import logging
        from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

        logger = get_logger(__name__)

        # Skip if model already trained and reuse enabled
        if (
            self.reuse_if_available
            and not force_retrain
            and os.path.isdir(self.out_dir)
            and any(f in os.listdir(self.out_dir) for f in ["pytorch_model.bin", "config.json"])
        ):
            logger.warning(f"[BART-PLM] Skipping fine-tuning — model already exists in {self.out_dir}.")
            return

        # Optional: subsample if too large (before advanced preprocessing)
        if max_train_samples and len(pairs) > max_train_samples:
            pairs = random.sample(pairs, max_train_samples)

        logger.info(f"[BART-PLM] Preparing fine-tuning dataset with {len(pairs)} examples...")

        # Apply advanced preprocessing
        pairs = self._apply_advanced_preprocessing(pairs)

        logger.info(f"[BART-PLM] After preprocessing: {len(pairs)} examples")

        # ------------------------------------------------------------------
        # Dataset creation + noise
        # ------------------------------------------------------------------
        # Use advanced noiser if available, otherwise use default
        if self.advanced_noiser:
            logger.info("[BART] Using advanced noising strategy")
            def _noise_str(x: str) -> str:
                return self.advanced_noiser.noise(x) if x else x
        else:
            logger.info("[BART] Using default noising strategy")
            def _noise_str(x: str) -> str:
                import re, random
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

        def to_rows(pairs):
            rows = []
            for ex in pairs:
                # Standard format - keep it simple!
                inp = f"{_noise_str(ex.src_val)} <sep> {_noise_str(ex.tgt_val)}"
                out = f"{ex.out_src} <sep> {ex.out_tgt}"
                rows.append({"input_text": inp, "output_text": out})
            return rows

        n_val = max(1, int(len(pairs) * val_split))
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]

        train_ds = HFDataset.from_list(to_rows(train_pairs))
        val_ds = HFDataset.from_list(to_rows(val_pairs))

        # ------------------------------------------------------------------
        # Preprocessing
        # ------------------------------------------------------------------
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        use_num_proc = None if torch.cuda.is_available() else num_proc
        logger.info(f"[BART-PLM] Tokenizing on {'GPU' if torch.cuda.is_available() else 'CPU'}...")

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

        # ------------------------------------------------------------------
        # TrainingArguments (retrocompatibile + EarlyStopping safe)
        # ------------------------------------------------------------------
        from transformers import TrainingArguments, Trainer
        import inspect

        has_eval_strategy = "evaluation_strategy" in inspect.signature(TrainingArguments).parameters

        if has_eval_strategy:
            eval_strategy = "epoch"
            save_strategy = "epoch"
            metric_for_best = "eval_loss"
            greater_is_better = False
        else:
            # versioni vecchie: disabilita early stopping
            eval_strategy = None
            save_strategy = "steps"
            metric_for_best = None
            greater_is_better = None

        # Get regularization parameters from config (with defaults)
        weight_decay = self.training_config.get("weight_decay", 0.01)
        warmup_steps = self.training_config.get("warmup_steps", 100)
        max_grad_norm = self.training_config.get("max_grad_norm", 1.0)

        args_kwargs = dict(
            output_dir=self.out_dir,
            overwrite_output_dir=force_retrain,
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            save_strategy=save_strategy,
            save_total_limit=2,
            logging_dir=os.path.join(self.out_dir, "logs"),
            logging_strategy="steps",
            logging_steps=50,
            report_to="none",
        )

        if has_eval_strategy:
            args_kwargs.update({
                "evaluation_strategy": eval_strategy,
                "load_best_model_at_end": True,
                "metric_for_best_model": metric_for_best,
                "greater_is_better": greater_is_better,
            })

        args = TrainingArguments(**args_kwargs)

        # ------------------------------------------------------------------
        # Trainer setup (use EarlyStopping only if supported)
        # ------------------------------------------------------------------
        from transformers import EarlyStoppingCallback

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

        logger.info(f"[BART-PLM] Starting fine-tuning "
              f"({'with early stopping' if has_eval_strategy else 'without evaluation'})...")

        trainer.train()

        # ------------------------------------------------------------------
        # Save final / best model
        # ------------------------------------------------------------------
        self.model.save_pretrained(self.out_dir)
        self.tokenizer.save_pretrained(self.out_dir)
        logger.info(f"[BART-PLM] Fine-tuned model saved to {self.out_dir}")

    # ------------------------------------------------------------------
    # Build supervised pairs (self-supervised, denoising auto-encoding)
    # ------------------------------------------------------------------
    def build_pairs_from_dataset(
        self,
        knowledge_graph_source,
        knowledge_graph_target,
        aligned_entities: Iterable[Tuple[URIRef, URIRef]],
        max_per_predicate: int = 5000,
    ) -> List[PairExample]:
        """
        Estrae coppie di training da TUTTI gli attributi dei due KG.

        Per ogni predicato (con stesso local name nei due grafi), raccoglie TUTTI i literal
        e crea coppie (src_val, tgt_val) per il fine-tuning auto-supervisionato:
          input  = "<pred> <SEP> noisy(src) <SEP> noisy(tgt)"
          output = "clean(src) | clean(tgt)"

        NOTE: Not limited to aligned entities - uses all available attributes.
        """
        # Collect all literals grouped by predicate (local name)
        def collect_literals_by_predicate(kg):
            """Returns: {predicate_local_name: [literal_values]}"""
            pred_to_literals = defaultdict(list)
            for s, p, o in kg.triples((None, None, None)):
                if isinstance(o, Literal):
                    lname = _clean_pred(p, kg)  # Pass kg to use attr_to_name mapping
                    pred_to_literals[lname].append(str(o))
            return pred_to_literals

        preds_src = collect_literals_by_predicate(knowledge_graph_source)
        preds_tgt = collect_literals_by_predicate(knowledge_graph_target)

        # Trova predicati in comune (stesso local name)
        common_predicates = set(preds_src.keys()) & set(preds_tgt.keys())

        # Calculate predicate frequencies for curriculum learning
        pred_frequencies = {lname: len(preds_src[lname]) + len(preds_tgt[lname])
                           for lname in common_predicates}
        max_freq = max(pred_frequencies.values()) if pred_frequencies else 1

        per_pred_count = {}
        examples: List[PairExample] = []

        # Per ogni predicato comune, crea coppie combinando i valori
        for lname in common_predicates:
            vals_src = preds_src[lname]
            vals_tgt = preds_tgt[lname]

            # Campiona casualmente coppie per evitare esplosione combinatoria
            # Se ci sono troppi valori, campiona un sottoinsieme
            max_vals = 100  # Limita il numero di valori per predicato
            if len(vals_src) > max_vals:
                vals_src = random.sample(vals_src, max_vals)
            if len(vals_tgt) > max_vals:
                vals_tgt = random.sample(vals_tgt, max_vals)

            # Infer predicate type once for all examples
            sample_val = vals_src[0] if vals_src else vals_tgt[0]
            pred_type = AttributeTypeInference.infer_type(lname, sample_val)

            # Crea coppie (possibilmente campionando)
            for v_src in vals_src:
                for v_tgt in vals_tgt:
                    # Limita esempi totali per predicato
                    c = per_pred_count.get(lname, 0)
                    if c >= max_per_predicate:
                        break
                    per_pred_count[lname] = c + 1

                    src_clean = _simple_clean(v_src)
                    tgt_clean = _simple_clean(v_tgt)

                    # Calculate difficulty for curriculum learning
                    difficulty = None
                    if self.curriculum_scheduler:
                        if self.curriculum_scheduler.strategy == "predicate_frequency":
                            # Normalize frequency to [0, 1], invert so rare = harder
                            difficulty = 1.0 - (pred_frequencies[lname] / max_freq)
                        else:
                            # Length-based difficulty (calculated later by scheduler)
                            difficulty = None

                    ex = PairExample(
                        predicate=lname,
                        src_val=src_clean,
                        tgt_val=tgt_clean,
                        out_src=self._canonicalize(lname, src_clean),
                        out_tgt=self._canonicalize(lname, tgt_clean),
                    )
                    # Add metadata as attributes (for compatibility)
                    ex.predicate_type = pred_type
                    ex.difficulty = difficulty
                    examples.append(ex)

                # Break outer loop too if we reached the limit
                if per_pred_count.get(lname, 0) >= max_per_predicate:
                    break

        random.shuffle(examples)
        return examples

    def build_pairs_from_dataset_with_context(
        self,
        knowledge_graph_source,
        knowledge_graph_target,
        aligned_entities: Iterable[Tuple[URIRef, URIRef]],
        max_per_predicate: int = 5000,
        max_context_attrs: int = 10,  # Limit context size
    ) -> List[PairExample]:
        """
        Build training pairs with context from other attributes of the same entity.

        For each aligned entity pair, collects ALL literal attributes and creates
        training examples where each attribute uses OTHER attributes as context.

        Format: "[attr1=val1|attr2=val2|...] src_val <SEP> tgt_val" → "interpolated"

        Args:
            knowledge_graph_source: Source KG
            knowledge_graph_target: Target KG
            aligned_entities: List of aligned (src_uri, tgt_uri) pairs
            max_per_predicate: Maximum examples per predicate
            max_context_attrs: Maximum attributes to include in context

        Returns:
            List of PairExample with context information
        """
        import logging
        logger = get_logger(__name__)

        logger.info("[BART] Building context-aware training pairs...")

        def collect_entity_literals(kg, entity_uri):
            """Collect all literal attributes for an entity."""
            attrs = {}
            for s, p, o in kg.triples((entity_uri, None, None)):
                if isinstance(o, Literal):
                    lname = _clean_pred(p, kg)  # Pass kg to use attr_to_name mapping
                    attrs[lname] = str(o)
            return attrs

        examples: List[PairExample] = []
        per_pred_count = {}

        # Process each aligned entity pair
        for src_uri, tgt_uri in aligned_entities:
            # Collect all attributes for both entities
            src_attrs = collect_entity_literals(knowledge_graph_source, src_uri)
            tgt_attrs = collect_entity_literals(knowledge_graph_target, tgt_uri)

            # Find common predicates (attributes that exist in both)
            common_preds = set(src_attrs.keys()) & set(tgt_attrs.keys())

            if not common_preds:
                continue

            # For each common predicate, create a training example with context
            for pred_to_generate in common_preds:
                # Check if we've reached the limit for this predicate
                if per_pred_count.get(pred_to_generate, 0) >= max_per_predicate:
                    continue

                # Build context from OTHER attributes (exclude the one we're generating)
                context_attrs = {}
                for pred, val in src_attrs.items():
                    if pred != pred_to_generate and len(context_attrs) < max_context_attrs:
                        context_attrs[pred] = _simple_clean(val)

                # Get the values to interpolate
                src_val = _simple_clean(src_attrs[pred_to_generate])
                tgt_val = _simple_clean(tgt_attrs[pred_to_generate])

                # Create example with context
                ex = PairExample(
                    predicate=pred_to_generate,
                    src_val=src_val,
                    tgt_val=tgt_val,
                    out_src=self._canonicalize(pred_to_generate, src_val),
                    out_tgt=self._canonicalize(pred_to_generate, tgt_val),
                )

                # Store context as attribute for later use
                ex.context_attrs = context_attrs

                # Infer type
                ex.predicate_type = AttributeTypeInference.infer_type(pred_to_generate, src_val)
                ex.difficulty = None  # Can be assigned later by curriculum scheduler

                examples.append(ex)
                per_pred_count[pred_to_generate] = per_pred_count.get(pred_to_generate, 0) + 1

        logger.info(f"[BART] Built {len(examples)} context-aware training pairs from {len(list(aligned_entities))} aligned entities")

        random.shuffle(examples)
        return examples

    # Small heuristic canonicalization (minimal for supervision)
    def _canonicalize(self, pred: str, val: str) -> str:
        if not val:
            return val
        v = val.strip()
        # names -> Title Case
        if any(k in pred.lower() for k in ["name", "surname", "givenname", "birthname", "fullname"]):
            v = " ".join(w.capitalize() for w in v.split())
        # simple dates (YYYY-MM-DD already ok), otherwise don't touch here
        return v


    # ------------------------------------------------------------------
    # Interpolation with adaptive α
    # ------------------------------------------------------------------
    def _mean_pool(self, H: torch.Tensor, attn: Optional[torch.Tensor] = None) -> torch.Tensor:
        # H: (seq, dim) o (1, seq, dim)
        if H.dim() == 3:
            H = H.squeeze(0)
        if attn is None:
            return H.mean(0)
        mask = attn.squeeze(0).unsqueeze(-1).float()  # (seq, 1)
        return (H * mask).sum(0) / mask.sum(0).clamp_min(1.0)

    def _adaptive_alpha(self, h1_mean: torch.Tensor, h2_mean: torch.Tensor) -> float:
        # cosine similarity ∈ [-1, 1] -> remap to [0,1]
        cos = F.cosine_similarity(h1_mean.unsqueeze(0), h2_mean.unsqueeze(0)).item()
        sim01 = (cos + 1.0) / 2.0
        # α = base ± spread * (2*sim - 1)
        # if high similarity → larger α (mix more), if low → smaller α
        alpha = self.base_alpha + self.alpha_spread * (2 * sim01 - 1)
        # clamp in [0.05, 0.95]
        return max(0.05, min(0.95, alpha))

    def interpolate_pair(
        self,
        val_src: str,
        val_tgt: str,
        max_new_tokens: int = 32,
        predicate: str = "",
    ) -> Tuple[str, str]:
        """
        Interpolate between source and target values using latent mixing.

        Args:
            val_src: Source value
            val_tgt: Target value
            max_new_tokens: Maximum tokens to generate
            predicate: Predicate name (for conservative alpha on names)

        Returns:
            Tuple of (interpolated_src, interpolated_tgt)
        """
        if not val_src and not val_tgt:
            return "", ""
        if not val_src:
            t = _simple_clean(val_tgt)
            return t, t
        if not val_tgt:
            s = _simple_clean(val_src)
            return s, s

        # Check if we should use sentence-level interpolation for long texts
        if self.enable_sentence_level and is_long_text_predicate(predicate):
            # Check text length
            max_len = max(
                count_tokens(val_src, self.tokenizer),
                count_tokens(val_tgt, self.tokenizer)
            )

            if max_len >= self.sentence_min_length_for_chunking:
                logger.debug(f"[INTERPOLATE_PAIR] Using sentence-level interpolation for predicate '{predicate}' ({max_len} tokens)")

                # Create a wrapper function for standard interpolation
                def _standard_interpolate(text1: str, text2: str) -> Tuple[str, str]:
                    if self.enable_retry_on_identical_tokens and self.enable_noise_injection:
                        return self._interpolate_with_retry(text1, text2, max_new_tokens, predicate)
                    else:
                        return self._interpolate_single(text1, text2, max_new_tokens, predicate)

                # Use sentence-level interpolation
                return interpolate_long_text(
                    val_src,
                    val_tgt,
                    _standard_interpolate,
                    self.tokenizer,
                    max_tokens=self.sentence_chunk_max_tokens,
                    min_length_for_chunking=self.sentence_min_length_for_chunking,
                )

        # Standard interpolation for short texts or non-long-text predicates
        if self.enable_retry_on_identical_tokens and self.enable_noise_injection:
            return self._interpolate_with_retry(val_src, val_tgt, max_new_tokens, predicate)
        else:
            return self._interpolate_single(val_src, val_tgt, max_new_tokens, predicate)

    def _calculate_token_overlap(self, input_text: str, output_text: str) -> float:
        """Calculate percentage of identical tokens between input and output (excluding stopwords).

        Returns:
            Overlap ratio (0.0-1.0): number of identical tokens / total output tokens
        """
        input_tokens = set(input_text.lower().split())
        output_tokens_list = output_text.lower().split()

        # Filter out only common stopwords (but keep short tokens like "al", "di", etc.)
        stopwords = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'of', 'and', 'or', 'is', 'are', 'was', 'were', 'be', 'been', 'by'}
        input_tokens = {t for t in input_tokens if t not in stopwords}
        output_tokens_filtered = [t for t in output_tokens_list if t not in stopwords]

        if not output_tokens_filtered:
            return 0.0

        # Count identical tokens in output
        identical_count = sum(1 for t in output_tokens_filtered if t in input_tokens)
        overlap_ratio = identical_count / len(output_tokens_filtered)

        return overlap_ratio

    def _calculate_output_similarity(self, out_src: str, out_tgt: str) -> float:
        """Calculate similarity between two outputs.

        Returns:
            Similarity ratio (0.0-1.0): Jaccard similarity of tokens
        """
        tokens_src = set(out_src.lower().split())
        tokens_tgt = set(out_tgt.lower().split())

        # Filter out only common stopwords (but keep short tokens like "al", "di", etc.)
        stopwords = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'of', 'and', 'or', 'is', 'are', 'was', 'were', 'be', 'been', 'by'}
        tokens_src = {t for t in tokens_src if t not in stopwords}
        tokens_tgt = {t for t in tokens_tgt if t not in stopwords}

        if not tokens_src and not tokens_tgt:
            return 0.0
        if not tokens_src or not tokens_tgt:
            return 0.0

        # Jaccard similarity
        intersection = tokens_src & tokens_tgt
        union = tokens_src | tokens_tgt

        return len(intersection) / len(union) if union else 0.0

    def _has_identical_tokens(self, input_text: str, output_text: str) -> bool:
        """Check if output contains identical tokens above threshold."""
        overlap = self._calculate_token_overlap(input_text, output_text)
        return overlap > self.identical_tokens_threshold

    def _get_identical_tokens(self, input_text: str, output_text: str) -> List[str]:
        """Get list of identical tokens between input and output (for blocking)."""
        input_tokens = set(input_text.lower().strip().split())
        output_tokens = set(output_text.lower().strip().split())

        # Filter out very short tokens (1-2 chars) and common stopwords
        stopwords = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'of', 'and', 'or', 'is', 'are', 'was', 'were', 'be', 'been', 'by'}
        input_tokens = {t for t in input_tokens if len(t) > 2 and t not in stopwords}
        output_tokens = {t for t in output_tokens if len(t) > 2 and t not in stopwords}

        identical = input_tokens & output_tokens
        return list(identical)

    def _interpolate_with_retry(
        self,
        val_src: str,
        val_tgt: str,
        max_new_tokens: int = 32,
        predicate: str = "",
    ) -> Tuple[str, str]:
        """Interpolate with retry mechanism if output contains identical tokens."""
        original_noise_std = self.noise_std
        original_temperature = self.gen_temperature
        blocked_tokens = []

        # Track best attempt (lowest overlap)
        best_out_src, best_out_tgt = None, None
        best_overlap = float('inf')

        for attempt in range(self.max_retries):
            # Generate output - force noise on retry attempts (attempt > 0)
            force_noise = (attempt > 0)
            out_src, out_tgt = self._interpolate_single(
                val_src, val_tgt, max_new_tokens, predicate,
                force_noise=force_noise,
                blocked_tokens=blocked_tokens if attempt > 0 else None
            )

            # Check for identical tokens between input and output
            overlap_src = self._calculate_token_overlap(val_src, out_src)
            overlap_tgt = self._calculate_token_overlap(val_tgt, out_tgt)

            max_overlap = max(overlap_src, overlap_tgt)
            has_identical_src = overlap_src > self.identical_tokens_threshold
            has_identical_tgt = overlap_tgt > self.identical_tokens_threshold

            # Debug log for the first attempt
            if attempt == 0:
                logger.debug(f"[RETRY_CHECK] Input: '{val_src}' / '{val_tgt}' → Output: '{out_src}' / '{out_tgt}'")
                logger.debug(f"[RETRY_CHECK] Overlaps: src={overlap_src:.1%} tgt={overlap_tgt:.1%} (threshold={self.identical_tokens_threshold:.1%})")
                logger.debug(f"[RETRY_CHECK] Will retry: {has_identical_src or has_identical_tgt}")

            # Track best attempt
            if max_overlap < best_overlap:
                best_overlap = max_overlap
                best_out_src, best_out_tgt = out_src, out_tgt

            if has_identical_src or has_identical_tgt:
                if attempt < self.max_retries - 1:
                    # Extract identical tokens to block in next attempt
                    identical_src = self._get_identical_tokens(val_src, out_src)
                    identical_tgt = self._get_identical_tokens(val_tgt, out_tgt)
                    blocked_tokens = list(set(identical_src + identical_tgt))

                    # Increase noise and temperature for next retry
                    self.noise_std += self.noise_increment
                    self.gen_temperature += self.temperature_increment
                    logger.debug(f"[RETRY] Attempt {attempt + 1}/{self.max_retries}: Overlap src={overlap_src:.1%} tgt={overlap_tgt:.1%} > threshold={self.identical_tokens_threshold:.1%}")
                    logger.debug(f"[CONSTRAINED_DECODING] Will block tokens: {blocked_tokens}")
                    logger.debug(f"[NOISE_INJECTION] Increasing noise to {self.noise_std:.3f}")
                    logger.debug(f"[TEMPERATURE_INJECTION] Increasing temperature to {self.gen_temperature:.3f}")
                    logger.debug(f"  Input: '{val_src}' / '{val_tgt}' → Output: '{out_src}' / '{out_tgt}'")
                    continue
                else:
                    logger.debug(f"[RETRY] Max retries reached, using best attempt (overlap: {best_overlap:.1%})")

            # Success - restore original parameters and return
            self.noise_std = original_noise_std
            self.gen_temperature = original_temperature
            return out_src, out_tgt

        # Max retries reached - restore parameters and return best attempt
        self.noise_std = original_noise_std
        self.gen_temperature = original_temperature
        logger.debug(f"[RETRY] Returning best attempt: '{best_out_src}' / '{best_out_tgt}' (overlap: {best_overlap:.1%})")
        return best_out_src, best_out_tgt

    def _interpolate_single(
        self,
        val_src: str,
        val_tgt: str,
        max_new_tokens: int = 32,
        predicate: str = "",
        force_noise: bool = False,
        blocked_tokens: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """Single interpolation attempt (original logic).

        Args:
            blocked_tokens: List of tokens to block from generation (constrained decoding)
        """

        # TOKEN-LEVEL CONSISTENCY with FORCED DECODING:
        # 1. Generate source with latent mixing
        # 2. Identify token alignment input→output
        # 3. Generate target FORCING same tokens for shared positions

        # Initialize variables for forced decoding
        shared_tokens = set()
        src_token_positions = {}
        tgt_token_positions = {}

        if self.enable_token_consistency:
            # Step 1: Tokenize inputs to identify shared tokens and their positions
            src_tokens = val_src.lower().split()
            tgt_tokens = val_tgt.lower().split()

            # Find shared tokens and their positions
            shared_tokens = set(src_tokens) & set(tgt_tokens)
            src_token_positions = {token: idx for idx, token in enumerate(src_tokens) if token in shared_tokens}
            tgt_token_positions = {token: idx for idx, token in enumerate(tgt_tokens) if token in shared_tokens}

            logger.verbose(f"[FORCED_DECODING] Input: '{val_src}' + '{val_tgt}'")
            logger.verbose(f"[FORCED_DECODING] Shared tokens: {shared_tokens}")
            logger.verbose(f"[FORCED_DECODING] Source positions: {src_token_positions}")
            logger.verbose(f"[FORCED_DECODING] Target positions: {tgt_token_positions}")

        # Standard approach: tokenize values directly
        toks = self.tokenizer([val_src, val_tgt], return_tensors="pt",
                              padding=True, truncation=True, max_length=self.max_len_in).to(self.device)

        self.model.eval()
        with torch.no_grad():
            enc = self.model.get_encoder()(toks.input_ids, attention_mask=toks.attention_mask)

        # split hidden & masks
        h1 = enc.last_hidden_state[0]  # (seq1, dim)
        h2 = enc.last_hidden_state[1]  # (seq2, dim)
        a1 = toks.attention_mask[0]  # (seq1,)
        a2 = toks.attention_mask[1]  # (seq2,)

        # adaptive alpha (more conservative on names/titles)
        m1 = self._mean_pool(h1, a1)
        m2 = self._mean_pool(h2, a2)
        alpha = self._adaptive_alpha(m1, m2)

        # TODO FIX MAYBE CAN BE INFERRED BY DATA
        if any(k in (predicate or "").lower() for k in
               ["name", "givenname", "surname", "fullname", "birthname", "title"]):
            alpha = max(0.10, min(alpha, 0.30))

        # asymmetric latent mix
        h_mix_src = (1 - alpha) * h1 + alpha * h2
        h_mix_tgt = (1 - alpha) * h2 + alpha * h1

        # Noise injection (force creativity when inputs are identical or when retrying)
        if self.enable_noise_injection:
            should_inject = False
            if force_noise:
                # Force noise during retry attempts
                should_inject = True
            elif self.noise_apply_when == "identical_inputs":
                # Only inject when source and target are identical
                should_inject = (val_src.lower().strip() == val_tgt.lower().strip())
            elif self.noise_apply_when == "always":
                # Always inject noise
                should_inject = True

            if should_inject:
                noise_src = torch.randn_like(h_mix_src) * self.noise_std
                noise_tgt = torch.randn_like(h_mix_tgt) * self.noise_std
                h_mix_src = h_mix_src + noise_src
                h_mix_tgt = h_mix_tgt + noise_tgt
                reason = "retry" if force_noise else "identical inputs"
                logger.verbose(f"[NOISE_INJECTION] Injected noise (std={self.noise_std:.3f}) for {reason}: '{val_src}' / '{val_tgt}'")

        enc_src = BaseModelOutput(last_hidden_state=h_mix_src.unsqueeze(0))  # (1, seq, dim)
        enc_tgt = BaseModelOutput(last_hidden_state=h_mix_tgt.unsqueeze(0))  # (1, seq, dim)
        mask_src = a1.unsqueeze(0)  # (1, seq)
        mask_tgt = a2.unsqueeze(0)  # (1, seq)

        start = torch.tensor([[self.model.config.decoder_start_token_id]], device=self.device)
        bad = self.tokenizer.convert_tokens_to_ids(['<SRC>', '<TGT>', '<SEP>'])

        # Add blocked tokens to bad_words_ids (constrained decoding)
        bad_words_ids_list = [[i] for i in bad]
        if blocked_tokens:
            # Tokenize blocked tokens and add to bad_words list
            for token in blocked_tokens:
                # Get token IDs for the blocked token (handle subword tokenization)
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                for tid in token_ids:
                    if [tid] not in bad_words_ids_list:
                        bad_words_ids_list.append([tid])
            logger.debug(f"[CONSTRAINED_DECODING] Blocking {len(blocked_tokens)} tokens: {blocked_tokens}")

        # Use configurable generation parameters
        gen_kwargs = dict(
            decoder_input_ids=start,
            max_new_tokens=self.gen_max_new_tokens if max_new_tokens == 32 else max_new_tokens,
            do_sample=self.gen_do_sample,
            top_k=self.gen_top_k,
            top_p=self.gen_top_p,
            temperature=self.gen_temperature,
            num_beams=self.gen_num_beams,
            no_repeat_ngram_size=self.gen_no_repeat_ngram_size,
            repetition_penalty=self.gen_repetition_penalty,
            length_penalty=self.gen_length_penalty,
            early_stopping=True,
            remove_invalid_values=True,
            bad_words_ids=bad_words_ids_list,
        )

        # If edit distance constraint is enabled, generate multiple candidates
        if self.enable_edit_distance_constraint:
            # Ensure num_beams >= num_candidates for diversity
            constraint_gen_kwargs = gen_kwargs.copy()
            constraint_gen_kwargs["num_beams"] = max(self.gen_num_beams, self.num_candidates)
            constraint_gen_kwargs["num_return_sequences"] = self.num_candidates
            constraint_gen_kwargs["do_sample"] = False  # Use beam search for better quality

            with torch.no_grad():
                ids_src_candidates = self.model.generate(
                    encoder_outputs=enc_src,
                    **constraint_gen_kwargs
                )
                ids_tgt_candidates = self.model.generate(
                    encoder_outputs=enc_tgt,
                    **constraint_gen_kwargs
                )

            # Filter candidates based on edit distance
            def select_best_candidate(candidates_ids, original_text):
                """Select best candidate that satisfies edit distance constraint."""
                valid_candidates = []
                for candidate_ids in candidates_ids:
                    candidate = _simple_clean(self.tokenizer.decode(candidate_ids, skip_special_tokens=True))
                    distance = levenshtein_distance(candidate.lower(), original_text.lower())
                    if distance <= self.max_edit_distance:
                        valid_candidates.append((candidate, distance))

                if valid_candidates:
                    # Return candidate with smallest edit distance (closest to original)
                    return min(valid_candidates, key=lambda x: x[1])[0]
                else:
                    # If no candidates satisfy constraint, return the one with smallest distance anyway
                    all_candidates = [
                        (_simple_clean(self.tokenizer.decode(cand, skip_special_tokens=True)),
                         levenshtein_distance(_simple_clean(self.tokenizer.decode(cand, skip_special_tokens=True)).lower(), original_text.lower()))
                        for cand in candidates_ids
                    ]
                    logger.warning(f"[EDIT_DISTANCE] No candidates within max_edit_distance={self.max_edit_distance} "
                                   f"for '{original_text}'. Using closest candidate.")
                    return min(all_candidates, key=lambda x: x[1])[0]

            out_src = select_best_candidate(ids_src_candidates, val_src)
            out_tgt = select_best_candidate(ids_tgt_candidates, val_tgt)

        else:
            # Standard generation (single candidate)

            # STEP 1: Generate source with latent mixing
            with torch.no_grad():
                ids_src = self.model.generate(
                    encoder_outputs=enc_src,
                    **gen_kwargs
                )
            out_src = _simple_clean(self.tokenizer.decode(ids_src[0], skip_special_tokens=True))

            # STEP 2: Generate target normally, then apply token consistency via post-processing
            with torch.no_grad():
                ids_tgt = self.model.generate(
                    encoder_outputs=enc_tgt,
                    **gen_kwargs
                )
            out_tgt = _simple_clean(self.tokenizer.decode(ids_tgt[0], skip_special_tokens=True))

            # STEP 3: If token consistency enabled, post-process target to match source transformations
            if self.enable_token_consistency and shared_tokens:
                src_output_tokens = out_src.split()  # Keep original case
                tgt_output_tokens = out_tgt.split()

                # Create mapping: input token → output token (from source)
                token_mapping = {}
                for input_token, src_pos in src_token_positions.items():
                    if src_pos < len(src_output_tokens):
                        output_token = src_output_tokens[src_pos]
                        token_mapping[input_token] = output_token
                    else:
                        # Output generated fewer tokens than input - cannot map this position
                        logger.verbose(
                            "[TOKEN_CONSISTENCY] Cannot map '%s' at position %s: output has only %s tokens (valid indices: 0-%s). Skipping this token.",
                            input_token,
                            src_pos,
                            len(src_output_tokens),
                            len(src_output_tokens) - 1,
                        )

                logger.verbose(f"[TOKEN_CONSISTENCY] Source output: '{out_src}'")
                logger.verbose(f"[TOKEN_CONSISTENCY] Target output (before): '{out_tgt}'")
                logger.verbose(f"[TOKEN_CONSISTENCY] Token mapping: {token_mapping}")

                # Apply consistent transformations to target
                for input_token, tgt_pos in tgt_token_positions.items():
                    if input_token in token_mapping:
                        if tgt_pos < len(tgt_output_tokens):
                            # Replace target token with source transformation
                            old_token = tgt_output_tokens[tgt_pos]
                            tgt_output_tokens[tgt_pos] = token_mapping[input_token]
                            logger.verbose(f"[TOKEN_CONSISTENCY] Position {tgt_pos}: '{old_token}' → '{token_mapping[input_token]}'")
                        else:
                            logger.verbose(f"[TOKEN_CONSISTENCY] Cannot apply mapping for '{input_token}' at position {tgt_pos}: "
                                        f"target output has only {len(tgt_output_tokens)} tokens (valid indices: 0-{len(tgt_output_tokens)-1})")
                    else:
                        logger.verbose(f"[TOKEN_CONSISTENCY] Skipping '{input_token}' at position {tgt_pos}: no mapping available from source")

                out_tgt = ' '.join(tgt_output_tokens)
                logger.verbose(f"[TOKEN_CONSISTENCY] Target output (after): '{out_tgt}'")

        return out_src, out_tgt
