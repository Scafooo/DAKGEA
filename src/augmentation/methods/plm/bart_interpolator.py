import os
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional, Dict, Any

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

from src.core.dataset import Dataset
from src.core.knowledge_graph import KnowledgeGraph

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
# Utils: semplice “noise injector”
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


def _clean_pred(p: str) -> str:
    # Tiene solo il “local name” del predicato
    # es: http://xmlns.com/foaf/0.1/name -> name ; dbo:birthPlace -> birthPlace
    if p is None:
        return ""
    m = re.split(r"[#/]", str(p))
    tail = m[-1] if m else str(p)
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

    Supporta:
      - fine-tuning "domain-aware" con denoising
      - α adattivo in base alla similarità
      - riuso automatico del modello già fine-tunato
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

        # Initialize advanced training modules
        self.advanced_config = advanced_training_config or {}
        self._init_advanced_modules()

        self.model, self.tokenizer = self._load_or_init_model()

    def _init_advanced_modules(self):
        """Initialize advanced training modules based on configuration."""
        import logging
        logger = logging.getLogger(__name__)

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
            )
            logger.info("[BART] Training augmentation enabled")
        else:
            self.training_augmenter = None

    # ------------------------------------------------------------------
    # Model loading / initialization logic
    # ------------------------------------------------------------------
    def _load_or_init_model(self):
        """
        Se esiste già un modello fine-tunato in out_dir → caricalo.
        Altrimenti inizializza un BART pre-addestrato.
        """
        if (
            self.reuse_if_available
            and os.path.isdir(self.out_dir)
            and any(f in os.listdir(self.out_dir) for f in ["pytorch_model.bin", "config.json"])
        ):
            print(f"[BART-PLM] Found fine-tuned model in {self.out_dir}, reusing it.")
            tok = BartTokenizer.from_pretrained(self.out_dir)
            mdl = BartForConditionalGeneration.from_pretrained(self.out_dir).to(self.device)
            specials = {'additional_special_tokens': ['<SRC>', '<TGT>', '<SEP>']}
            tok.add_special_tokens(specials)
            mdl.resize_token_embeddings(len(tok))
        else:
            print(f"[BART-PLM] Initializing from pretrained {self.model_name}.")
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

    def _build_hf_dataset_from_pairs(self, pairs, balance_by_predicate: bool = True):
        """
        Costruisce un Dataset HuggingFace dai PairExample.
        Opzionale: bilancia per predicato (round-robin semplice).
        """
        if not balance_by_predicate:
            rows = []
            for ex in pairs:
                rows.append({
                    "input_text": f"{ex.predicate} <SEP> {_noise_str(ex.src_val)} <SEP> {_noise_str(ex.tgt_val)}",
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
                    rows.append({
                        "input_text": f"{ex.predicate} <SEP> {_noise_str(ex.src_val)} <SEP> {_noise_str(ex.tgt_val)}",
                        "output_text": f"{ex.out_src} | {ex.out_tgt}",
                        "predicate": ex.predicate,
                    })

        return HFDataset.from_list(rows)

    def _apply_advanced_preprocessing(self, pairs: List[PairExample]) -> List[PairExample]:
        """Apply advanced training preprocessing: stratified sampling, augmentation, etc."""
        import logging
        logger = logging.getLogger(__name__)

        # 1. Training Data Augmentation
        if self.training_augmenter:
            pairs = self.training_augmenter.augment(pairs)

        # 2. Stratified Sampling
        if self.stratified_sampler:
            # Group by predicate
            by_predicate = defaultdict(list)
            for ex in pairs:
                by_predicate[ex.predicate].append(ex)
            pairs = self.stratified_sampler.sample(by_predicate)

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

        logger = logging.getLogger(__name__)

        # 🧩 skip se modello già addestrato e reuse abilitato
        if (
            self.reuse_if_available
            and not force_retrain
            and os.path.isdir(self.out_dir)
            and any(f in os.listdir(self.out_dir) for f in ["pytorch_model.bin", "config.json"])
        ):
            print(f"[BART-PLM] Skipping fine-tuning — model already exists in {self.out_dir}.")
            return

        # ✂️ opzionale: sottocampiona se troppo grande (before advanced preprocessing)
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
            def _noise_str(x: str) -> str:
                return self.advanced_noiser.noise(x) if x else x
        else:
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
                inp = f"{ex.predicate} <s> {_noise_str(ex.src_val)} <t> {_noise_str(ex.tgt_val)}"
                out = f"{ex.out_src} <t> {ex.out_tgt}"
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
        print(f"[BART-PLM] Tokenizing on {'GPU' if torch.cuda.is_available() else 'CPU'}...")

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

        args_kwargs = dict(
            output_dir=self.out_dir,
            overwrite_output_dir=force_retrain,
            num_train_epochs=epochs,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
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
        # Trainer setup (usa EarlyStopping solo se supportato)
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

        print(f"[BART-PLM] Starting fine-tuning "
              f"({'with early stopping' if has_eval_strategy else 'without evaluation'})...")

        trainer.train()

        # ------------------------------------------------------------------
        # Save final / best model
        # ------------------------------------------------------------------
        self.model.save_pretrained(self.out_dir)
        self.tokenizer.save_pretrained(self.out_dir)
        print(f"[BART-PLM] Fine-tuned model saved to {self.out_dir}")

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

        NOTE: Non si limita alle entità allineate - usa tutti gli attributi disponibili.
        """
        # Raccogli tutti i literal raggruppati per predicato (local name)
        def collect_literals_by_predicate(kg):
            """Returns: {predicate_local_name: [literal_values]}"""
            pred_to_literals = defaultdict(list)
            for s, p, o in kg.triples((None, None, None)):
                if isinstance(o, Literal):
                    lname = _clean_pred(p)
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

    # piccola canonicalizzazione euristica (solo minima per supervision)
    def _canonicalize(self, pred: str, val: str) -> str:
        if not val:
            return val
        v = val.strip()
        # nomi -> Title Case
        if any(k in pred.lower() for k in ["name", "surname", "givenname", "birthname", "fullname"]):
            v = " ".join(w.capitalize() for w in v.split())
        # date semplici (YYYY-MM-DD già ok), altrimenti non toccare qui
        return v


    # ------------------------------------------------------------------
    # Interpolazione con α adattivo
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
        # similarità coseno ∈ [-1, 1] -> rimappi a [0,1]
        cos = F.cosine_similarity(h1_mean.unsqueeze(0), h2_mean.unsqueeze(0)).item()
        sim01 = (cos + 1.0) / 2.0
        # α = base ± spread * (2*sim - 1)
        # se sim alta → α più grande (mischio di più), se bassa → α più piccolo
        alpha = self.base_alpha + self.alpha_spread * (2 * sim01 - 1)
        # clamp in [0.05, 0.95]
        return max(0.05, min(0.95, alpha))

    def interpolate_pair(self, val_src: str, val_tgt: str, max_new_tokens: int = 32, predicate: str = "") -> Tuple[
        str, str]:
        if not val_src and not val_tgt:
            return "", ""
        if not val_src:
            t = _simple_clean(val_tgt)
            return t, t
        if not val_tgt:
            s = _simple_clean(val_src)
            return s, s

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

        # alpha adattivo (più conservativo su nomi/titoli)
        m1 = self._mean_pool(h1, a1)
        m2 = self._mean_pool(h2, a2)
        alpha = self._adaptive_alpha(m1, m2)
        if any(k in (predicate or "").lower() for k in
               ["name", "givenname", "surname", "fullname", "birthname", "title"]):
            alpha = max(0.10, min(alpha, 0.30))

        # mix latente asimmetrico
        h_mix_src = (1 - alpha) * h1 + alpha * h2
        h_mix_tgt = (1 - alpha) * h2 + alpha * h1

        enc_src = BaseModelOutput(last_hidden_state=h_mix_src.unsqueeze(0))  # (1, seq, dim)
        enc_tgt = BaseModelOutput(last_hidden_state=h_mix_tgt.unsqueeze(0))  # (1, seq, dim)
        mask_src = a1.unsqueeze(0)  # (1, seq)
        mask_tgt = a2.unsqueeze(0)  # (1, seq)

        start = torch.tensor([[self.model.config.decoder_start_token_id]], device=self.device)
        bad = self.tokenizer.convert_tokens_to_ids(['<SRC>', '<TGT>', '<SEP>'])

        gen_kwargs = dict(
            decoder_input_ids=start,
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
            bad_words_ids=[[i] for i in bad],
        )

        with torch.no_grad():
            ids_src = self.model.generate(
                encoder_outputs=enc_src,
                **gen_kwargs
            )
            ids_tgt = self.model.generate(
                encoder_outputs=enc_tgt,
                **gen_kwargs
            )

        out_src = _simple_clean(self.tokenizer.decode(ids_src[0], skip_special_tokens=True))
        out_tgt = _simple_clean(self.tokenizer.decode(ids_tgt[0], skip_special_tokens=True))
        return out_src, out_tgt

