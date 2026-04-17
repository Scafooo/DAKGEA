"""MultiKE (Multi-view Knowledge Embedding) alignment model wrapper.

Wraps the legacy TF1-based MultiKE implementation located in the top-level
``MultiKE/`` directory by executing it in a subprocess.  Data is first
exported to the OpenEA/MultiKE directory layout expected by that codebase,
then the ITC variant (``MultiKE_CSL.MultiKE_CV``) is launched and its
printed metrics are captured and returned.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdflib import Literal, URIRef

from src.alignment_models.registry import MODEL_REGISTRY
from src.config.loader import PROJECT_ROOT, load_yaml
from src.core.dataset import Dataset
from src.logger import get_logger

logger = get_logger(__name__)

_MULTIKÉ_DIR = PROJECT_ROOT / "MultiKE"


def _write_tsv(path: Path, rows: List[Tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write("\t".join(str(x) for x in row) + "\n")


@MODEL_REGISTRY.register("multiKE")
class MultiKEAlignment:
    """Integration wrapper that executes the legacy MultiKE ITC pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.stage_config = config or {}
        self.model_config = self._load_model_config()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset_reduced: Dataset,
        dataset_augmented: Optional[Dataset],
    ) -> Dict[str, float]:
        dataset = dataset_augmented or dataset_reduced
        meta = self.stage_config.get("experiment", {})
        dataset_name = meta.get("dataset", "dataset")

        logger.info("[MultiKE] Evaluating dataset '%s' (aligned=%d)", dataset_name, len(dataset.aligned_entities))

        with tempfile.TemporaryDirectory(prefix="multiKE_") as tmp_str:
            tmp = Path(tmp_str)
            data_dir = tmp / dataset_name
            division = self.model_config.get("dataset_division", "631/")

            self._export_dataset(dataset, data_dir, division)
            args_path = self._write_args_json(data_dir, tmp / "output", division)
            metrics = self._run_multiKE(args_path, data_dir)

        logger.info("[MultiKE] Metrics: hits@1=%.4f hits@10=%.4f mrr=%.4f", metrics["hits@1"], metrics["hits@10"], metrics["mrr"])
        return metrics

    # ------------------------------------------------------------------
    # Data export
    # ------------------------------------------------------------------

    def _export_dataset(self, dataset: Dataset, data_dir: Path, division: str) -> None:
        """Write MultiKE-format files into *data_dir*."""
        data_dir.mkdir(parents=True, exist_ok=True)

        for kg, suffix in [
            (dataset.knowledge_graph_source, "1"),
            (dataset.knowledge_graph_target, "2"),
        ]:
            rel_triples: List[Tuple[str, str, str]] = []
            attr_triples: List[Tuple[str, str, str]] = []
            for s, p, o in kg:
                if isinstance(o, Literal):
                    attr_triples.append((str(s), str(p), str(o)))
                else:
                    rel_triples.append((str(s), str(p), str(o)))
            _write_tsv(data_dir / f"rel_triples_{suffix}", rel_triples)
            _write_tsv(data_dir / f"attr_triples_{suffix}", attr_triples)
            logger.debug("[MultiKE] KG%s: %d rel, %d attr triples", suffix, len(rel_triples), len(attr_triples))

        # Alignment splits – OpenEA 631 convention: 30% train, 60% test, 10% valid
        all_pairs = sorted((str(e1), str(e2)) for e1, e2 in dataset.aligned_entities)
        n = len(all_pairs)
        n_train = int(n * 0.3)
        n_test = int(n * 0.6)
        train_pairs = all_pairs[:n_train]
        test_pairs = all_pairs[n_train : n_train + n_test]
        valid_pairs = all_pairs[n_train + n_test :]

        split_dir = data_dir / division
        split_dir.mkdir(parents=True, exist_ok=True)
        _write_tsv(split_dir / "train_links", train_pairs)
        _write_tsv(split_dir / "test_links", test_pairs)
        _write_tsv(split_dir / "valid_links", valid_pairs)

        logger.info("[MultiKE] Alignment split: train=%d test=%d valid=%d", len(train_pairs), len(test_pairs), len(valid_pairs))

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _write_args_json(self, data_dir: Path, output_dir: Path, division: str) -> Path:
        """Produce an args.json compatible with MultiKE's ``load_args``."""
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.model_config

        args: Dict[str, Any] = {
            "training_data": str(data_dir) + "/",
            "output": str(output_dir) + "/",
            "word2vec_path": cfg.get("word2vec_path", ""),
            "dataset_division": division,
            "alignment_module": cfg.get("alignment_module", "swapping"),
            "encoder_epoch": cfg.get("encoder_epoch", 100),
            "encoder_active": cfg.get("encoder_active", "thah"),
            "encoder_normalize": cfg.get("encoder_normalize", True),
            "retrain_literal_embeds": cfg.get("retrain_literal_embeds", True),
            "literal_normalize": cfg.get("literal_normalize", True),
            "dim": cfg.get("dim", 75),
            "learning_rate": cfg.get("learning_rate", 0.001),
            "optimizer": cfg.get("optimizer", "Adagrad"),
            "max_epoch": cfg.get("max_epoch", 200),
            "shared_learning_max_epoch": cfg.get("shared_learning_max_epoch", 200),
            "batch_size": cfg.get("batch_size", 5000),
            "entity_batch_size": cfg.get("entity_batch_size", 5000),
            "attribute_batch_size": cfg.get("attribute_batch_size", 5000),
            "neg_triple_num": cfg.get("neg_triple_num", 10),
            "neg_sampling": cfg.get("neg_sampling", "truncated"),
            "truncated_epsilon": cfg.get("truncated_epsilon", 0.98),
            "truncated_freq": cfg.get("truncated_freq", 20),
            "batch_threads_num": cfg.get("batch_threads_num", 4),
            "test_threads_num": cfg.get("test_threads_num", 8),
            "start_valid": cfg.get("start_valid", 100),
            "eval_freq": cfg.get("eval_freq", 10),
            "stop_metric": cfg.get("stop_metric", "mrr"),
            "top_k": cfg.get("top_k", [1, 5, 10, 50]),
            "is_save": False,
            "orthogonal_weight": cfg.get("orthogonal_weight", 2),
            "cv_name_weight": cfg.get("cv_name_weight", 1),
            "cv_weight": cfg.get("cv_weight", 1),
            "start_predicate_soft_alignment": cfg.get("start_predicate_soft_alignment", 10),
            "predicate_soft_sim": cfg.get("predicate_soft_sim", 0.85),
            "predicate_init_sim": cfg.get("predicate_init_sim", 0.90),
            "relation_learning_rate": cfg.get("relation_learning_rate", 0.005),
            "ITC_learning_rate": cfg.get("ITC_learning_rate", 0.004),
        }

        args_path = output_dir / "args.json"
        with args_path.open("w", encoding="utf-8") as fh:
            json.dump(args, fh, indent=2)
        logger.debug("[MultiKE] args.json written to %s", args_path)
        return args_path

    # ------------------------------------------------------------------
    # Subprocess execution
    # ------------------------------------------------------------------

    def _run_multiKE(self, args_path: Path, data_dir: Path) -> Dict[str, float]:
        """Launch MultiKE ITC in a subprocess and return parsed metrics."""
        python_exe = self.model_config.get("python_executable", sys.executable)

        # Inline runner: inserts MultiKE/ onto sys.path and calls the pipeline
        runner_script = textwrap.dedent(f"""\
            import sys, os
            sys.path.insert(0, {repr(str(_MULTIKÉ_DIR))})
            os.chdir({repr(str(_MULTIKÉ_DIR))})

            from utils import load_args
            from data_model import DataModel
            from predicate_alignment import PredicateAlignModel
            from MultiKE_CSL import MultiKE_CV

            args = load_args({repr(str(args_path))})
            args.training_data = {repr(str(data_dir) + "/")}
            data = DataModel(args)
            attr_align_model = PredicateAlignModel(data.kgs, args)
            model = MultiKE_CV(data, args, attr_align_model)
            model.run()
        """)

        timeout = self.model_config.get("timeout_seconds", 7200)
        try:
            result = subprocess.run(
                [python_exe, "-c", runner_script],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error("[MultiKE] Process timed out after %d s", timeout)
            return {"hits@1": 0.0, "hits@10": 0.0, "mrr": 0.0}

        combined = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            logger.error("[MultiKE] Process exited with code %d\n%s", result.returncode, combined[-2000:])

        metrics = self._parse_metrics(combined)
        return metrics

    # ------------------------------------------------------------------
    # Metric parsing
    # ------------------------------------------------------------------

    def _parse_metrics(self, output: str) -> Dict[str, float]:
        """Extract the last reported hits@1, hits@10, mrr from MultiKE output.

        MultiKE prints two formats depending on *accurate* mode:
          ``quick results: hits@[1, 5, 10, 50] = [h1, h5, h10, h50]%, time = T s``
          ``accurate results: hits@[1, 5, 10, 50] = [h1, h5, h10, h50]%, mr = X, mrr = Y``
        """
        hits1, hits10, mrr = 0.0, 0.0, 0.0

        # "accurate results" carries mrr directly
        accurate_re = re.compile(
            r"accurate results.*?hits@\[.*?\] = \[([^\]]+)\]%.*?mrr = ([0-9.]+)"
        )
        # "quick results" only carries hits (mrr is not printed but computed)
        quick_re = re.compile(
            r"quick results.*?hits@\[.*?\] = \[([^\]]+)\]%"
        )

        for line in output.splitlines():
            m = accurate_re.search(line)
            if m:
                hits_list = [float(h.strip()) for h in m.group(1).split(",")]
                if len(hits_list) >= 3:
                    hits1 = hits_list[0] / 100.0
                    hits10 = hits_list[2] / 100.0
                mrr = float(m.group(2))
                continue

            m = quick_re.search(line)
            if m:
                hits_list = [float(h.strip()) for h in m.group(1).split(",")]
                if len(hits_list) >= 3:
                    hits1 = hits_list[0] / 100.0
                    hits10 = hits_list[2] / 100.0

        return {"hits@1": hits1, "hits@10": hits10, "mrr": mrr}

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def _load_model_config(self) -> Dict[str, Any]:
        path = PROJECT_ROOT / "config/models/multiKE.yaml"
        if not path.exists():
            return {}
        payload = load_yaml(path)
        model_section: Dict[str, Any] = payload.get("model", {})

        # Allow experiment-level overrides
        stage_override = self.stage_config.get("models", {}).get("multiKE", {})
        if stage_override:
            model_section = {**model_section, **stage_override}

        return model_section


__all__ = ["MultiKEAlignment"]
