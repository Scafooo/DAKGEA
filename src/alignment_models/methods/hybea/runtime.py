"""Runtime configuration bridge for the legacy HybEA components."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from .configuration import AttributeConfig, HybeaConfig, DatasetSpec, StructureConfig


_NAME_FILES = {
    "D_W_15K_V1": ("DBpedia_names.xlsx", "Wikidata_names.xlsx"),
    "D_W_15K_V2": ("DBpedia_names.xlsx", "Wikidata_names.xlsx"),
    "SRPRS_D_W_15K_V1": ("DBpedia_names.xlsx", "Wikidata_names.xlsx"),
    "SRPRS_D_W_15K_V2": ("DBpedia_names.xlsx", "Wikidata_names.xlsx"),
    "BBC_DB": ("BBC_names.xlsx", "DBpedia_names.xlsx"),
    "fr_en": ("fr_names.xlsx", "en_names.xlsx"),
    "ja_en": ("ja_names.xlsx", "en_names.xlsx"),
    "zh_en": ("zh_names.xlsx", "en_names.xlsx"),
    "ICEW_WIKI": ("icew_names.xlsx", "wiki_names.xlsx"),
    "ICEW_YAGO": ("icew_names.xlsx", "yago_names.xlsx"),
}


@dataclass
class RuntimeState:
    MODE: str = "Hybea"
    STRUCTURAL_MODEL: str = "Knowformer"
    MODEL: str = "hybea"
    DATASET: str = "dataset"
    SEED: int = 42
    SEED_NUM: int = 11037
    SIZE_AFTER_REDUCTION: float = 1.0
    SIZE_AFTER_REDUCTION_IN_PERCENTAGE: float = 100.0

    BASE_DIR: str = "."
    DATA_DIR: str = "."
    RAW_DATA_DIR: str = "."
    PROCESSED_DATA_DIR: str = "."
    AUGMENTED_DATA_DIR: str = "."
    REDUCED_DATA_DIR: str = "."
    DATA_TARGET: str = "."
    RESULT_DIR: str = "."
    DATA_PATH: str = "attribute_data/"

    CUDA_NUM: int = 0

    MODEL_INPUT_DIM: int = 768
    EPOCH_NUM: int = 200
    NEAREST_SAMPLE_NUM: int = 128
    CANDIDATE_GENERATOR_BATCH_SIZE: int = 128
    NEG_NUM: int = 2
    MARGIN: float = 3.0
    LEARNING_RATE: float = 1e-5
    TRAIN_BATCH_SIZE: int = 24
    TEST_BATCH_SIZE: int = 128
    CSLS: int = 2

    RANDOM_INITIALIZATION: bool = False
    HIDDEN_SIZE: int = 768
    NUM_HIDDEN_LAYERS: int = 12
    NUM_ATTENTION_HEADS: int = 4
    INPUT_DROPOUT_PROB: float = 0.5
    ATTENTION_DROPOUT_PROB: float = 0.1
    HIDDEN_DROPOUT_PROB: float = 0.3
    RESIDUAL_DROPOUT_PROB: float = 0.1
    INITIALIZER_RANGE: float = 0.02
    INTERMEDIATE_SIZE: int = 2048
    RESIDUAL_W: float = 0.5
    EPOCH: int = 200
    MIN_EPOCHS: int = 10
    LEARNING_RATE_STRUCTURE: float = 5e-4
    BATCH_SIZE: int = 2048
    EVAL_BATCH_SIZE: int = 4096
    EARLY_STOP_MAX_TIMES: int = 3
    SOFT_LABEL: float = 0.25
    EVAL_FREQ: int = 5
    START_EVAL: int = 0
    SWA_PRE_NUM: int = 5
    DO_TRAIN: bool = True
    DO_TEST: bool = True
    USE_GELU: bool = False
    ADDITION_LOSS_W: float = 0.1
    RELATION_COMBINE_DROPOUT_PROB: float = 0.2

    attribute: AttributeConfig = field(default_factory=AttributeConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    _dataset_specs: Dict[str, DatasetSpec] = field(default_factory=dict)

    def dataset_spec(self, dataset: str) -> Optional[DatasetSpec]:
        return self._dataset_specs.get(dataset)

    def topk_inputsize1_inputsize2(self, dataset: str) -> Tuple[int, int, int]:
        spec = self.dataset_spec(dataset)
        if spec:
            return spec.candidate_topk, spec.source_input_size, spec.target_input_size
        return 1000, 512, 512


runtime = RuntimeState()


def apply_settings(
    config: HybeaConfig,
    dataset: str,
    *,
    base_dir: Path,
    data_root: Path,
    results_dir: Path,
    reduction_ratio: float,
    workspace_dir: Optional[Path] = None,
) -> None:
    """Populate the runtime state using the experiment configuration."""

    runtime.MODE = config.mode
    runtime.STRUCTURAL_MODEL = config.structural_model
    runtime.MODEL = "hybea"
    runtime.DATASET = dataset
    runtime.SEED = config.pipeline_seed
    runtime.SEED_NUM = config.attribute_seed
    runtime.SIZE_AFTER_REDUCTION = reduction_ratio
    runtime.SIZE_AFTER_REDUCTION_IN_PERCENTAGE = reduction_ratio * 100

    Path(base_dir).mkdir(parents=True, exist_ok=True)
    runtime.BASE_DIR = str(base_dir)

    workspace = workspace_dir or data_root.parent
    Path(workspace).mkdir(parents=True, exist_ok=True)
    runtime.DATA_DIR = str(workspace)
    runtime.RAW_DATA_DIR = runtime.DATA_DIR
    runtime.PROCESSED_DATA_DIR = runtime.DATA_DIR
    runtime.AUGMENTED_DATA_DIR = runtime.DATA_DIR
    runtime.REDUCED_DATA_DIR = runtime.DATA_DIR

    runtime.DATA_TARGET = str(data_root)
    runtime.RESULT_DIR = str(results_dir)
    runtime.DATA_PATH = os.path.join(runtime.DATA_TARGET, "attribute_data", "")

    device = config.device or "cuda"
    if device.startswith("cuda"):
        try:
            runtime.CUDA_NUM = int(device.split(":")[1])
        except (IndexError, ValueError):
            runtime.CUDA_NUM = 0
    else:
        runtime.CUDA_NUM = 0

    attr = config.attribute
    runtime.attribute = attr
    runtime.MODEL_INPUT_DIM = attr.model_input_dim
    runtime.EPOCH_NUM = attr.epochs
    runtime.NEAREST_SAMPLE_NUM = attr.nearest_sample_num
    runtime.CANDIDATE_GENERATOR_BATCH_SIZE = attr.candidate_generator_batch_size
    runtime.NEG_NUM = attr.negatives
    runtime.MARGIN = attr.margin
    runtime.LEARNING_RATE = attr.learning_rate
    runtime.TRAIN_BATCH_SIZE = attr.train_batch_size
    runtime.TEST_BATCH_SIZE = attr.test_batch_size
    runtime.CSLS = attr.csls

    struct = config.structure
    runtime.structure = struct
    runtime.RANDOM_INITIALIZATION = struct.random_initialization
    runtime.HIDDEN_SIZE = struct.hidden_size
    runtime.NUM_HIDDEN_LAYERS = struct.num_hidden_layers
    runtime.NUM_ATTENTION_HEADS = struct.num_attention_heads
    runtime.INPUT_DROPOUT_PROB = struct.input_dropout_prob
    runtime.ATTENTION_DROPOUT_PROB = struct.attention_dropout_prob
    runtime.HIDDEN_DROPOUT_PROB = struct.hidden_dropout_prob
    runtime.RESIDUAL_DROPOUT_PROB = struct.residual_dropout_prob
    runtime.INITIALIZER_RANGE = struct.initializer_range
    runtime.INTERMEDIATE_SIZE = struct.intermediate_size
    runtime.RESIDUAL_W = struct.residual_w
    runtime.EPOCH = struct.epochs
    runtime.MIN_EPOCHS = struct.min_epochs
    runtime.LEARNING_RATE_STRUCTURE = struct.learning_rate
    runtime.BATCH_SIZE = struct.batch_size
    runtime.EVAL_BATCH_SIZE = struct.eval_batch_size
    runtime.EARLY_STOP_MAX_TIMES = struct.early_stop_max_times
    runtime.SOFT_LABEL = struct.soft_label
    runtime.EVAL_FREQ = struct.eval_freq
    runtime.START_EVAL = struct.start_eval
    runtime.SWA_PRE_NUM = struct.swa_pre_num
    runtime.DO_TRAIN = struct.enabled
    runtime.DO_TEST = struct.enabled
    runtime.USE_GELU = struct.use_gelu
    runtime.ADDITION_LOSS_W = struct.addition_loss_w
    runtime.RELATION_COMBINE_DROPOUT_PROB = struct.relation_combine_dropout_prob

    runtime._dataset_specs = config.datasets


def path_for_KG(dataset: str) -> Tuple[str, str]:
    return _NAME_FILES.get(dataset, ("kg1_names.xlsx", "kg2_names.xlsx"))


def dataset_spec(dataset: str) -> Optional[DatasetSpec]:
    return runtime.dataset_spec(dataset)


def topk_inputsize1_inputsize2(dataset: str) -> Tuple[int, int, int]:
    return runtime.topk_inputsize1_inputsize2(dataset)


def __getattr__(name: str):
    return getattr(runtime, name)
