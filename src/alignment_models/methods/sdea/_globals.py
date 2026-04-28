"""Mutable module-level globals shared across SDEA internals.

The adapter (model.py) populates these before running training.
"""

import os
import sys
import random
from types import SimpleNamespace

import numpy as np
import torch as t
from transformers import BertConfig

# ── Fixed hyper-parameters ────────────────────────────────────────────────
seq_max_len = 128
bert_output_dim = 300
PARALLEL = False          # Disable DataParallel; adapter uses single device
MARGIN = 1
SCORE_DISTANCE_LEVEL = 2
functionality_control = False
functionality_threshold = 0.9

# ── Mutable objects set by the adapter before each training run ───────────
args = SimpleNamespace(
    pretrain_bert_path="bert-base-multilingual-cased",
    relation=True,
    blocking=False,
    functionality=False,
)

dataset1 = None
dataset2 = None
links = None
