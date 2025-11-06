# from config.config import DATASET, BASE_DIR
from src.alignment_models.methods.hybea import legacy_config as cfg

# Choose the dataset that will be used to extract the entity names
# Choose between D_W_15K_V1, D_W_15K_V2, SRPRS_D_W_15K_V1, SRPRS_D_W_15K_V2, BBC_DB
DATASET = cfg.DATASET
BASE_DIR = cfg.BASE_DIR