import os

MODEL = "hybea"

# =====================================================
# DAEA PARAMETERS
# =====================================================

# Choose between BBC_DB,
#                D_W_15K_V1,
#                D_W_15K_V2,
#                fr_en,
#                ICEW_WIKI,
#                ICEW_YAGO,
#                ja_en,
#                SRPRS_D_W_15K_V1,
#                SRPRS_D_W_15K_V2,
#                zh_en
DATASET="D_W_15K_V1"

# Choose between Knowformer,
#                RREA
STRUCTURAL_MODEL = "Knowformer"

# Choose between Hybea,
#                Hybea_struct_first,
#                Hybea_without_structure,
#                Hybea_without_factual,
#                Hybea_basic,
#                Hybea_basic_structure_first
MODE = "Hybea"

# =====================================================
# REDUCTION PARAMETERS
# =====================================================
SEED = 42
# Choose between [0.1, ...,  0.9],
# Ex. 0.5 reduction of 50% i.e. remove half of the entities
SIZE_AFTER_REDUCTION = 0.30

SIZE_AFTER_REDUCTION_IN_PERCENTAGE = SIZE_AFTER_REDUCTION * 100

# =====================================================
# DIR INFO
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Choose between RAW,
#                PROCESSED,
#                AUGMENTED,
#                REDUCED
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
AUGMENTED_DATA_DIR = os.path.join(DATA_DIR, "augmented", str(SIZE_AFTER_REDUCTION_IN_PERCENTAGE))
REDUCED_DATA_DIR = os.path.join(DATA_DIR, "reduced", str(SIZE_AFTER_REDUCTION_IN_PERCENTAGE))

# Choose between RAW_DATA_DIR,
#                PROCESSED_DATA_DIR,
#                AUGMENTED_DATA_DIR,
#                REDUCED_DATA_DIR
DATA_TARGET = AUGMENTED_DATA_DIR

RESULT_DIR = os.path.join(BASE_DIR, "results")

# =====================================================
# PARAMETERS FOR HYBEA ATTRIBUTE MODEL
# =====================================================
CUDA_NUM = 0 # used GPU num
MODEL_INPUT_DIM  = 768

SEED_NUM = 11037

EPOCH_NUM = 200 #training epoch num

NEAREST_SAMPLE_NUM = 128
CANDIDATE_GENERATOR_BATCH_SIZE = 128

NEG_NUM = 2 # negative sample num
MARGIN = 3 # margin
LEARNING_RATE = 1e-5 # learning rate
TRAIN_BATCH_SIZE = 24
TEST_BATCH_SIZE = 128

FOLD = "2"

# Choose between D_W_15K_V1, D_W_15K_V2, SRPRS_D_W_15K_V1, SRPRS_D_W_15K_V2, BBC_DB
DATASET = DATASET

CSLS = 2

def topk_inputsize1_inputsize2(dataset):
    if dataset == "D_W_15K_V1":
        return 1000,  341, 649
    elif dataset == "BBC_DB":
        return 939, 4, 723
    elif dataset == "D_W_15K_V2":
        return 1000, 175, 457
    elif dataset == "SRPRS_D_W_15K_V1":
        return 1000, 363, 652
    elif dataset == "SRPRS_D_W_15K_V2":
        return 1000, 256, 531
    elif dataset == "fr_en":
        return 1000, 4544, 6420
    elif dataset == "ja_en":
        return 1000, 5878, 6063
    elif dataset == "zh_en":
        return 1000, 8183, 7170
    elif dataset == "ICEW_WIKI":
        return 507, 1, 1
    elif dataset == "ICEW_YAGO":
        return 300, 1, 1

DATA_PATH = DATA_TARGET + "/attribute_data/" + DATASET + "/"


# =====================================================
# PARAMETERS FOR HYBEA STRUCTURE MODEL
# =====================================================
def path_for_KG(dataset):
    if dataset == "D_W_15K_V1":
        return 'DBpedia_names.xlsx', 'Wikidata_names.xlsx'
    elif dataset == "BBC_DB":
        return 'BBC_names.xlsx', 'DBpedia_names.xlsx'
    elif dataset == "D_W_15K_V2":
        return 'DBpedia_names.xlsx', 'Wikidata_names.xlsx'
    elif dataset == "SRPRS_D_W_15K_V1":
        return 'DBpedia_names.xlsx', 'Wikidata_names.xlsx'
    elif dataset == "SRPRS_D_W_15K_V2":
        return 'DBpedia_names.xlsx', 'Wikidata_names.xlsx'
    elif dataset == "fr_en":
        return 'fr_names.xlsx', 'en_names.xlsx'
    elif dataset == "ja_en":
        return 'ja_names.xlsx', 'en_names.xlsx'
    elif dataset == "zh_en":
        return 'zh_names.xlsx', 'en_names.xlsx'
    elif dataset == "ICEW_WIKI":
        return 'icew_names.xlsx', 'wiki_names.xlsx'
    elif dataset == "ICEW_YAGO":
        return 'icew_names.xlsx', 'yago_names.xlsx'
    return None, None

TASK = "entity-alignment"

# def random_initialization(default):
#     if default:
#         return 256
#     else:
#         return 768

RANDOM_INITIALIZATION = False
if RANDOM_INITIALIZATION:
    HIDDEN_SIZE = 256
else:
    HIDDEN_SIZE = 768

NUM_HIDDEN_LAYERS = 12
NUM_ATTENTION_HEADS = 4
INPUT_DROPOUT_PROB = 0.5
ATTENTION_DROPOUT_PROB = 0.1
HIDDEN_DROPOUT_PROB = 0.3
RESIDUAL_DROPOUT_PROB = 0.1
INITIALIZER_RANGE = 0.02
INTERMEDIATE_SIZE = 2048
RESIDUAL_W = 0.5
EPOCH = 200
MIN_EPOCHS = 10
LEARNING_RATE = 5e-4
BATCH_SIZE = 2048
EVAL_BATCH_SIZE = 4096
EARLY_STOP_MAX_TIMES = 3
SOFT_LABEL = 0.25
EVAL_FREQ = 5
START_EVAL = 0
SWA_PRE_NUM = 5
DO_TRAIN = True
DO_TEST = True
USE_GELU = False
ADDITION_LOSS_W = 0.1
RELATION_COMBINE_DROPOUT_PROB = 0.2
CSLS = 2