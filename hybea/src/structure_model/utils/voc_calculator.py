import os

from src.alignment_models.methods.hybea import legacy_config as cfg

def calculate_voc_lim(dataset = cfg.DATASET, dataset_target = cfg.DATA_TARGET, structural_model = cfg.STRUCTURAL_MODEL):
    # Read vocab
    dataset = cfg.DATASET
    dataset_target = cfg.DATA_TARGET
    structural_model = cfg.STRUCTURAL_MODEL

    vocab_list = []
    vocab_path = os.path.join(dataset_target, structural_model.lower() + "_data", dataset, "vocab.txt")
    with open(vocab_path, "r", newline='\n') as fp:
        for line in fp:
            vocab_list.append(line.split()[0])

    ent_ids_1_list = []
    ent_ids_1_path = os.path.join(dataset_target, structural_model.lower() + "_data", dataset, "ent_ids_1")
    with open(ent_ids_1_path, "r", newline='\n') as fp:
        for line in fp:
            ent_ids_1_list.append(line.strip().split("\t")[1])

    ent_ids_2_list = []
    ent_ids_2_path = os.path.join(dataset_target, structural_model.lower() + "_data", dataset, "ent_ids_2")
    with open(ent_ids_2_path, "r", newline='\n') as fp:
        for line in fp:
            ent_ids_2_list.append(line.strip().split("\t")[1])

    voc_limit_2 = None
    voc_limit_1 = None
    for i in range(len(vocab_list) - 1, -1, -1):
        if vocab_list[i] in ent_ids_2_list:
            if voc_limit_2 is None:
                voc_limit_2 = i + 1
        if vocab_list[i] in ent_ids_1_list:
            if voc_limit_1 is None:
                voc_limit_1 = i + 1

    return voc_limit_1, voc_limit_2