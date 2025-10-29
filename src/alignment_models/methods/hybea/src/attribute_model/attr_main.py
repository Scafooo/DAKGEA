import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import torch.optim as optim
import random
import pickle

from .utils.Read_data_func import read_data
from .model.Basic_Bert_Unit_model import Basic_Bert_Unit_model
from .utils.Batch_TrainData_Generator import Batch_TrainData_Generator
from .train.train_func import train
import numpy as np
from sentence_transformers import SentenceTransformer
from src.alignment_models.methods.hybea import runtime as cfg

log = cfg.logger

if torch.cuda.is_available() and cfg.CUDA_NUM >= 0:
    torch.cuda.set_device(cfg.CUDA_NUM)
    log.info("Using CUDA device %s (%s)", torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device()))

def fixed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def run(new_pairs):

    #read data
    log.info("Loading attribute-model data from %s", cfg.DATA_PATH)
    ent_ill, train_ill, test_ill, valid_ill, \
    index2rel, index2entity, rel2index, entity2index, \
    ent2data, rel_triples_1, rel_triples_2 = read_data(cfg.DATA_PATH)
    
    # ent2data_types = ent2data[1]
    # entid2data = ent2data[0]
    # result = set()
    # for lst in ent2data_types.values():
    #     for x in lst:
    #         result.add(x)
    # print(len(result))

    # ent2data_types = ent2data[3]
    # entid2data = ent2data[2]
    # result = set()
    # for lst in ent2data_types.values():
    #     for x in lst:
    #         result.add(x)
    # print(len(result))
    # exit()
    
    log.info("Loaded %d ILLs, %d relations, %d entities", len(ent_ill), len(index2rel), len(index2entity))
    log.info("Triple counts: KG1=%d KG2=%d", len(rel_triples_1), len(rel_triples_2))


    #get train/test ILLs from file.
    log.info("Train/valid/test splits: train=%d, test=%d, valid=%d", len(train_ill), len(test_ill), len(valid_ill))
    log.debug("Train ∪ Test ∪ Valid size: %d", len(set(train_ill) | set(test_ill) | set(valid_ill)))
    log.debug("Train ∩ Test ∩ Valid size: %d", len(set(train_ill) & set(test_ill) & set(valid_ill)))

    if cfg.TRAIN_RATIO is not None:
        train_ratio = cfg.TRAIN_RATIO
        valid_ratio = cfg.VALID_RATIO or 0.0
        if not 0 < train_ratio < 1:
            raise ValueError(f"HybEA attribute train_ratio must be between 0 and 1 (got {train_ratio})")
        if valid_ratio < 0 or train_ratio + valid_ratio >= 1:
            raise ValueError(
                f"HybEA attribute split ratios must satisfy train_ratio + valid_ratio < 1 "
                f"(got train={train_ratio}, valid={valid_ratio})"
            )
        rng = random.Random(cfg.SEED_NUM)
        unique_pairs = list(dict.fromkeys(ent_ill))
        rng.shuffle(unique_pairs)
        total_pairs = len(unique_pairs)

        train_count = max(1, int(total_pairs * train_ratio))
        valid_count = max(0, int(total_pairs * valid_ratio))
        remaining = total_pairs - train_count - valid_count
        if remaining <= 0:
            remaining = 1
            if train_count > valid_count:
                train_count = max(1, train_count - 1)
            elif valid_count > 0:
                valid_count -= 1
        test_count = remaining

        new_train = unique_pairs[:train_count]
        new_valid = unique_pairs[train_count : train_count + valid_count]
        new_test = unique_pairs[train_count + valid_count :]

        train_ill = list(new_train)
        valid_ill = list(new_valid)
        test_ill = list(new_test)
        ent_ill = train_ill + test_ill + valid_ill

        log.info(
            "Resampled ILL splits (train=%.2f, valid=%.2f, test=%.2f) -> train=%d, test=%d, valid=%d",
            train_ratio,
            valid_ratio,
            1 - train_ratio - valid_ratio,
            len(train_ill),
            len(test_ill),
            len(valid_ill),
        )
        log.debug("Resampled union size: %d", len(set(train_ill) | set(test_ill) | set(valid_ill)))
        log.debug("Resampled intersections: train∩test=%d train∩valid=%d test∩valid=%d",
                  len(set(train_ill) & set(test_ill)),
                  len(set(train_ill) & set(valid_ill)),
                  len(set(test_ill) & set(valid_ill)))

    log.info("Incorporating %d new pairs", len(new_pairs))
    train_inter = set(train_ill).intersection(new_pairs)
    test_inter = set(test_ill).intersection(new_pairs)
    valid_inter = set(valid_ill).intersection(new_pairs)
    log.debug("Overlap with new pairs - train:%d test:%d valid:%d", len(train_inter), len(test_inter), len(valid_inter))
    
    for pair in new_pairs:
        train_ill.append(pair)
        
    log.info("Updated train/test/valid sizes: train=%d, test=%d, valid=%d", len(train_ill), len(test_ill), len(valid_ill))

    sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    attr_dict = {}
    for i in range(0, 3, 2):
        entid2data = ent2data[i]
        for eid in entid2data:
            attr_dict[eid] = [] 
            for x in entid2data[eid]:
                attr_dict[eid].append(x)
    sentences = [value for value_list in attr_dict.values() for value in value_list]
    
    log.info("Generating sentence-transformer embeddings for %d sentences", len(sentences))
    sent_emb = sbert_model.encode(sentences)
    log.debug("Sentence embeddings computed")
    
    emb_dict = {}
    for index, sentence in enumerate(sentences):
        emb_dict[sentence] = sent_emb[index]
    
    Model = Basic_Bert_Unit_model(cfg.topk_inputsize1_inputsize2(cfg.DATASET)[1], cfg.topk_inputsize1_inputsize2(cfg.DATASET)[2], 768, emb_dict)

    Model.cuda(cfg.CUDA_NUM)

    # Criterion = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    Criterion = nn.MarginRankingLoss(margin=cfg.MARGIN, reduction="mean")
    Optimizer = AdamW(Model.parameters(),lr=cfg.LEARNING_RATE)

    ent1 = [e1 for e1,e2 in ent_ill]
    ent2 = [e2 for e1,e2 in ent_ill]

    #training data generator(can generate batch-size training data)
    Train_gene = Batch_TrainData_Generator(train_ill, ent1, ent2,index2entity,batch_size=cfg.TRAIN_BATCH_SIZE,neg_num=cfg.NEG_NUM)

    ent_ill, res_mat, res_mat_2, loss_list, hits_1_list = train(Model,Criterion,Optimizer,Train_gene,train_ill,test_ill, valid_ill, ent2data)
    
    return ent_ill, res_mat, res_mat_2, loss_list, hits_1_list

def run_attr_model(new_pairs):
    fixed(cfg.SEED_NUM)
    return run(new_pairs)
