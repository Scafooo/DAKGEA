import os
import pickle
import numpy as np
import sys

from logger import get_logger

logger = get_logger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.alignment_models.methods.hybea import legacy_config as cfg

def calc_hits(dataset, mypath, mode):

    logger.info(f"Calculating Hits@1 and Hits@10 for {dataset}, mode {mode}, from {mypath}")

    test_set = []
    with open(cfg.DATA_TARGET + "/knowformer_data/" + cfg.DATASET + "/ref_ents.txt", "r") as fp:
        for line in fp:
            e1 = line.split("\t")[0]
            e2 = line.split("\t")[1].rstrip()
            test_set.append((e1, e2))
            
    ents_1 = [e1 for e1,e2 in test_set]
    ents_2 = [e2 for e1,e2 in test_set]

    if mode == "Hybea":
        index = 0
        while os.path.exists(mypath + "/" + "res_mat_1_struct" + str(index) + ".pickle"):
                index += 1
        with open(mypath + "res_mat_1_struct" + str(index - 1) + ".pickle", "rb") as file:
            res_mat_1 = pickle.load(file)
    elif mode == "Hybea_struct_first":
        index = 0
        while os.path.exists(mypath + "/" + "res_mat_1_attr" + str(index) + ".pickle"):
                index += 1
        with open(mypath + "res_mat_1_attr" + str(index - 1) + ".pickle", "rb") as file:
            res_mat_1 = pickle.load(file).detach().cpu().numpy()
    elif mode == "Hybea_basic":
        with open(mypath + "res_mat_1_struct0.pickle", "rb") as file:
            res_mat_1 = pickle.load(file)
    elif mode == "Hybea_basic_structure_first":
        with open(mypath + "res_mat_1_attr0.pickle", "rb") as file:
            res_mat_1 = pickle.load(file).detach().cpu().numpy()
    else:
        logger.error(f"Wrong Mode!")
        exit()

    index = 0
    while os.path.exists(mypath + "/" + "rec_new_pairs_from_structure" + str(index) + ".pickle"):
        index += 1
        
    if mode == "Hybea_basic" or mode == "Hybea_basic_structure_first":
        index = 1
    
    newp_struct_list = []
    for i in range(0, index):
        with open(mypath + "rec_new_pairs_from_structure" + str(i) + ".pickle", "rb") as file:
            newp_struct = pickle.load(file)
        temp = [ pair[0] for pair in newp_struct]

        for t in temp:
            newp_struct_list.append(t)

    index = 0
    while os.path.exists(mypath + "/" + "rec_new_pairs_from_attr" + str(index) + ".pickle"):
        index += 1

    if mode == "Hybea_basic" or mode == "Hybea_basic_structure_first":
        index = 1

    newp_attr_list = []
    for i in range(0, index):
        with open(mypath + "rec_new_pairs_from_attr" + str(i) + ".pickle", "rb") as file:
            newp_attr = pickle.load(file)
        temp = [ pair[0] for pair in newp_attr]

        for t in temp:
            newp_attr_list.append(t)


    logger.info(f"New Pairs structure: {len(newp_struct_list)}")
    logger.info(f"New Pairs attribute: {len(newp_attr_list)}")
    
    c = 0
    cn = 0
    hits_1 = 0
    hits_10 = 0
    MRR = 0
    for i in range(len(ents_1)):

        if ents_1[i] in newp_struct_list or ents_1[i] in newp_attr_list:
            c += 1
            hits_1 +=1
            hits_10 += 1
            MRR += 1
            cn += 1
            continue
        
        rank = (-res_mat_1[i, :]).argsort()
        rank_index = np.where(rank == i)[0][0]
        if rank_index == 0:
            hits_1 +=1
            hits_10 += 1
        elif rank_index < 5:
            hits_10 +=1
        elif rank_index < 10:
            hits_10 += 1

        MRR += 1/ (rank_index + 1)

        if i == rank[0]:
            c += 1
    
    total = len(ents_1)
    logger.info(f"Accuracy: {c/total * 100}")

    hits_1_percentage = hits_1/total * 100
    hits_10_percentage = hits_10/total * 100
    MRR_percentage = MRR/total

    logger.info(f"Hits@1: {hits_1_percentage}")
    logger.info(f"Hits@10: {hits_10_percentage}")
    logger.info(f"MRR: {MRR_percentage}")

    logger.info(str(cn) + " out of " + str(len(set(newp_struct_list).union(set(newp_attr_list)))) + " new proposed pairs are correct")

    return hits_1_percentage, hits_10_percentage, MRR_percentage