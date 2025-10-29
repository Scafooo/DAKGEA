import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pickle
from ..valid.eval_function import cos_sim_mat_generate,batch_topk,hit_res
from src.alignment_models.methods.hybea import runtime as cfg

log = cfg.logger

EARLY_STOP_EVAL_FREQ = 5
EARLY_STOP_PATIENCE = 3


def entlist2emb(Model,entids,entid2data,cuda_num, which):
    """
    return basic bert unit output embedding of entities
    """
    batch_sentences = []
    batch_attr_types = []
    
    if which == "1":
        ent2data_types = entid2data[1]
        entid2data = entid2data[0]
        # 341
        input_size = cfg.topk_inputsize1_inputsize2(cfg.DATASET)[1]
        # result = set()
        # for lst in ent2data_types.values():
        #     for x in lst:
        #         result.add(x)
        # print(len(result))
        # exit()
    elif which == "2":
        ent2data_types = entid2data[3]
        entid2data = entid2data[2]
        # 649
        input_size = cfg.topk_inputsize1_inputsize2(cfg.DATASET)[2]
        # result = set()
        # for lst in ent2data_types.values():
        #     for x in lst:
        #         result.add(x)
        # print(len(result))
        # exit()
    for eid in entids:
        if eid in entid2data:
            for x in entid2data[eid]:
                batch_sentences.append(x)
            temp_attr_types = ent2data_types[eid]
        else:
            temp_attr_types = []

        batch_attr_types.append(temp_attr_types)
        
    max_len = max(len(lst) for lst in batch_attr_types)
    alpha_i_list = [torch.tensor(alpha_i) for alpha_i in batch_attr_types]
    padded_alpha_i = []
    for x in alpha_i_list:
        padding_values_to_add = torch.full((max_len - len(x),), input_size, dtype=torch.float32)
        padded_alpha_i.append(np.array(torch.cat((x, padding_values_to_add))))
    padded_alpha_i = torch.tensor(np.array(padded_alpha_i)).int().cuda(cfg.CUDA_NUM)

    batch_emb = Model(batch_attr_types, batch_sentences, padded_alpha_i, which)
    del batch_sentences
    del padded_alpha_i
    del alpha_i_list
    del batch_attr_types
    return batch_emb


def generate_candidate_dict(Model,train_ent1s,train_ent2s,for_candidate_ent1s,for_candidate_ent2s,
                                entid2data,index2entity,
                                nearest_sample_num = cfg.NEAREST_SAMPLE_NUM, batch_size = cfg.CANDIDATE_GENERATOR_BATCH_SIZE):

    start_time = time.time()
    Model.eval()
    torch.cuda.empty_cache()
    candidate_dict = dict()
    phase_times = {}
    with torch.no_grad():
        log.debug("Generating candidates: phase 0/3")
        #langauge1 (KG1)
        train_emb1 = []
        for_candidate_emb1 = []
        phase_start = time.time()

        for i in range(0,len(train_ent1s),batch_size):
            temp_emb = entlist2emb(Model,train_ent1s[i:i+batch_size],entid2data, cfg.CUDA_NUM, "1").cuda(cfg.CUDA_NUM).tolist()
            train_emb1.extend(temp_emb)
        for i in range(0,len(for_candidate_ent2s),batch_size):
            temp_emb = entlist2emb(Model,for_candidate_ent2s[i:i+batch_size],entid2data, cfg.CUDA_NUM, "2").cuda(cfg.CUDA_NUM).tolist()
            for_candidate_emb1.extend(temp_emb)
        phase_times["kg1_embeddings"] = time.time() - phase_start
        log.debug("Generating candidates: phase 1/3")
        
        #language2 (KG2)
        train_emb2 = []
        for_candidate_emb2 = []
        phase_start = time.time()
        for i in range(0,len(train_ent2s),batch_size):
            temp_emb = entlist2emb(Model,train_ent2s[i:i+batch_size],entid2data,cfg.CUDA_NUM, "2").cuda(cfg.CUDA_NUM).tolist()
            train_emb2.extend(temp_emb)
        for i in range(0,len(for_candidate_ent1s),batch_size):
            temp_emb = entlist2emb(Model,for_candidate_ent1s[i:i+batch_size],entid2data,cfg.CUDA_NUM, "1").cuda(cfg.CUDA_NUM).tolist()
            for_candidate_emb2.extend(temp_emb)
        torch.cuda.empty_cache()
        phase_times["kg2_embeddings"] = time.time() - phase_start
        log.debug("Generating candidates: phase 2/3")

        #cos sim
        phase_start = time.time()
        cos_sim_mat1 = cos_sim_mat_generate(train_emb1,for_candidate_emb1)
        cos_sim_mat2 = cos_sim_mat_generate(train_emb2,for_candidate_emb2)
        torch.cuda.empty_cache()
        phase_times["similarity"] = time.time() - phase_start

        #topk index
        phase_start = time.time()
        _,topk_index_1 = batch_topk(cos_sim_mat1,topn=nearest_sample_num,largest=True)
        topk_index_1 = topk_index_1.tolist()

        _,topk_index_2 = batch_topk(cos_sim_mat2,topn=nearest_sample_num,largest=True)
        topk_index_2 = topk_index_2.tolist()
        phase_times["topk"] = time.time() - phase_start

        #get candidate
        for x in range(len(topk_index_1)):
            e = train_ent1s[x]
            candidate_dict[e] = []
            for y in topk_index_1[x]:
                c = for_candidate_ent2s[y]
                candidate_dict[e].append(c)

        for x in range(len(topk_index_2)):
            e = train_ent2s[x]
            candidate_dict[e] = []
            for y in topk_index_2[x]:
                c = for_candidate_ent1s[y]
                candidate_dict[e].append(c)
        #show
        # def rstr(string):
        #     return string.split(r'/resource/')[-1]
        # for e in train_ent1s[100:105]:
        #     print(rstr(index2entity[e]),"---",[rstr(index2entity[eid]) for eid in candidate_dict[e][:6]])
        # for e in train_ent2s[100:105]:
        #     print(rstr(index2entity[e]),"---",[rstr(index2entity[eid]) for eid in candidate_dict[e][:6]])
    duration = time.time() - start_time
    log.debug(
        "Candidate generation completed in %.3f seconds (kg1=%.2fs, kg2=%.2fs, similarity=%.2fs, topk=%.2fs)",
        duration,
        phase_times.get("kg1_embeddings", 0.0),
        phase_times.get("kg2_embeddings", 0.0),
        phase_times.get("similarity", 0.0),
        phase_times.get("topk", 0.0),
    )
    torch.cuda.empty_cache()
    return candidate_dict, duration

def train(Model,Criterion,Optimizer,Train_gene,train_ill,test_ill,valid_ill,entid2data):
    train_size = len(train_ill)
    test_size = len(test_ill)
    valid_size = len(valid_ill)
    log.info(
        "Starting attribute-model training (train=%d, valid=%d, test=%d, batch=%d, negatives=%d, patience=%d)",
        train_size,
        valid_size,
        test_size,
        cfg.TRAIN_BATCH_SIZE,
        cfg.NEG_NUM,
        EARLY_STOP_PATIENCE,
    )
    log.info(
        "Validation frequency every %d epochs; using margin=%.3f, learning_rate=%s",
        EARLY_STOP_EVAL_FREQ,
        cfg.MARGIN,
        cfg.LEARNING_RATE,
    )
    loss_list = []
    hits_1_list = []
    max_hits1 = 0.0
    wait = 0
    total_candidate_time = 0.0
    total_train_time = 0.0
    eval_counter = 0
    epochs_completed = 0
    for epoch in range(cfg.EPOCH_NUM):
        epoch_start = time.time()
        log.info("Epoch %d/%d", epoch + 1, cfg.EPOCH_NUM)
        #generate candidate_dict
        #(candidate_dict is used to generate negative example for train_ILL)
        train_ent1s = [e1 for e1,e2 in train_ill]
        train_ent2s = [e2 for e1,e2 in train_ill]
        for_candidate_ent1s = Train_gene.ent_ids1
        for_candidate_ent2s = Train_gene.ent_ids2
        log.debug(
            "Training entities: ent1=%d ent2=%d candidate1=%d candidate2=%d",
            len(train_ent1s),
            len(train_ent2s),
            len(for_candidate_ent1s),
            len(for_candidate_ent2s),
        )
        candidate_start = time.time()
        candidate_dict, candidate_time = generate_candidate_dict(
            Model,
            train_ent1s,
            train_ent2s,
            for_candidate_ent1s,
            for_candidate_ent2s,
            entid2data,
            Train_gene.index2entity,
        )
        total_candidate_time += candidate_time

        Train_gene.train_index_gene(candidate_dict) #generate training data with candidate_dict
        #train
        epoch_loss,epoch_train_time = ent_align_train(Model,Criterion,Optimizer,Train_gene,entid2data)
        loss_list.append(epoch_loss)
        total_train_time += epoch_train_time
        Optimizer.zero_grad()
        torch.cuda.empty_cache()
        epoch_elapsed = time.time() - epoch_start
        log.info(
            "Epoch %d summary: loss=%.3f | train_time=%.1fs | candidate_time=%.1fs | total=%.1fs",
            epoch + 1,
            epoch_loss,
            epoch_train_time,
            candidate_time,
            epoch_elapsed,
        )
        epochs_completed = epoch + 1
        if epoch % EARLY_STOP_EVAL_FREQ == 0:
            _, _, hits_1 = test(Model, valid_ill, entid2data, cfg.TEST_BATCH_SIZE, context="EVAL IN VALID SET:", second_mat = False, csls=cfg.CSLS)
            hits_1_list.append(hits_1)
            eval_counter += 1

            if hits_1 > max_hits1:
                max_hits1 = hits_1
                wait = 0
                log.info(
                    "Validation hits@1 improved to %.2f%% (eval #%d)",
                    hits_1,
                    eval_counter,
                )
            else:
                wait += 1
                log.info(
                    "Validation hits@1 %.2f%% (best %.2f%%, patience %d/%d)",
                    hits_1,
                    max_hits1,
                    wait,
                    EARLY_STOP_PATIENCE,
                )
            if wait >= EARLY_STOP_PATIENCE and epochs_completed >= EARLY_STOP_EVAL_FREQ:
                log.info("Early stopping triggered at epoch %d (no improvement for %d evaluations)", epochs_completed, wait)
                break
            
    _, _, _, = test(Model, test_ill, entid2data, cfg.TEST_BATCH_SIZE, context="EVAL IN TEST SET WITHOUT CSLS:", second_mat = False, csls=0)
    ent_ill, res_mat, res_mat_2, hits_1 = test(Model, test_ill, entid2data, cfg.TEST_BATCH_SIZE, context="EVAL IN TEST SET WITH CSLS:", second_mat = True, csls=cfg.CSLS)
    log.info(
        "Training finished: best_hits@1=%.2f%%, epochs_run=%d, total_train_time=%.1fs, total_candidate_time=%.1fs",
        max_hits1,
        epochs_completed,
        total_train_time,
        total_candidate_time,
    )
    return ent_ill, res_mat, res_mat_2, loss_list, hits_1_list


def test(Model,ent_ill,entid2data,batch_size,context = "", second_mat=False, csls=cfg.CSLS):
    log.debug("----- test start -----")
    start_time = time.time()
    if context:
        log.info("%s", context)
    Model.eval()
    with torch.no_grad():
        ents_1 = [e1 for e1,e2 in ent_ill]
        ents_2 = [e2 for e1,e2 in ent_ill]

        emb1 = []
        for i in range(0,len(ents_1),batch_size):
            batch_ents_1 = ents_1[i: i+batch_size]
            batch_emb_1 = entlist2emb(Model,batch_ents_1,entid2data,cfg.CUDA_NUM, "1").detach().cuda(cfg.CUDA_NUM).tolist()
            emb1.extend(batch_emb_1)
            del batch_emb_1

        emb2 = []
        for i in range(0,len(ents_2),batch_size):
            batch_ents_2 = ents_2[i: i+batch_size]
            batch_emb_2 = entlist2emb(Model,batch_ents_2,entid2data,cfg.CUDA_NUM, "2").detach().cuda(cfg.CUDA_NUM).tolist()
            emb2.extend(batch_emb_2)
            del batch_emb_2

        log.debug("Computing cosine similarity for %d pairs", len(ents_1))
        res_mat = cos_sim_mat_generate(emb1,emb2,batch_size,cuda_num=cfg.CUDA_NUM, csls=csls)
        score,top_index = batch_topk(res_mat,batch_size,topn = cfg.topk_inputsize1_inputsize2(cfg.DATASET)[0],largest=True,cuda_num=cfg.CUDA_NUM)
        hits_1 = hit_res(top_index)
        if second_mat:
            res_mat_2 = cos_sim_mat_generate(emb2,emb1,batch_size,cuda_num=cfg.CUDA_NUM, csls=csls)
            log.debug("Test stage completed in %.3f seconds", time.time()-start_time)
            return ent_ill, res_mat, res_mat_2, hits_1
        
    log.debug("Test stage completed in %.3f seconds", time.time()-start_time)
    return ent_ill, res_mat, hits_1

def ent_align_train(Model,Criterion,Optimizer,Train_gene,entid2data):
    start_time = time.time()
    all_loss = 0
    Model.train()
    for pe1s, pe2s, ne1s, ne2s in Train_gene:
        Optimizer.zero_grad()
        pos_emb1 = entlist2emb(Model,pe1s,entid2data,cfg.CUDA_NUM, "1")
        pos_emb2 = entlist2emb(Model,pe2s,entid2data,cfg.CUDA_NUM, "2")
        batch_length = pos_emb1.shape[0]
        pos_score = F.pairwise_distance(pos_emb1,pos_emb2,p=2,keepdim=True)
        del pos_emb1
        del pos_emb2

        neg_emb1 = entlist2emb(Model,ne1s,entid2data,cfg.CUDA_NUM, "1")
        neg_emb2 = entlist2emb(Model,ne2s,entid2data,cfg.CUDA_NUM, "2")
        neg_score = F.pairwise_distance(neg_emb1,neg_emb2,p=2,keepdim=True)
        del neg_emb1
        del neg_emb2

        sum1 = torch.sum(0.2 * pos_score)
        sum2 = torch.sum(0.8 * torch.clamp(3.0 - neg_score, min=0.0))


        batch_loss = sum1 + sum2
    
        batch_loss.backward()
        Optimizer.step()

        all_loss += batch_loss.item()

    all_using_time = time.time() - start_time
    return np.mean(all_loss), all_using_time

# def ent_align_train(Model,Criterion,Optimizer,Train_gene,entid2data):
#     start_time = time.time()
#     all_loss = 0
#     Model.train()
#     for pe1s,pe2s,ne1s,ne2s in Train_gene:
#         Optimizer.zero_grad()
#         pos_emb1 = entlist2emb(Model,pe1s,entid2data,CUDA_NUM, "1")
#         pos_emb2 = entlist2emb(Model,pe2s,entid2data,CUDA_NUM, "2")
#         batch_length = pos_emb1.shape[0]
#         pos_score = F.pairwise_distance(pos_emb1,pos_emb2,p=1,keepdim=True)#L1 distance
#         del pos_emb1
#         del pos_emb2

#         neg_emb1 = entlist2emb(Model,ne1s,entid2data,CUDA_NUM, "1")
#         neg_emb2 = entlist2emb(Model,ne2s,entid2data,CUDA_NUM, "2")
#         neg_score = F.pairwise_distance(neg_emb1,neg_emb2,p=1,keepdim=True)
#         del neg_emb1
#         del neg_emb2

#         label_y = -torch.ones(pos_score.shape).cuda(CUDA_NUM) #pos_score < neg_score
#         batch_loss = Criterion( pos_score , neg_score , label_y )
#         del pos_score
#         del neg_score
#         del label_y
#         batch_loss.backward()
#         Optimizer.step()

#         all_loss += batch_loss.item() * batch_length
#     all_using_time = time.time()-start_time
#     return all_loss,all_using_time
