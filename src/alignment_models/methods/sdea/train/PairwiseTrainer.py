import numpy as np
import torch as t
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import BertTokenizer

from .._globals import args, seq_max_len, PARALLEL, SCORE_DISTANCE_LEVEL, MARGIN, links
from ..preprocess.KBStore import KBStore
from ..tools.Announce import Announce
from ..tools.ModelTools import ModelTools
from ..tools.MultiprocessingTool import MPTool, MultiprocessingTool
from ..tools.TrainingTools import TrainingTools
from .BasicBertModel import BasicBertModel
from .PairwiseDataset import PairwiseDataset
from .train_utils import cos_sim_mat_generate, batch_topk, hits_open
import torch.nn.functional as F

VALID = True


class PairwiseTrainer:
    def __init__(self):
        self.get_emb_batch = 512
        self.train_emb_batch = 8
        self.nearest_sample_num = 128
        self.neg_num = 2
        self.score_distance_level = SCORE_DISTANCE_LEVEL

    def data_prepare(self, eid2tids1: dict, eid2tids2: dict, fs1: KBStore, fs2: KBStore):
        self.fs1 = fs1
        self.fs2 = fs2
        from .._globals import links as _links, args as _args
        tokenizer = BertTokenizer.from_pretrained(_args.pretrain_bert_path)
        cls_token, sep_token = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])

        eid2data1 = [self.tids_solver(item, cls_token=cls_token, sep_token=sep_token, freqs=None)
                     for item in eid2tids1.items()]
        self.eid2data1 = {key: value for key, value in eid2data1}

        eid2data2 = [self.tids_solver(item, cls_token=cls_token, sep_token=sep_token, freqs=None)
                     for item in eid2tids2.items()]
        self.eid2data2 = {key: value for key, value in eid2data2}

        print(Announce.printMessage(), 'eid2data1 len:', len(self.eid2data1))
        print(Announce.printMessage(), 'eid2data2 len:', len(self.eid2data2))

        self.train_links = self.load_links(_links.train, fs1.entity_ids, fs2.entity_ids)
        self.valid_links = self.load_links(_links.valid, fs1.entity_ids, fs2.entity_ids)
        self.test_links = self.load_links(_links.test, fs1.entity_ids, fs2.entity_ids)
        self.train_links = list(set(self.train_links))
        self.valid_links = list(set(self.valid_links))
        self.test_links = list(set(self.test_links))

        self.train_links_p = [(e1, e2) for e1, e2 in self.train_links if e1 in self.eid2data1 and e2 in self.eid2data2]
        self.valid_links_p = [(e1, e2) for e1, e2 in self.valid_links if e1 in self.eid2data1 and e2 in self.eid2data2]
        self.test_links_p = [(e1, e2) for e1, e2 in self.test_links if e1 in self.eid2data1 and e2 in self.eid2data2]

        self.train_ent1s = list({e1 for e1, e2 in self.train_links})
        self.train_ent2s = list({e2 for e1, e2 in self.train_links})
        self.valid_ent1s = [e1 for e1, e2 in self.valid_links]
        self.valid_ent2s = [e2 for e1, e2 in self.valid_links]
        self.test_ent1s = [e1 for e1, e2 in self.test_links]
        self.test_ent2s = [e2 for e1, e2 in self.test_links]

        self.train_ent1s_p = list({e1 for e1, e2 in self.train_links_p})
        self.train_ent2s_p = list({e2 for e1, e2 in self.train_links_p})
        self.valid_ent1s_p = [e1 for e1, e2 in self.valid_links_p]
        self.valid_ent2s_p = [e2 for e1, e2 in self.valid_links_p]
        self.test_ent1s_p = [e1 for e1, e2 in self.test_links_p]
        self.test_ent2s_p = [e2 for e1, e2 in self.test_links_p]
        self.all_ent1s_p = list(self.eid2data1.keys())
        self.all_ent2s_p = list(self.eid2data2.keys())
        self.all_ent1s = list(fs1.entity_ids.values())
        self.all_ent2s = list(fs2.entity_ids.values())
        self.all_ent1s_p_idx = [self.all_ent1s.index(ent) for ent in self.all_ent1s_p]
        self.all_ent2s_p_idx = [self.all_ent2s.index(ent) for ent in self.all_ent2s_p]

        # Precompute correct column indices for open evaluation (rank against all KG2 entities)
        ent2s_p_set = {e: i for i, e in enumerate(self.all_ent2s_p)}
        self.valid_correct_idx2s = [ent2s_p_set[e2] for e2 in self.valid_ent2s_p]
        self.test_correct_idx2s = [ent2s_p_set[e2] for e2 in self.test_ent2s_p]

        self.block_loader1, self.block_loader2 = self.links_pair_loader(self.all_ent1s_p, self.all_ent2s_p)
        if VALID:
            self.valid_link_loader1 = self.links_pair_loader(self.valid_ent1s_p, self.valid_ent2s_p)[0]
        self.test_link_loader1 = self.links_pair_loader(self.test_ent1s_p, self.test_ent2s_p)[0]

    def links_pair_loader(self, ent1s, ent2s):
        inputs1 = self.get_tensor_data(ent1s, self.eid2data1)
        inputs2 = self.get_tensor_data(ent2s, self.eid2data2)
        ds1 = TensorDataset(*inputs1)
        ds2 = TensorDataset(*inputs2)
        loader1 = DataLoader(ds1, sampler=SequentialSampler(ds1), batch_size=self.get_emb_batch)
        loader2 = DataLoader(ds2, sampler=SequentialSampler(ds2), batch_size=self.get_emb_batch)
        return loader1, loader2

    @staticmethod
    def get_tensor_data(ents: list, eid2data: dict):
        inputs = [eid2data.get(key) for key in ents]
        input_ids = t.stack([ids for ids, mask in inputs], dim=0)
        masks = t.stack([mask for ids, mask in inputs], dim=0)
        return input_ids, masks

    def train(self, epochs=100, device='cpu'):
        from .._globals import links as _links
        bert_model = BasicBertModel(args.pretrain_bert_path)
        if PARALLEL and t.cuda.device_count() > 1:
            bert_model = t.nn.DataParallel(bert_model)
        bert_model.to(device)
        criterion = t.nn.MarginRankingLoss(MARGIN)
        optimizer = t.optim.AdamW(bert_model.parameters(), lr=1e-5)

        self.get_hits(bert_model, self.test_link_loader1, self.test_correct_idx2s, device=device)

        if VALID:
            model_tool = ModelTools(3, 'max')

        for epoch in range(1, epochs + 1):
            print(Announce.doing(), 'Epoch', epoch, '/', epochs, 'start')
            train_tups = self.generate_train_tups(
                bert_model, self.train_ent1s_p, self.train_ent2s_p, self.train_links_p, device
            )
            bert_model.train()
            train_ds = PairwiseDataset(train_tups, self.eid2data1, self.eid2data2)
            train_loader = DataLoader(train_ds, sampler=SequentialSampler(train_ds), batch_size=self.train_emb_batch)
            tt = TrainingTools(train_loader, device=device)
            for i, batch in tt.batches(lambda batch: len(batch[0][0])):
                optimizer.zero_grad()
                loss = self.train_batch(bert_model, tt, batch, criterion, device)
                loss.backward()
                optimizer.step()
            print()

            if VALID:
                bert_model.eval()
                with t.no_grad():
                    hit_values, mrr = self.get_hits(
                        bert_model, self.valid_link_loader1, self.valid_correct_idx2s, device=device
                    )
                stop = model_tool.early_stopping(bert_model, _links.model_save, hit_values[0])
            else:
                ModelTools.save_model(bert_model, _links.model_save)

            print(Announce.printMessage(), 'test phase')
            self.get_hits(bert_model, self.test_link_loader1, self.test_correct_idx2s, device=device)
            print(Announce.done(), 'Epoch', epoch, '/', epochs, 'end')

            if VALID and epoch > 5 and stop:
                print(Announce.done(), 'early stopping')
                break

        # Final test evaluation
        final_hit_values, final_mrr = self.get_hits(
            bert_model, self.test_link_loader1, self.test_correct_idx2s, device=device
        )
        return {
            "hits@1": final_hit_values[0] if len(final_hit_values) > 0 else 0.0,
            "hits@5": final_hit_values[1] if len(final_hit_values) > 1 else 0.0,
            "hits@10": final_hit_values[2] if len(final_hit_values) > 2 else 0.0,
            "mrr": final_mrr,
            "model_path": _links.model_save,
        }

    def get_hits(self, bert_model, source_loader1, correct_indices, device):
        """Open evaluation: rank each source entity against ALL KG2 entities."""
        src_emb1s = self.get_emb_valid(source_loader1, bert_model, device=device)
        all_emb2s = self.get_emb_valid(self.block_loader2, bert_model, device=device)
        cos_sim_mat = cos_sim_mat_generate(src_emb1s, all_emb2s, device=device)
        _, topk_idx = batch_topk(cos_sim_mat, topn=self.nearest_sample_num, largest=True)
        return hits_open(topk_idx, correct_indices)

    def train_batch(self, bert_model, tt, batch, criterion, device):
        pos_emb1 = bert_model(batch[0][0].to(device), batch[0][1].to(device))
        pos_emb2 = bert_model(batch[1][0].to(device), batch[1][1].to(device))
        batch_size = pos_emb1.shape[0]
        pos_score = F.pairwise_distance(pos_emb1, pos_emb2, p=self.score_distance_level, keepdim=True)
        y_pred1 = t.cosine_similarity(pos_emb1, pos_emb2).reshape(batch_size, 1)
        y1_0 = t.ones([batch_size, 1]).to(device) - y_pred1
        y_pred1 = t.cat([y1_0, y_pred1], dim=1)
        neg_emb1 = bert_model(batch[2][0].to(device), batch[2][1].to(device))
        neg_emb2 = bert_model(batch[3][0].to(device), batch[3][1].to(device))
        neg_score = F.pairwise_distance(neg_emb1, neg_emb2, p=self.score_distance_level, keepdim=True)
        y_pred2 = t.cosine_similarity(neg_emb1, neg_emb2).reshape(batch_size, 1)
        y2_0 = t.ones([batch_size, 1]).to(device) - y_pred2
        y_pred2 = t.cat([y2_0, y_pred2], dim=1)
        y = -t.ones(pos_score.shape).to(device)
        loss = criterion(pos_score, neg_score, y)
        if PARALLEL and t.cuda.device_count() > 1:
            loss = loss.mean()
        print(float(loss), end='')
        labels = t.cat([t.ones([batch_size], dtype=t.long), t.zeros([batch_size], dtype=t.long)]).to(device)
        y_pred = t.cat([y_pred1, y_pred2])
        tt.update_metrics(loss, y_pred, labels, batch_size=batch_size * 2)
        return loss

    def generate_train_tups(self, bert_model, train_ent1s, train_ent2s, train_links, device):
        bert_model.eval()
        all_emb1s = self.get_emb_valid(self.block_loader1, bert_model, device=device)
        all_emb2s = self.get_emb_valid(self.block_loader2, bert_model, device=device)
        train_ent_idx1s = [self.all_ent1s_p.index(e) for e in train_ent1s]
        train_ent_idx2s = [self.all_ent2s_p.index(e) for e in train_ent2s]
        train_emb1s = all_emb1s[train_ent_idx1s]
        train_emb2s = all_emb2s[train_ent_idx2s]
        candidate_dic1 = self.get_candidate_dict(train_ent1s, train_emb1s, self.all_ent2s_p, all_emb2s, device=device)
        candidate_dic2 = self.get_candidate_dict(train_ent2s, train_emb2s, self.all_ent1s_p, all_emb1s, device=device)
        train_tups = []
        for pe1, pe2 in train_links:
            for _ in range(self.neg_num):
                if np.random.rand() <= 0.5:
                    ne1s = candidate_dic2[pe2]
                    ne1 = ne1s[np.random.randint(self.nearest_sample_num)]
                    ne2 = pe2
                else:
                    ne1 = pe1
                    ne2s = candidate_dic1[pe1]
                    ne2 = ne2s[np.random.randint(self.nearest_sample_num)]
                if pe1 != ne1 or pe2 != ne2:
                    train_tups.append([pe1, pe2, ne1, ne2])
        return train_tups

    def get_candidate_dict(self, train_ents, train_embs, all_ents, all_embs, device):
        cos_sim_mat = cos_sim_mat_generate(train_embs, all_embs, device=device)
        _, topk_idx = batch_topk(cos_sim_mat, topn=self.nearest_sample_num, largest=True)
        topk_idx = topk_idx.tolist()
        candidate_dic = {
            train_ent: [all_ents[all_ent_idx] for all_ent_idx in all_ent_idxs]
            for train_ent, all_ent_idxs in zip(train_ents, topk_idx)
        }
        return candidate_dic

    @staticmethod
    def get_emb_valid(loader: DataLoader, model, device='cpu'):
        results = []
        with t.no_grad():
            model.eval()
            for i, batch in TrainingTools.batch_iter(loader, 'get embedding'):
                emb = model(batch[0].to(device), batch[1].to(device)).cpu()
                results.append(emb)
            embs = t.cat(results, dim=0)
        embs.requires_grad = False
        return embs

    @staticmethod
    def load_links(link_path, entity_ids1: dict, entity_ids2: dict):
        def links_line(line: str):
            parts = line.strip().split('\t')
            sbj = entity_ids1.get(parts[0])
            obj = entity_ids2.get(parts[1])
            return sbj, obj

        with open(link_path, 'r', encoding='utf-8') as rfile:
            lnks = [links_line(line) for line in rfile if line.strip()]
            lnks = list(filter(lambda x: x[0] is not None and x[1] is not None, lnks))
        return lnks

    @staticmethod
    def tids_solver(item, cls_token, sep_token, pad_token=0, freqs=None):
        eid, tids = item
        assert eid is not None
        assert len(tids) > 0
        tids = PairwiseTrainer.reduce_tokens(list(tids), max_len=seq_max_len)
        pad_length = seq_max_len - len(tids)
        input_ids = [cls_token] + tids + [pad_token] * pad_length
        masks = [1] * (len(tids) + 1) + [pad_token] * pad_length
        assert len(input_ids) == seq_max_len + 1
        assert len(masks) == seq_max_len + 1
        input_ids = t.tensor(input_ids, dtype=t.long)
        masks = t.tensor(masks, dtype=t.long)
        return eid, (input_ids, masks)

    @staticmethod
    def reduce_tokens(tids, max_len=200):
        while len(tids) > max_len:
            tids.pop()
        return tids
