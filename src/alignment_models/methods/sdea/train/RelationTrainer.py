import numpy as np
import torch as t
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

from .._globals import args, PARALLEL, SCORE_DISTANCE_LEVEL, MARGIN, links
from ..preprocess.KBStore import KBStore
from ..tools.Announce import Announce
from ..tools.ModelTools import ModelTools
from ..tools.TrainingTools import TrainingTools
from .PairwiseTrainer import PairwiseTrainer
from .RelationDataset import RelationDataset
from .RelationModel import RelationModel
from .RelationValidDataset import RelationValidDataset
from .train_utils import cos_sim_mat_generate, batch_topk, hits_open
import os

VALID = True


class RelationTrainer:
    def __init__(self):
        self.get_emb_batch = 512
        self.rel_batch = 256
        self.rel_valid_batch = 8
        self.nearest_sample_num = 128
        self.neg_num = 2
        self.score_distance_level = SCORE_DISTANCE_LEVEL

    def data_prepare(self, eid2tids1: dict, eid2tids2: dict, fs1: KBStore, fs2: KBStore):
        from .._globals import links as _links, args as _args
        self.fs1 = fs1
        self.fs2 = fs2
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(_args.pretrain_bert_path)
        cls_token, sep_token = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])

        eid2data1 = [PairwiseTrainer.tids_solver(item, cls_token=cls_token, sep_token=sep_token, freqs=None)
                     for item in eid2tids1.items()]
        self.eid2data1 = {key: value for key, value in eid2data1}

        eid2data2 = [PairwiseTrainer.tids_solver(item, cls_token=cls_token, sep_token=sep_token, freqs=None)
                     for item in eid2tids2.items()]
        self.eid2data2 = {key: value for key, value in eid2data2}

        self.train_links = PairwiseTrainer.load_links(_links.train, fs1.entity_ids, fs2.entity_ids)
        self.valid_links = PairwiseTrainer.load_links(_links.valid, fs1.entity_ids, fs2.entity_ids)
        self.test_links = PairwiseTrainer.load_links(_links.test, fs1.entity_ids, fs2.entity_ids)
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

        # Correct column indices for open evaluation (rank against all KG2 entities)
        ent2s_p_set = {e: i for i, e in enumerate(self.all_ent2s_p)}
        self.valid_correct_idx2s = [ent2s_p_set[e2] for e2 in self.valid_ent2s_p]
        self.test_correct_idx2s = [ent2s_p_set[e2] for e2 in self.test_ent2s_p]

        self.block_loader1, self.block_loader2 = self.links_pair_loader(self.all_ent1s_p, self.all_ent2s_p)

    def links_pair_loader(self, ent1s, ent2s):
        inputs1 = RelationTrainer.get_tensor_data(ent1s, self.eid2data1)
        inputs2 = RelationTrainer.get_tensor_data(ent2s, self.eid2data2)
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

    def train(self, basic_bert_path, epochs=100, device='cpu'):
        from .._globals import links as _links
        bert_model = ModelTools.load_model(basic_bert_path)
        if PARALLEL and t.cuda.device_count() > 1:
            bert_model = t.nn.DataParallel(bert_model)
        bert_model.to(device)
        bert_model.eval()

        # Get/cache BERT entity embeddings
        if os.path.exists(_links.kb_prop_emb_1):
            all_embed1s_p = t.load(_links.kb_prop_emb_1, map_location=device)
        else:
            all_embed1s_p = PairwiseTrainer.get_emb_valid(self.block_loader1, bert_model, device=device)
            t.save(all_embed1s_p, _links.kb_prop_emb_1)

        if os.path.exists(_links.kb_prop_emb_2):
            all_embed2s_p = t.load(_links.kb_prop_emb_2, map_location=device)
        else:
            all_embed2s_p = PairwiseTrainer.get_emb_valid(self.block_loader2, bert_model, device=device)
            t.save(all_embed2s_p, _links.kb_prop_emb_2)

        train_tups = self.generate_train_tups(
            bert_model, self.train_ent1s_p, self.train_ent2s_p, self.train_links_p,
            all_embed1s_p, all_embed2s_p, device
        )

        # Build full embedding matrices (index-aligned with entity_ids)
        all_embed1s = t.zeros(
            (len(self.all_ent1s) + 1, all_embed1s_p.shape[1]), dtype=all_embed1s_p.dtype, requires_grad=False
        )
        all_embed2s = t.zeros(
            (len(self.all_ent2s) + 1, all_embed2s_p.shape[1]), dtype=all_embed2s_p.dtype, requires_grad=False
        )
        for idx, embed in zip(self.all_ent1s_p, all_embed1s_p):
            all_embed1s[idx] = embed
        for idx, embed in zip(self.all_ent2s_p, all_embed2s_p):
            all_embed2s[idx] = embed

        all_embed1s = all_embed1s.to(device)
        all_embed2s = all_embed2s.to(device)

        rel_model = RelationModel(
            len(self.fs1.relation_ids), len(self.fs2.relation_ids),
            all_embed1s, all_embed2s, device
        )
        optimizer = Adam(rel_model.parameters(), lr=0.001, weight_decay=5e-4)

        train_tups_r = [
            (pe1, pe2, ne1, ne2) for pe1, pe2, ne1, ne2 in train_tups
            if pe1 in self.fs1.facts and pe2 in self.fs2.facts
            and ne1 in self.fs1.facts and ne2 in self.fs2.facts
        ]
        print(Announce.printMessage(), 'train_tups len:', len(train_tups))
        print(Announce.printMessage(), 'train_tups_r len:', len(train_tups_r))

        # Loaders for ALL KG2 entities (used as candidate pool in open evaluation)
        all_link_loader_r2 = RelationValidDataset(self.all_ent2s_p, self.fs2, all_embed2s, self.rel_valid_batch)

        if VALID:
            model_tool1 = ModelTools(5, 'max')
            model_tool2 = ModelTools(5, 'max')
            valid_link_loader_r1 = RelationValidDataset(self.valid_ent1s, self.fs1, all_embed1s, self.rel_valid_batch)

        test_link_loader_r1 = RelationValidDataset(self.test_ent1s_p, self.fs1, all_embed1s, self.rel_valid_batch)

        for epoch in range(1, epochs + 1):
            print(Announce.doing(), 'Epoch', epoch, '/', epochs, 'start')
            rel_model.train()
            rel_train_ds = RelationDataset(train_tups_r, self.fs1, self.fs2, all_embed1s, all_embed2s)
            train_loader = DataLoader(rel_train_ds, sampler=SequentialSampler(rel_train_ds), batch_size=self.rel_batch)
            tt = TrainingTools(train_loader, device=device)
            stop1 = stop2 = False
            for i, batch in tt.batches(lambda batch: len(batch[0])):
                optimizer.zero_grad()
                y, labels, loss = rel_model(*batch)
                tt.update_metrics(loss, y, labels, y.shape[0])
                print(float(loss.cpu()), end='')
                loss.backward()
                optimizer.step()
            print()

            rel_model.eval()
            if VALID:
                valid_link_loader_r1 = RelationValidDataset(self.valid_ent1s, self.fs1, all_embed1s, self.rel_valid_batch)
                all_link_loader_r2 = RelationValidDataset(self.all_ent2s_p, self.fs2, all_embed2s, self.rel_valid_batch)
                hit_values1, mrr1 = self.get_hits_r(rel_model, valid_link_loader_r1, self.valid_correct_idx2s, all_link_loader_r2, 'rel', device=device)
                all_link_loader_r2 = RelationValidDataset(self.all_ent2s_p, self.fs2, all_embed2s, self.rel_valid_batch)
                hit_values2, mrr2 = self.get_hits_r(rel_model, valid_link_loader_r1, self.valid_correct_idx2s, all_link_loader_r2, 'all', device=device)
                if epoch > 5:
                    stop1 = model_tool1.early_stopping(rel_model, _links.rel_model_save, hit_values1[0])
                    stop2 = model_tool2.early_stopping(rel_model, _links.rel_model_save, hit_values2[0])

            test_link_loader_r1 = RelationValidDataset(self.test_ent1s_p, self.fs1, all_embed1s, self.rel_valid_batch)
            all_link_loader_r2 = RelationValidDataset(self.all_ent2s_p, self.fs2, all_embed2s, self.rel_valid_batch)
            self.get_hits_r(rel_model, test_link_loader_r1, self.test_correct_idx2s, all_link_loader_r2, 'rel', device=device)
            test_link_loader_r1 = RelationValidDataset(self.test_ent1s_p, self.fs1, all_embed1s, self.rel_valid_batch)
            all_link_loader_r2 = RelationValidDataset(self.all_ent2s_p, self.fs2, all_embed2s, self.rel_valid_batch)
            self.get_hits_r(rel_model, test_link_loader_r1, self.test_correct_idx2s, all_link_loader_r2, 'all', device=device)
            print(Announce.done(), 'Epoch', epoch, '/', epochs, 'end')

            if VALID and epoch > 20 and stop1 and stop2:
                print(Announce.done(), 'early stopping')
                break

        # Load best model for final evaluation
        print(Announce.printMessage(), 'Final Result')
        if os.path.exists(_links.rel_model_save):
            rel_model = ModelTools.load_model(_links.rel_model_save)
            rel_model.to(device)

        test_link_loader_r1 = RelationValidDataset(self.test_ent1s_p, self.fs1, all_embed1s, self.rel_valid_batch)
        all_link_loader_r2 = RelationValidDataset(self.all_ent2s_p, self.fs2, all_embed2s, self.rel_valid_batch)
        _, _ = self.get_hits_r(rel_model, test_link_loader_r1, self.test_correct_idx2s, all_link_loader_r2, 'rel', device=device)

        test_link_loader_r1 = RelationValidDataset(self.test_ent1s_p, self.fs1, all_embed1s, self.rel_valid_batch)
        all_link_loader_r2 = RelationValidDataset(self.all_ent2s_p, self.fs2, all_embed2s, self.rel_valid_batch)
        final_hit_values, final_mrr = self.get_hits_r(
            rel_model, test_link_loader_r1, self.test_correct_idx2s, all_link_loader_r2, 'all', device=device
        )

        return {
            "hits@1": final_hit_values[0] if len(final_hit_values) > 0 else 0.0,
            "hits@5": final_hit_values[1] if len(final_hit_values) > 1 else 0.0,
            "hits@10": final_hit_values[2] if len(final_hit_values) > 2 else 0.0,
            "mrr": final_mrr,
        }

    def get_hits_r(self, rel_model: RelationModel, src_loader1, correct_indices, all_loader2, mode, device='cpu'):
        """Open evaluation: rank each source entity against ALL KG2 entities."""
        print('hits_r:', mode)
        rel_model.eval()
        src_emb1s = self.get_emb_valid_r(src_loader1, rel_model, rel_model.rel_embedding1, rel_model.ent_embedding1, mode, device=device)
        all_emb2s = self.get_emb_valid_r(all_loader2, rel_model, rel_model.rel_embedding2, rel_model.ent_embedding2, mode, device=device)
        cos_sim_mat = cos_sim_mat_generate(src_emb1s, all_emb2s, device=device)
        _, topk_idx = batch_topk(cos_sim_mat, topn=self.nearest_sample_num, largest=True)
        return hits_open(topk_idx, correct_indices)

    def generate_train_tups(self, bert_model, train_ent1s, train_ent2s, train_links, all_emb1s, all_emb2s, device):
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
        return {
            train_ent: [all_ents[all_ent_idx] for all_ent_idx in all_ent_idxs]
            for train_ent, all_ent_idxs in zip(train_ents, topk_idx)
        }

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
    def get_emb_valid_r(loader, model: RelationModel, rel_embedding, all_embed, mode, device='cpu'):
        results = []
        with t.no_grad():
            model.eval()
            for batch in loader:
                emb = model.get_valid_emb(batch, rel_embedding, all_embed, mode).cpu()
                results.append(emb)
            embs = t.cat(results, dim=0)
        return embs
