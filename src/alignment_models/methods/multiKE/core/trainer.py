"""MultiKETrainer: PyTorch-based training of MultiKE_Late."""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing

from .attr_batch import generate_all_attribute_batches
from .batch import generate_all_relation_batches, generate_neighbours
from .conv_net import AttributeConvNet
from .evaluation import evaluate
from .losses import (
    alignment_loss,
    logistic_loss_wo_negs,
    relation_logistic_loss,
    relation_logistic_loss_wo_negs,
    space_mapping_loss,
)
from .predicate_alignment import PredicateAlignModel


def _xavier(shape, device='cpu'):
    t = torch.empty(*shape, device=device)
    nn.init.xavier_uniform_(t)
    return nn.Parameter(t)


def _orthogonal(shape, device='cpu'):
    t = torch.empty(*shape, device=device)
    nn.init.orthogonal_(t)
    return nn.Parameter(t)


def _make_optimizer(name: str, params, lr: float):
    return {
        'Adagrad': torch.optim.Adagrad,
        'Adadelta': torch.optim.Adadelta,
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
    }.get(name, torch.optim.SGD)(params, lr=lr)


def _lookup(table: torch.Tensor, ids) -> torch.Tensor:
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids, dtype=torch.long, device=table.device)
    return table[ids]


class MultiKETrainer(nn.Module):
    """
    PyTorch trainer for the MultiKE_Late pipeline.

    Embedding tables
    ----------------
    literal_embeds   : buffer [n_values, dim]    — fixed, not trained
    name_embeds      : buffer [n_entities, dim]  — fixed, not trained
    rv_ent_embeds    : param  [n_entities, dim]  — relation view
    rel_embeds       : param  [n_relations, dim]
    av_ent_embeds    : param  [n_entities, dim]  — attribute view
    attr_embeds      : buffer [n_attributes, dim]— fixed random hash
    ent_embeds       : param  [n_entities, dim]  — shared
    nv_mapping       : param  [dim, dim]
    rv_mapping       : param  [dim, dim]
    av_mapping       : param  [dim, dim]
    conv_net         : AttributeConvNet
    """

    def __init__(self, data_model, predicate_align_model: PredicateAlignModel, args,
                 device: str = 'cpu'):
        super().__init__()
        self.data = data_model
        self.pred_align = predicate_align_model
        self.args = args
        self.kgs = data_model.kgs
        self.kg1 = self.kgs.kg1
        self.kg2 = self.kgs.kg2
        self.device = torch.device(device)
        dim = args.dim

        # Fixed buffers (not trained)
        self.register_buffer('literal_embeds',
                              torch.tensor(data_model.value_vectors, dtype=torch.float32))
        self.register_buffer('name_embeds',
                              torch.tensor(data_model.local_name_vectors, dtype=torch.float32))
        self.register_buffer('attr_embeds',
                              torch.zeros(self.kgs.attributes_num, dim).uniform_(-0.1, 0.1))
        self.register_buffer('eye_mat',
                              torch.eye(dim))

        # Trainable parameters
        self.rv_ent_embeds = _xavier([self.kgs.entities_num, dim])
        self.rel_embeds    = _xavier([self.kgs.relations_num, dim])
        self.av_ent_embeds = _xavier([self.kgs.entities_num, dim])
        self.ent_embeds    = _xavier([self.kgs.entities_num, dim])
        self.nv_mapping    = _orthogonal([dim, dim])
        self.rv_mapping    = _orthogonal([dim, dim])
        self.av_mapping    = _orthogonal([dim, dim])

        # CNN for attribute scoring
        self.conv_net = AttributeConvNet(dim)

        self.to(self.device)

        # Separate optimizers per training phase
        lr  = getattr(args, 'learning_rate', 0.001)
        ilr = getattr(args, 'ITC_learning_rate', 0.004)
        opt = getattr(args, 'optimizer', 'Adagrad')

        rv_params = [self.rv_ent_embeds, self.rel_embeds]
        av_params = [self.av_ent_embeds] + list(self.conv_net.parameters())
        shared_params = [self.ent_embeds, self.nv_mapping, self.rv_mapping, self.av_mapping]
        itc_params = [self.rv_ent_embeds, self.av_ent_embeds, self.ent_embeds]

        self.rv_opt      = _make_optimizer(opt, rv_params, lr)
        self.av_opt      = _make_optimizer(opt, av_params, lr)
        self.ckge_rv_opt = _make_optimizer(opt, rv_params, lr)
        self.ckge_av_opt = _make_optimizer(opt, av_params, lr)
        self.ckgp_rv_opt = _make_optimizer(opt, rv_params, lr)
        self.ckga_av_opt = _make_optimizer(opt, av_params, lr)
        self.shared_opt  = _make_optimizer(opt, shared_params, lr)
        self.itc_opt     = _make_optimizer(opt, itc_params, ilr)

    # ------------------------------------------------------------------ #
    #  Relation view                                                       #
    # ------------------------------------------------------------------ #

    def train_relation_view_1epo(self, epoch, neighbor1=None, neighbor2=None):
        kg1, kg2 = self.kg1, self.kg2
        total = kg1.local_relation_triples_num + kg2.local_relation_triples_num
        steps = int(math.ceil(total / self.args.batch_size))
        batches = generate_all_relation_batches(
            kg1.local_relation_triples_list, kg2.local_relation_triples_list,
            kg1.local_relation_triples_set, kg2.local_relation_triples_set,
            kg1.entities_list, kg2.entities_list,
            self.args.batch_size, steps, neighbor1, neighbor2, self.args.neg_triple_num)

        t0 = time.time(); epoch_loss = n = 0
        for pos_batch, neg_batch in batches:
            if not pos_batch:
                continue
            ph = torch.tensor([x[0] for x in pos_batch], device=self.device)
            pr = torch.tensor([x[1] for x in pos_batch], device=self.device)
            pt = torch.tensor([x[2] for x in pos_batch], device=self.device)
            nh = torch.tensor([x[0] for x in neg_batch], device=self.device)
            nr = torch.tensor([x[1] for x in neg_batch], device=self.device)
            nt = torch.tensor([x[2] for x in neg_batch], device=self.device)
            self.rv_opt.zero_grad()
            loss = relation_logistic_loss(
                self.rv_ent_embeds[ph], self.rel_embeds[pr], self.rv_ent_embeds[pt],
                self.rv_ent_embeds[nh], self.rel_embeds[nr], self.rv_ent_embeds[nt])
            loss.backward()
            self.rv_opt.step()
            epoch_loss += loss.item(); n += len(pos_batch)

        random.shuffle(kg1.local_relation_triples_list)
        random.shuffle(kg2.local_relation_triples_list)
        print(f'epoch {epoch} of rel. view, avg. loss: {epoch_loss/max(n,1):.4f}, '
              f'time: {time.time()-t0:.4f}s')

    # ------------------------------------------------------------------ #
    #  Attribute view                                                      #
    # ------------------------------------------------------------------ #

    def train_attribute_view_1epo(self, epoch, neighbor1=None, neighbor2=None):
        pa = self.pred_align
        total = len(pa.attribute_triples_w_weights1) + len(pa.attribute_triples_w_weights2)
        steps = int(math.ceil(total / self.args.attribute_batch_size))
        batches = generate_all_attribute_batches(
            pa.attribute_triples_w_weights1, pa.attribute_triples_w_weights2,
            pa.attribute_triples_w_weights_set1, pa.attribute_triples_w_weights_set2,
            self.kg1.entities_list, self.kg2.entities_list,
            self.args.attribute_batch_size, steps, neighbor1, neighbor2)

        t0 = time.time(); epoch_loss = n = 0
        for pos_batch, _ in batches:
            if not pos_batch:
                continue
            ph = torch.tensor([x[0] for x in pos_batch], device=self.device)
            pa_ = torch.tensor([x[1] for x in pos_batch], device=self.device)
            pv = torch.tensor([x[2] for x in pos_batch], device=self.device)
            pw = torch.tensor([x[3] for x in pos_batch], dtype=torch.float32, device=self.device)
            self.conv_net.train()
            self.av_opt.zero_grad()
            score = self.conv_net(self.av_ent_embeds[ph], self.attr_embeds[pa_], self.literal_embeds[pv])
            loss = torch.sum(F.softplus(-score) * pw)
            loss.backward()
            self.av_opt.step()
            epoch_loss += loss.item(); n += len(pos_batch)

        random.shuffle(pa.attribute_triples_w_weights1)
        random.shuffle(pa.attribute_triples_w_weights2)
        print(f'epoch {epoch} of att. view, avg. loss: {epoch_loss/max(n,1):.4f}, '
              f'time: {time.time()-t0:.4f}s')

    # ------------------------------------------------------------------ #
    #  Cross-KG entity inference                                          #
    # ------------------------------------------------------------------ #

    def _batch_steps(self, triples, batch_size):
        steps = int(math.ceil(len(triples) / batch_size))
        return batch_size if steps > 1 else len(triples)

    def train_cross_kg_entity_inference_rv_1epo(self, epoch, sup_triples):
        if not sup_triples:
            return
        t0 = time.time(); epoch_loss = n = 0
        bs = self._batch_steps(sup_triples, self.args.batch_size)
        for _ in range(int(math.ceil(len(sup_triples) / bs))):
            batch = random.sample(sup_triples, bs)
            ph = torch.tensor([x[0] for x in batch], device=self.device)
            pr = torch.tensor([x[1] for x in batch], device=self.device)
            pt = torch.tensor([x[2] for x in batch], device=self.device)
            self.ckge_rv_opt.zero_grad()
            loss = 2.0 * relation_logistic_loss_wo_negs(
                self.rv_ent_embeds[ph], self.rel_embeds[pr], self.rv_ent_embeds[pt])
            loss.backward()
            self.ckge_rv_opt.step()
            epoch_loss += loss.item(); n += len(batch)
        print(f'epoch {epoch} of cross-kg entity inf. (rv), avg. loss: {epoch_loss/max(n,1):.4f}, '
              f'time: {time.time()-t0:.4f}s')

    def train_cross_kg_entity_inference_av_1epo(self, epoch, sup_triples):
        if not sup_triples:
            return
        t0 = time.time(); epoch_loss = n = 0
        bs = self._batch_steps(sup_triples, self.args.attribute_batch_size)
        for _ in range(int(math.ceil(len(sup_triples) / bs))):
            batch = random.sample(sup_triples, bs)
            ph = torch.tensor([x[0] for x in batch], device=self.device)
            pa = torch.tensor([x[1] for x in batch], device=self.device)
            pv = torch.tensor([x[2] for x in batch], device=self.device)
            self.conv_net.train()
            self.ckge_av_opt.zero_grad()
            score = self.conv_net(self.av_ent_embeds[ph], self.attr_embeds[pa], self.literal_embeds[pv])
            loss = 2.0 * torch.sum(F.softplus(-score))
            loss.backward()
            self.ckge_av_opt.step()
            epoch_loss += loss.item(); n += len(batch)
        print(f'epoch {epoch} of cross-kg entity inf. (av), avg. loss: {epoch_loss/max(n,1):.4f}, '
              f'time: {time.time()-t0:.4f}s')

    # ------------------------------------------------------------------ #
    #  Cross-KG predicate inference                                       #
    # ------------------------------------------------------------------ #

    def train_cross_kg_relation_inference_1epo(self, epoch, sup_triples):
        if not sup_triples:
            return
        t0 = time.time(); epoch_loss = n = 0
        bs = self._batch_steps(sup_triples, self.args.batch_size)
        for _ in range(int(math.ceil(len(sup_triples) / bs))):
            batch = random.sample(sup_triples, bs)
            ph = torch.tensor([x[0] for x in batch], device=self.device)
            pr = torch.tensor([x[1] for x in batch], device=self.device)
            pt = torch.tensor([x[2] for x in batch], device=self.device)
            pw = torch.tensor([x[3] for x in batch], dtype=torch.float32, device=self.device)
            self.ckgp_rv_opt.zero_grad()
            loss = 2.0 * logistic_loss_wo_negs(
                self.rv_ent_embeds[ph], self.rel_embeds[pr], self.rv_ent_embeds[pt], pw)
            loss.backward()
            self.ckgp_rv_opt.step()
            epoch_loss += loss.item(); n += len(batch)
        print(f'epoch {epoch} of cross-kg rel. inf., avg. loss: {epoch_loss/max(n,1):.4f}, '
              f'time: {time.time()-t0:.4f}s')

    def train_cross_kg_attribute_inference_1epo(self, epoch, sup_triples):
        if not sup_triples:
            return
        t0 = time.time(); epoch_loss = n = 0
        bs = self._batch_steps(sup_triples, self.args.attribute_batch_size)
        for _ in range(int(math.ceil(len(sup_triples) / bs))):
            batch = random.sample(sup_triples, bs)
            ph = torch.tensor([x[0] for x in batch], device=self.device)
            pa = torch.tensor([x[1] for x in batch], device=self.device)
            pv = torch.tensor([x[2] for x in batch], device=self.device)
            pw = torch.tensor([x[3] for x in batch], dtype=torch.float32, device=self.device)
            self.conv_net.train()
            self.ckga_av_opt.zero_grad()
            score = self.conv_net(self.av_ent_embeds[ph], self.attr_embeds[pa], self.literal_embeds[pv])
            loss = torch.sum(F.softplus(-score) * pw)
            loss.backward()
            self.ckga_av_opt.step()
            epoch_loss += loss.item(); n += len(batch)
        print(f'epoch {epoch} of cross-kg attr. inf., avg. loss: {epoch_loss/max(n,1):.4f}, '
              f'time: {time.time()-t0:.4f}s')

    # ------------------------------------------------------------------ #
    #  Common space learning (ITC)                                        #
    # ------------------------------------------------------------------ #

    def train_common_space_learning_1epo(self, epoch, entities):
        t0 = time.time(); epoch_loss = n = 0
        bs = self._batch_steps(entities, self.args.entity_batch_size)
        cv_name_w = getattr(self.args, 'cv_name_weight', 1.0)
        cv_w      = getattr(self.args, 'cv_weight', 1.0)
        for _ in range(int(math.ceil(len(entities) / bs))):
            ids = torch.tensor(random.sample(entities, bs), device=self.device)
            self.itc_opt.zero_grad()
            final_e = self.ent_embeds[ids]
            name_e  = self.name_embeds[ids]
            rv_e    = self.rv_ent_embeds[ids]
            av_e    = self.av_ent_embeds[ids]
            loss = cv_w * (cv_name_w * alignment_loss(final_e, name_e)
                           + alignment_loss(final_e, rv_e)
                           + alignment_loss(final_e, av_e))
            loss.backward()
            self.itc_opt.step()
            epoch_loss += loss.item(); n += len(ids)
        print(f'epoch {epoch} of common space learning, avg. loss: {epoch_loss/max(n,1):.4f}, '
              f'time: {time.time()-t0:.4f}s')

    # ------------------------------------------------------------------ #
    #  Shared space mapping (Phase 2)                                     #
    # ------------------------------------------------------------------ #

    def train_shared_space_mapping_1epo(self, epoch, entities):
        t0 = time.time(); epoch_loss = n = 0
        bs = self._batch_steps(entities, self.args.entity_batch_size)
        orth_w    = getattr(self.args, 'orthogonal_weight', 2.0)
        cv_name_w = getattr(self.args, 'cv_name_weight', 1.0)
        for _ in range(int(math.ceil(len(entities) / bs))):
            ids = torch.tensor(random.sample(entities, bs), device=self.device)
            self.shared_opt.zero_grad()
            final_e = self.ent_embeds[ids]
            rv_e    = self.rv_ent_embeds[ids]
            av_e    = self.av_ent_embeds[ids]
            loss = (space_mapping_loss(rv_e, final_e, self.rv_mapping, self.eye_mat, orth_w)
                    + space_mapping_loss(av_e, final_e, self.av_mapping, self.eye_mat, orth_w))
            if cv_name_w > 0:
                nv_e = self.name_embeds[ids]
                loss = loss + cv_name_w * space_mapping_loss(nv_e, final_e, self.nv_mapping, self.eye_mat, orth_w)
            loss.backward()
            self.shared_opt.step()
            epoch_loss += loss.item(); n += len(ids)
        print(f'epoch {epoch} of shared space mapping, avg. loss: {epoch_loss/max(n,1):.4f}, '
              f'time: {time.time()-t0:.4f}s')

    # ------------------------------------------------------------------ #
    #  Embedding retrieval                                                 #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _get_embeds(self, entity_ids, choice, w=(1.0, 1.0, 1.0)):
        ids = torch.tensor(entity_ids, device=self.device)
        if choice == 'nv':
            return self.name_embeds[ids].cpu().numpy()
        elif choice == 'rv':
            return self.rv_ent_embeds[ids].cpu().numpy()
        elif choice == 'av':
            return self.av_ent_embeds[ids].cpu().numpy()
        elif choice == 'final':
            return self.ent_embeds[ids].cpu().numpy()
        else:  # avg
            cv_name_w = getattr(self.args, 'cv_name_weight', 1.0)
            rv = self.rv_ent_embeds[ids]
            av = self.av_ent_embeds[ids]
            if cv_name_w > 0:
                nv = self.name_embeds[ids]
                return (w[0] * cv_name_w * nv + w[1] * rv + w[2] * av).cpu().numpy()
            return (w[1] * rv + w[2] * av).cpu().numpy()

    def get_test_embeds(self, choice='avg'):
        return (self._get_embeds(self.kgs.test_entities1, choice),
                self._get_embeds(self.kgs.test_entities2, choice))

    def get_valid_embeds(self, choice='avg'):
        return (self._get_embeds(self.kgs.valid_entities1, choice),
                self._get_embeds(self.kgs.valid_entities2 + self.kgs.test_entities2, choice))

    @torch.no_grad()
    def get_kg1_useful_embeds(self):
        return self._get_embeds(self.kgs.useful_entities_list1, 'rv')

    @torch.no_grad()
    def get_kg2_useful_embeds(self):
        return self._get_embeds(self.kgs.useful_entities_list2, 'rv')

    # ------------------------------------------------------------------ #
    #  Full training loop                                                  #
    # ------------------------------------------------------------------ #

    def run(self) -> Dict[str, float]:
        args = self.args
        kgs = self.kgs
        entity_list = kgs.kg1.entities_list + kgs.kg2.entities_list
        cross_rel    = kgs.kg1.sup_relation_triples_list + kgs.kg2.sup_relation_triples_list
        cross_attr   = kgs.kg1.sup_attribute_triples_list + kgs.kg2.sup_attribute_triples_list
        rel_inf  = (self.pred_align.sup_relation_alignment_triples1
                    + self.pred_align.sup_relation_alignment_triples2)
        attr_inf = (self.pred_align.sup_attribute_alignment_triples1
                    + self.pred_align.sup_attribute_alignment_triples2)

        neighbor1 = neighbor2 = None
        start_pred = getattr(args, 'start_predicate_soft_alignment', 10)
        neg_samp   = getattr(args, 'neg_sampling', 'truncated')
        trunc_eps  = getattr(args, 'truncated_epsilon', 0.98)
        trunc_freq = getattr(args, 'truncated_freq', 20)
        start_valid = getattr(args, 'start_valid', 100)
        eval_freq   = getattr(args, 'eval_freq', 10)
        max_epoch   = getattr(args, 'max_epoch', 200)
        shared_max  = getattr(args, 'shared_learning_max_epoch', 200)
        top_k       = tuple(getattr(args, 'top_k', [1, 5, 10]))
        threads     = getattr(args, 'test_threads_num', 4)

        best_mrr = 0.0
        best_metrics: Dict[str, float] = {}

        for i in range(1, max_epoch + 1):
            print(f'epoch {i}:')
            self.train_relation_view_1epo(i, neighbor1, neighbor2)
            self.train_cross_kg_entity_inference_rv_1epo(i, cross_rel)
            if i > start_pred:
                self.train_cross_kg_relation_inference_1epo(i, rel_inf)

            self.train_attribute_view_1epo(i, neighbor1, neighbor2)
            self.train_cross_kg_entity_inference_av_1epo(i, cross_attr)
            if i > start_pred:
                self.train_cross_kg_attribute_inference_1epo(i, attr_inf)

            self.train_common_space_learning_1epo(i, entity_list)

            if i >= start_valid and i % eval_freq == 0:
                for view in ('rv', 'av', 'avg'):
                    e1, e2 = self.get_valid_embeds(view)
                    m = evaluate(e1, e2, top_k=top_k, normalize=True)
                    k10 = m.get('hits@10', m.get(f'hits@{max(top_k)}', 0.0))
                    print(f'  [valid/{view}] hits@1={m["hits@1"]:.4f} hits@10={k10:.4f} mrr={m["mrr"]:.4f}')
                    if m['mrr'] > best_mrr:
                        best_mrr = m['mrr']
                        best_metrics = m

                if i >= start_pred and i % 10 == 0:
                    self.pred_align.update_predicate_alignment(
                        self.rel_embeds.detach().cpu().numpy())
                    self.pred_align.update_predicate_alignment(
                        self.attr_embeds.detach().cpu().numpy(), predicate_type='attribute')
                    rel_inf  = (self.pred_align.sup_relation_alignment_triples1
                                + self.pred_align.sup_relation_alignment_triples2)
                    attr_inf = (self.pred_align.sup_attribute_alignment_triples1
                                + self.pred_align.sup_attribute_alignment_triples2)

            if neg_samp == 'truncated' and i % trunc_freq == 0:
                n1 = min(int((1 - trunc_eps) * len(kgs.kg1.entities_list)),
                         len(kgs.useful_entities_list1) - 1)
                n2 = min(int((1 - trunc_eps) * len(kgs.kg2.entities_list)),
                         len(kgs.useful_entities_list2) - 1)
                if n1 > 0:
                    neighbor1 = generate_neighbours(
                        self.get_kg1_useful_embeds(), kgs.useful_entities_list1, n1, threads)
                if n2 > 0:
                    neighbor2 = generate_neighbours(
                        self.get_kg2_useful_embeds(), kgs.useful_entities_list2, n2, threads)

        # Phase 2: shared space mapping
        for i in range(1, shared_max + 1):
            self.train_shared_space_mapping_1epo(i, entity_list)
            if i >= start_valid and i % eval_freq == 0:
                e1, e2 = self.get_valid_embeds('final')
                m = evaluate(e1, e2, top_k=top_k, normalize=True)
                print(f'  [valid/final] hits@1={m["hits@1"]:.4f} mrr={m["mrr"]:.4f}')
                if m['mrr'] > best_mrr:
                    best_mrr = m['mrr']
                    best_metrics = m

        # Final test evaluation across all embedding choices
        test_metrics: Dict[str, float] = {}
        for choice in ('nv', 'rv', 'av', 'avg', 'final'):
            e1, e2 = self.get_test_embeds(choice)
            m = evaluate(e1, e2, top_k=top_k, normalize=True)
            k10 = m.get('hits@10', m.get(f'hits@{max(top_k)}', 0.0))
            print(f'  [test/{choice}] hits@1={m["hits@1"]:.4f} hits@10={k10:.4f} mrr={m["mrr"]:.4f}')
            if m['mrr'] > test_metrics.get('mrr', 0.0):
                test_metrics = m

        return test_metrics if test_metrics else best_metrics
