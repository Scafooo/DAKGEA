import torch as t

from .._globals import bert_output_dim, MARGIN, SCORE_DISTANCE_LEVEL
from .GRUAttnNet import GRUAttnNet
from .HighwayNet import Highway


class RelationModel(t.nn.Module):
    def __init__(self, rel_count1, rel_count2, all_embed1, all_embed2, device):
        super(RelationModel, self).__init__()
        self.device = device
        self.score_distance_level = SCORE_DISTANCE_LEVEL
        self.rel_count1 = rel_count1
        self.rel_count2 = rel_count2
        self.rel_embedding1 = None
        self.rel_embedding2 = None
        embedding_dim = bert_output_dim

        rnn_hidden_dim = 64
        self.rnn = GRUAttnNet(embedding_dim, rnn_hidden_dim, 1, dropout=0.2, device=device)
        self.bn = t.nn.Sequential(
            t.nn.BatchNorm1d(rnn_hidden_dim),
            t.nn.Softsign(),
        ).to(device)
        attr_out_dim = 300
        self.combiner = t.nn.Sequential(
            t.nn.Linear(attr_out_dim + rnn_hidden_dim, attr_out_dim),
            t.nn.Dropout(0.15, inplace=True),
            t.nn.BatchNorm1d(attr_out_dim),
            t.nn.ReLU(inplace=True),
            Highway(attr_out_dim, device),
            t.nn.BatchNorm1d(attr_out_dim),
            t.nn.ReLU(),
        ).to(device)
        self.ent_embedding1 = t.nn.Embedding.from_pretrained(
            all_embed1.detach(), padding_idx=all_embed1.shape[0] - 1
        )
        self.ent_embedding2 = t.nn.Embedding.from_pretrained(
            all_embed2.detach(), padding_idx=all_embed2.shape[0] - 1
        )
        self.to(device)

    def forward(self, pe1s, pe2s, ne1s, ne2s, bpn1s, bpn2s, bnn1s, bnn2s, bpr1s, bpr2s, bnr1s, bnr2s):
        pe1s = pe1s.to(self.device)
        pe2s = pe2s.to(self.device)
        ne1s = ne1s.to(self.device)
        ne2s = ne2s.to(self.device)
        bpn1s = bpn1s.to(self.device)
        bpn2s = bpn2s.to(self.device)
        bnn1s = bnn1s.to(self.device)
        bnn2s = bnn2s.to(self.device)
        bpr1s = bpr1s.to(self.device)
        bpr2s = bpr2s.to(self.device)
        bnr1s = bnr1s.to(self.device)
        bnr2s = bnr2s.to(self.device)

        pr1s = self.get_rel_embeds(bpn1s, bpr1s, pe1s, self.rel_embedding1, self.ent_embedding1)
        pr2s = self.get_rel_embeds(bpn2s, bpr2s, pe2s, self.rel_embedding2, self.ent_embedding2)
        nr1s = self.get_rel_embeds(bnn1s, bnr1s, ne1s, self.rel_embedding1, self.ent_embedding1)
        nr2s = self.get_rel_embeds(bnn2s, bnr2s, ne2s, self.rel_embedding2, self.ent_embedding2)

        pos_emb1 = t.cat((pe1s, pr1s), dim=1)
        pos_emb2 = t.cat((pe2s, pr2s), dim=1)
        pos_emb1 = self.combiner(pos_emb1)
        pos_emb2 = self.combiner(pos_emb2)
        pos_emb1 = t.cat((pr1s, pos_emb1), dim=1)
        pos_emb2 = t.cat((pr2s, pos_emb2), dim=1)

        pos_score = t.nn.functional.pairwise_distance(pos_emb1, pos_emb2, p=self.score_distance_level, keepdim=True)
        y_pred1 = t.cosine_similarity(pos_emb1, pos_emb2).reshape(pe1s.shape[0], 1)
        y1_0 = t.ones([pe1s.shape[0], 1], device=self.device) - y_pred1
        y_pred1 = t.cat([y1_0, y_pred1], dim=1)

        neg_emb1 = t.cat((ne1s, nr1s), dim=1)
        neg_emb2 = t.cat((ne2s, nr2s), dim=1)
        neg_emb1 = self.combiner(neg_emb1)
        neg_emb2 = self.combiner(neg_emb2)
        neg_emb1 = t.cat((nr1s, neg_emb1), dim=1)
        neg_emb2 = t.cat((nr2s, neg_emb2), dim=1)

        neg_score = t.nn.functional.pairwise_distance(neg_emb1, neg_emb2, p=self.score_distance_level, keepdim=True)
        y_pred2 = t.cosine_similarity(neg_emb1, neg_emb2).reshape(pe1s.shape[0], 1)
        y2_0 = t.ones([pe1s.shape[0], 1], device=self.device) - y_pred2
        y_pred2 = t.cat([y2_0, y_pred2], dim=1)

        y = -t.ones(pos_score.shape, device=self.device)
        loss = t.nn.MarginRankingLoss(MARGIN)(pos_score, neg_score, y)
        y_pred = t.cat((y_pred1, y_pred2))
        labels = t.cat((
            t.ones(y_pred1.shape[0], dtype=t.long, device=self.device),
            t.zeros(y_pred1.shape[0], dtype=t.long, device=self.device),
        ))
        return y_pred, labels, loss

    def get_valid_emb(self, batch, rel_embedding, all_embed, mode):
        with t.no_grad():
            ents, fs = batch
            ents = ents.to(self.device)
            pad_idx = all_embed.weight.shape[0] - 1
            bns, brs = self.get_neighbors_batch(fs, pad_idx, device=self.device)
            rel_embs = self.get_rel_embeds(bns, brs, ents, rel_embedding, all_embed)
            if mode == 'all':
                # Mirror the training forward(): combiner([BERT|GRU]) → [GRU | combined]
                combined = self.combiner(t.cat((ents, rel_embs), dim=1))
                final_embds = t.cat((rel_embs, combined), dim=1)
            elif mode == 'rel':
                final_embds = rel_embs
            else:
                raise ValueError(f'unknown mode: {mode}')
            return final_embds

    def case_study(self, batch, rel_embedding, all_embed, mode):
        with t.no_grad():
            ents, fs = batch
            ents = ents.to(self.device)
            bns, brs = self.get_neighbors_batch(fs, all_embed.weight.shape[0] - 1, device=self.device)
            pad_idx = all_embed.weight.shape[0] - 1
            ones = t.ones(bns.shape, device=self.device)
            zeros = t.zeros(bns.shape, device=self.device)
            neighbor_mask = t.where(bns == pad_idx, ones, zeros)
            batch_nei_embs = all_embed(bns)
            h = batch_nei_embs
            h_prime, weights = self.rnn(h, neighbor_mask)
            rel_embs = self.bn(h_prime)
            if mode == 'all':
                final_embds = t.cat((ents, rel_embs), dim=1).to(self.device)
            elif mode == 'rel':
                final_embds = rel_embs
            else:
                raise ValueError(f'unknown mode: {mode}')
            return final_embds, weights

    def get_rel_embeds(self, batch_neighbors, batch_relations, batch_ent, rel_embedding, all_embed):
        pad_idx = all_embed.weight.shape[0] - 1
        ones = t.ones(batch_neighbors.shape, device=self.device)
        zeros = t.zeros(batch_neighbors.shape, device=self.device)
        neighbor_mask = t.where(batch_neighbors == pad_idx, ones, zeros)
        batch_nei_embs = all_embed(batch_neighbors)
        h = batch_nei_embs
        h = t.nn.functional.relu(h)
        h_prime, _ = self.rnn(h, neighbor_mask)
        h_prime = self.bn(h_prime)
        return h_prime

    @staticmethod
    def get_neighbors_batch(batch_facts, pad_idx, device=None):
        if device is None:
            device = t.device('cpu')
        lens = [len(facts) if facts is not None else 0 for facts in batch_facts]
        N = max(max(lens), 1) if lens else 1
        batch_neighbors = [
            [ent for rel, ent in facts] if facts is not None else []
            for facts in batch_facts
        ]
        for neighbors in batch_neighbors:
            while len(neighbors) < N:
                neighbors.append(pad_idx)
        batch_neighbors = t.tensor(batch_neighbors, dtype=t.long, device=device)
        return batch_neighbors, None
