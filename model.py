import torch
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from layers import *
import torch.nn as nn
import numpy as np


class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        return

    def forward(self, y_pred_psa, region_ont_idx):
        pred1 = F.softmax(y_pred_psa, dim=1)
        loss_batch = - torch.log(pred1 + 1e-10)
        loss = loss_batch[range(pred1.shape[0]), region_ont_idx].mean()
        return loss



class MetaPathDecoder(torch.nn.Module):
    def __init__(self, d, **kwargs):
        super(MetaPathDecoder, self).__init__()
        edim = kwargs['edim']
        if kwargs['pretrain'] == 'true':
            freeze_bool = True if kwargs['freeze'] == 'true' else False
            self.E = nn.Embedding.from_pretrained(kwargs['E_pretrain'], freeze=freeze_bool)
            self.R = nn.Embedding.from_pretrained(kwargs['R_pretrain'], freeze=freeze_bool)
        else:
            self.E = torch.nn.Embedding(len(d.ent2id), edim)
            self.R = torch.nn.Embedding(len(d.rel2id), edim)
            self.init()
        self.psa_loss = MyLoss()
        self.layers = nn.ModuleList()
        self.n_layer = kwargs['n_layer']
        for i in range(kwargs['n_layer']):
            self.layers.append(MyCompGCN(indim=edim, outdim=edim, nr=len(d.rel2id), dropout=kwargs['gcn_dropout'], opn=kwargs['opn'], etype2eids=kwargs['etype2eids']))
        self.d = d

        self.rel_path_str = [['rel_placestoreat', 'rel_nearby'], ['rel_placestoreat', 'rel_simpoi'], ['rel_placestoreat', 'rel_od'],
                             ['rel_placestoreat', 'rel_baserve_rev', 'rel_baserve'], ['rel_relatedbrand', 'rel_placestoreat'],
                             ['rel_brandof', 'rel_competitive', 'rel_locateat'], ['rel_brand2cat1', 'rel_1_catof_rev', 'rel_locateat'],
                             ['rel_brand2cat1', 'rel_brand2cat1_rev', 'rel_placestoreat']]
        self.device = kwargs['device']
        self.rel_path_id = [torch.tensor([self.d.rel2id[rel_i] for rel_i in rel_path_i], dtype=torch.long, device=self.device) for rel_path_i in self.rel_path_str]

        self.rnn_comps = nn.ModuleList()
        for i in range(len(self.rel_path_id)):
            self.rnn_comps.append(nn.GRU(input_size=edim, hidden_size=edim, batch_first=True))
        self.attn_paths = nn.MultiheadAttention(embed_dim=edim, num_heads=1,)
        self.comp_opn = kwargs['comp_opn']
        self.brand_cate_mlp = nn.Linear(2 * edim, edim)
        self.psa_alpha = kwargs['psa_alpha']

    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight)
        torch.nn.init.xavier_normal_(self.R.weight)

    def forward(self, g, h_idx, r_idx, t_idx, cat_idx):
        x, r = self.E.weight, self.R.weight
        for i in range(self.n_layer):
            x, r = self.layers[i](g, x, r)
            x = torch.tanh(x)
            r = torch.tanh(r)
        h_emb = torch.index_select(x, 0, h_idx) 
        t_emb = torch.index_select(x, 0, t_idx) 
        r_emb = torch.index_select(r, 0, r_idx)

        cate_emb = torch.index_select(x, 0, cat_idx)
        h_emb = torch.cat((h_emb, cate_emb), dim=1)
        h_emb = F.relu(self.brand_cate_mlp(h_emb)) 
        nb, edim, nr = h_emb.shape[0], h_emb.shape[1], r.shape[0]

        for i, rel_path_i in enumerate(self.rel_path_id):
            rel_path_vec = torch.index_select(r, 0, rel_path_i) 
            input_i = rel_path_vec.view(1, -1, edim).repeat(nb, 1, 1) 
            h_0 = h_emb.view(1, nb, edim)
            _, h_n = self.rnn_comps[i](input_i, h_0)  
            if i == 0:
                meta_path_rels = h_n
            else:
                meta_path_rels = torch.cat((meta_path_rels, h_n), dim=0)
       
        attn_output, attn_weight = self.attn_paths(query=h_emb.view(1, nb, edim), key=meta_path_rels, value=meta_path_rels)
    

        pred1 = attn_output.view(nb, edim) @ t_emb.transpose(1, 0)
        pred2 = (h_emb * r_emb) @ t_emb.transpose(1, 0)
        pred = (1-self.psa_alpha) * F.normalize(pred1, dim=1) + self.psa_alpha * F.normalize(pred2, dim=1)
        return pred, attn_weight

