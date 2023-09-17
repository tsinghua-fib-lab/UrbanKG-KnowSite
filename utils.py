import numpy as np
import torch
import dgl
import torch.nn.functional as F
import dgl.function as fn


def get_ndcg(y_pred, relat_score, k):
    relat_score_sort = -np.sort(-relat_score)
    idcg = (2**relat_score_sort-1) / np.log2(np.arange(1, len(relat_score_sort)+1)+1)
    idcg = np.sum(idcg[0:k])
    y_pred_rank = np.argsort(-y_pred)
    dcg = (2**relat_score[y_pred_rank]-1) / np.log2(np.arange(1, len(y_pred_rank)+1)+1)
    dcg = np.sum(dcg[0:k])
    ndcg = dcg / idcg
    return ndcg


def get_hit(y_pred, relat_score, k):
    y_pred_rank = np.argsort(-y_pred)
    for i in y_pred_rank[0:k]:
        if relat_score[i] > 0.0:
            return 1.0
    return 0.0


def get_pre_rec_ap(y_pred, relat_score, k):
    y_pred_rank = np.argsort(-y_pred)
    y_true_rank = np.argsort(-relat_score)
    n_nonzero = len(relat_score.nonzero()[0])
    n = len(set(y_true_rank[0:n_nonzero]).intersection(y_pred_rank[0:k]))
    pre = n/k
    rec = n/min(n_nonzero, k)
    pre_k = 0
    for j in range(k):
        pre_k += len(set(y_true_rank[0:n_nonzero]).intersection(y_pred_rank[0:j+1])) / (j+1) * int(relat_score[y_pred_rank[j]]>0)
    ap = pre_k / min(n_nonzero, k)
    return pre, rec, ap


def load_dgl_graph(d, device):
    g = dgl.DGLGraph().to(device)
    g.add_nodes(d.num_ents)
    kg_data = torch.tensor(d.kg_data, dtype=torch.long, device=device)
    g.add_edges(kg_data[:, 0], kg_data[:, 2])
    g.edata['etype'] = kg_data[:, 1]
    in_norm = torch.pow(g.in_degrees().float().clamp(min=1), -0.5).view(-1, 1)
    g.ndata['in_norm'] = in_norm
    g.apply_edges(lambda edges: {'etype_id': F.one_hot(edges.data['etype'], kg_data[:, 1].max()+1).float()})
    g.update_all(fn.copy_edge('etype_id', 'm'), fn.sum(msg='m', out='etype_num'))
    g.apply_edges(lambda edges: {'etype_norm': torch.pow(edges.dst['etype_num'][range(g.number_of_edges()), edges.data['etype']], -1).to(device)})
    g.edata.pop('etype_id')
    g.ndata.pop('etype_num')

    g.edata['psa_id'] = torch.ones(len(kg_data), 1, dtype=torch.long, device=device) * 31
    return g
