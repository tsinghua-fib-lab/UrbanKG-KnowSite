import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn
import dgl.function as fn


class MyCompGCN(nn.Module):
    def __init__(self, indim, outdim, nr, dropout, opn, etype2eids):
        super(MyCompGCN, self).__init__()
        self.indim, self.outdim, self.opn, self.etype2eids = indim, outdim, opn, etype2eids
        self.W_R = nn.Parameter(torch.Tensor(nr, indim, outdim)) 
        self.W_rel = nn.Parameter(torch.Tensor(indim, outdim))  
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.Tensor(1, indim))
        self.init_params()
        self.bn = torch.nn.BatchNorm1d(outdim)

    def init_params(self):
        """Reinitialize learnable parameters."""
        init.xavier_normal_(self.W_R)
        init.xavier_normal_(self.W_rel)
        init.zeros_(self.bias)

    def message_func(self, edges):
        edge_data = self.comp(edges.src['h'], edges.data['eh']) 
        msg = edge_data[self.etype2eids[0][1]] @ self.W_R[self.etype2eids[0][0]]
        for t2e in self.etype2eids[1:]:
            msg = torch.cat([msg, edge_data[t2e[1]] @ self.W_R[t2e[0]]])
        msg *= edges.data['etype_norm'].view(-1, 1) 
     
        return {'msg': msg}

    def comp(self, h, edge_data):
        def com_mult(a, b):
            r1, i1 = a[..., 0], a[..., 1]
            r2, i2 = b[..., 0], b[..., 1]
            return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

        def conj(a):
            a[..., 1] = -a[..., 1]
            return a

        def ccorr(a, b):
            return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

        def rotate(h, r):
            d = h.shape[-1]
            h_re, h_im = torch.split(h, d // 2, -1)
            r_re, r_im = torch.split(r, d // 2, -1)
            return torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1)

        if self.opn == 'mult':
            return h * edge_data
        elif self.opn == 'sub':
            return h - edge_data
        elif self.opn == 'corr':
            return ccorr(h, edge_data.expand_as(h))
        elif self.opn == 'rotate':
            return rotate(h, edge_data.expand_as(h))
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')

    def forward(self, graph, node_feat, edge_feat):
        graph = graph.local_var()
        graph.ndata['h'] = node_feat
        graph.apply_edges(lambda edges: {'eh': edge_feat[edges.data['etype']]})
        graph.update_all(self.message_func, fn.sum(msg='msg', out='h'))

        x = self.dropout(graph.ndata.pop('h'))
        x = self.bn(x)
        return torch.tanh(x), torch.matmul(edge_feat, self.W_rel)
