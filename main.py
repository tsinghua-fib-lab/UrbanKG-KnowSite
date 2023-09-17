from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import utils
import argparse
import mlflow
from mlflow.tracking import MlflowClient
import shutil
import os
from tqdm import tqdm
import json
import copy
import random
import dgl

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Experiment:

    def __init__(self, lr, edim, batch_size, dr):
        self.lr = lr
        self.edim = edim
        self.batch_size = batch_size
        self.dr = dr
        ordered_types = list(range(len(d.rel2id)))
        etype2eids = [(i, torch.where(k_graph.edata['etype'] == i)[0]) for i in ordered_types]
        if params['pretrain'] == 'true':
            print('Loading pretrained weights....')
            pretrain_emb = np.load('./pretrain_emb/ER_' + args.dataset + '_TuckER' + '_' + str(edim) + '.npz')
            params['E_pretrain'] = torch.from_numpy(pretrain_emb['E_pretrain']).to(device)
            params['R_pretrain'] = torch.from_numpy(pretrain_emb['R_pretrain']).to(device)
        params['etype2eids'] = etype2eids
        params['device'] = device
        self.kwargs = params

    def get_er_vocab(self, d):
        er_vocab_train, er_vocab_train_valid, er_vocab_train_valid_test = defaultdict(list), defaultdict(list), defaultdict(list)
        for x in d.train_data:
            er_vocab_train[x[0]].append(x[2])
            er_vocab_train_valid[x[0]].append(x[2])
            er_vocab_train_valid_test[x[0]].append(x[2])
        for k, x in d.valid_data.items():
            for y in x:
                if y[-1] > 0:
                    er_vocab_train_valid[k].append(y[2])
                    er_vocab_train_valid_test[k].append(y[2])
        for k, x in d.test_data.items():
            for y in x:
                if y[-1] > 0:
                    er_vocab_train_valid_test[k].append(y[2])
        return er_vocab_train, er_vocab_train_valid, er_vocab_train_valid_test

    def get_train_psa_batch(self, data, idx):
        batch = data[idx:idx + self.batch_size]
        brand_idx = torch.tensor([x[0] for x in batch], device=device)
        region_ont_idx = torch.tensor([x[2] for x in batch], device=device)
        cat_idx = torch.tensor([x[4] for x in batch], device=device)
        return brand_idx, region_ont_idx, cat_idx

    def evaluate(self, model, test_data, mode, K_group):
        ndcgs, hits, pres, recs, aps = [[] for _ in K_group], [[] for _ in K_group], [[] for _ in K_group], [[] for _ in K_group], [[] for _ in K_group]
        sample_hits = [[] for _ in K_group]
        if mode == 'valid':
            er_vocab_val, er_vocab_all = er_vocab_train, er_vocab_train_valid
        elif mode == 'test':
            er_vocab_val, er_vocab_all = er_vocab_train_valid, er_vocab_train_valid_test

        attn_weight_all = []
        t_idx = torch.tensor(d.region_list, dtype=torch.long, device=device)
        for test_brand_idx in tqdm(sorted(list(test_data.keys()), key=lambda x: int(x))):
            r_psa_idx = d.psa_id * torch.ones(1, dtype=torch.long, device=device)
            brand_idx = test_brand_idx * torch.ones(1, dtype=torch.long, device=device)
            cat_idx = test_data[test_brand_idx][0][4] * torch.ones(1, dtype=torch.long, device=device)
            y_pred, attn_weight_i = model.forward(k_graph, brand_idx, r_psa_idx, t_idx, cat_idx)

            attn_weight_all.append(attn_weight_i.cpu().numpy())

            y_pred_val = y_pred.clone()
            filt_val = er_vocab_val[test_brand_idx]
            y_pred_val[0, filt_val] = -1e10
            relat_score = np.array([x[-1] for x in test_data[test_brand_idx]])
            y_pred_val = y_pred_val.squeeze().cpu().numpy()
            for i, k in enumerate(K_group):
                ndcg_k = utils.get_ndcg(y_pred_val, relat_score, k)
                hit_k = utils.get_hit(y_pred_val, relat_score, k)
                pre_k, rec_k, ap_k = utils.get_pre_rec_ap(y_pred_val, relat_score, k)
                ndcgs[i].append(ndcg_k)
                hits[i].append(hit_k)
                pres[i].append(pre_k)
                recs[i].append(rec_k)
                aps[i].append(ap_k)
        print('Hits @10: %.6f' % (np.mean(hits[K_group.index(10)])))
        print('NDCG @10: %.6f' % (np.mean(ndcgs[K_group.index(10)])))
        print('Precision @10: %.6f' % (np.mean(pres[K_group.index(10)])))
        print('Recall @10: %.6f' % (np.mean(recs[K_group.index(10)])))
        print('MAP @10: %.6f' % (np.mean(aps[K_group.index(10)])))
        attn_weight_all = np.array(attn_weight_all)
        return [np.mean(ndcg_k) for ndcg_k in ndcgs], [np.mean(hr_k) for hr_k in hits], [np.mean(pre_k) for pre_k in pres],\
               [np.mean(rec_k) for rec_k in recs], [np.mean(ap_k) for ap_k in aps], attn_weight_all

    def train_and_eval(self):
        print("Number of KG edges: %d, Number of KG nodes: %d" % (k_graph.number_of_edges(), k_graph.number_of_nodes()))
        print('number of training data %d' % len(d.train_data))
        model = globals()[args.model_name](d, **self.kwargs)
        model = model.to(device)
        for name, x in model.named_parameters():
            print(name, x.size())
        mlflow.log_param('param_num', sum(p.numel() for p in model.parameters()))

        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        if self.dr:
            scheduler = ExponentialLR(opt, self.dr)

        best_valid_metric, best_valid_iter = 0.0, 0
        t_idx = torch.tensor(d.region_list, dtype=torch.long, device=device)
        print("Starting training...")
        for it in range(1, args.num_iterations + 1):
            print('\n=============== Epoch %d Starts...===============' % it)
            start_train = time.time()
            model.train()
            losses = []
            np.random.shuffle(d.train_data)
            for j in tqdm(range(0, len(d.train_data), self.batch_size)):
                opt.zero_grad()
                brand_idx, region_ont_idx, cat_idx = self.get_train_psa_batch(d.train_data, j)
                r_psa_idx = d.psa_id * torch.ones(len(brand_idx), dtype=torch.long, device=device)
                y_pred_psa, _ = model.forward(k_graph, brand_idx, r_psa_idx, t_idx, cat_idx)
                loss = model.psa_loss(y_pred_psa, region_ont_idx)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.dr:
                scheduler.step()
            print('\nEpoch=%d, train time cost %.4fs, loss:%.4f' % (it, time.time() - start_train, np.mean(losses)))
            mlflow.log_metrics({'train_time': time.time() - start_train, 'current_it': it, 'loss': np.mean(losses)}, step=it)

            model.eval()
            with torch.no_grad():
                print("Validation:")
                start_valid = time.time()
                ndcg_valid, hit_valid, pre_valid, rec_valid, map_valid, _ = self.evaluate(model, d.valid_data, 'valid', K_group)

                ndcg_valid_dict = dict([('valid_ndcg_' + str(x), ndcg_valid[i]) for i, x in enumerate(K_group)])
                hit_valid_dict = dict([('valid_hit_' + str(x), hit_valid[i]) for i, x in enumerate(K_group)])
                pre_valid_dict = dict([('valid_pre_' + str(x), pre_valid[i]) for i, x in enumerate(K_group)])
                rec_valid_dict = dict([('valid_rec_' + str(x), rec_valid[i]) for i, x in enumerate(K_group)])
                map_valid_dict = dict([('valid_map_' + str(x), map_valid[i]) for i, x in enumerate(K_group)])
                mlflow.log_metric(key='valid_time', value=time.time() - start_valid, step=it)
                mlflow.log_metrics(ndcg_valid_dict, step=it)
                mlflow.log_metrics(hit_valid_dict, step=it)
                mlflow.log_metrics(pre_valid_dict, step=it)
                mlflow.log_metrics(rec_valid_dict, step=it)
                mlflow.log_metrics(map_valid_dict, step=it)

                print("Test:")
                start_test = time.time()
                ndcg_test, hit_test, pre_test, rec_test, map_test, attn_weight_all = self.evaluate(model, d.test_data, 'test', K_group)
                ndcg_test_dict = dict([('test_ndcg_' + str(x), ndcg_test[i]) for i, x in enumerate(K_group)])
                hit_test_dict = dict([('test_hit_' + str(x), hit_test[i]) for i, x in enumerate(K_group)])
                pre_test_dict = dict([('test_pre_' + str(x), pre_test[i]) for i, x in enumerate(K_group)])
                rec_test_dict = dict([('test_rec_' + str(x), rec_test[i]) for i, x in enumerate(K_group)])
                map_test_dict = dict([('test_map_' + str(x), map_test[i]) for i, x in enumerate(K_group)])
                mlflow.log_metric(key='test_time', value=time.time() - start_test, step=it)
                mlflow.log_metrics(ndcg_test_dict, step=it)
                mlflow.log_metrics(hit_test_dict, step=it)
                mlflow.log_metrics(pre_test_dict, step=it)
                mlflow.log_metrics(rec_test_dict, step=it)
                mlflow.log_metrics(map_test_dict, step=it)

                if ndcg_valid[K_group.index(10)] > best_valid_metric:
                    best_valid_metric = ndcg_valid[K_group.index(10)]
                    best_valid_iter = it
                    best_test_hit10 = hit_test[K_group.index(10)]
                    best_test_ndcg10 = ndcg_test[K_group.index(10)]
                    best_test_pre10 = pre_test[K_group.index(10)]
                    best_test_rec10 = rec_test[K_group.index(10)]
                    best_test_map10 = map_test[K_group.index(10)]
                    print('valid NDCG@10 increases, Best Test Hit@10=%.4f, NDCG@10=%.4f' % (best_test_hit10, best_test_ndcg10))
                else:
                    if it - best_valid_iter >= args.patience:
                        print('\n\n=========== Final Results ===========')
                        print('Best Epoch: %d\nTest Hit@10: %.8f\nTest NDCG@10: %.8f\n' % (best_valid_iter, best_test_hit10, best_test_ndcg10))
                        break
                    else:
                        print('valid_NDCG@10 didn\'t increase for %d epochs, Best Iter=%d, Best Hit@10=%.4f, NDCG@10=%.4f, Best_Valid_Metric=%.4f,' %
                              (it - best_valid_iter, best_valid_iter, best_test_hit10, best_test_ndcg10, best_valid_metric))
                mlflow.log_metric(key='best_it', value=best_valid_iter, step=it)
                mlflow.log_metric(key='best_valid_ndcg10', value=best_valid_metric, step=it)
                mlflow.log_metric(key='best_test_hit10', value=best_test_hit10, step=it)
                mlflow.log_metric(key='best_test_ndcg10', value=best_test_ndcg10, step=it)
                mlflow.log_metric(key='best_test_pre10', value=best_test_pre10, step=it)
                mlflow.log_metric(key='best_test_rec10', value=best_test_rec10, step=it)
                mlflow.log_metric(key='best_test_map10', value=best_test_map10, step=it)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--prefix", type=str, default="test")
    parser.add_argument("--patience", type=int, default=10, help="valid patience.")
    parser.add_argument("--seed", type=int, default=67, help="random seed.")

    parser.add_argument("--dataset", type=str, default="beijing", help="dataset, beijing/shanghai.")
    parser.add_argument("--num_iterations", type=int, default=200, help="number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?", help="batch size.")
    parser.add_argument("--lr", type=float, default=0.001, nargs="?", help="learning rate.")
    parser.add_argument("--dr", type=float, default=0.995, nargs="?", help="decay rate.")
    parser.add_argument("--edim", type=int, default=64, nargs="?", help="entity embedding dimensionality.")
    parser.add_argument("--model_name", type=str, default="MetaPathDecoder", help="MetaPathDecoder and its variants.")
    parser.add_argument("--kg_name", type=str, default="kg", help='KG name for task.')
    parser.add_argument("--pretrain", type=str, default="true", help="whether to use pretrain embedding.")
    parser.add_argument("--freeze", type=str, default="true", help="whether to freeze parameters in training.")
    parser.add_argument("--opn", type=str, default="rotate", help="composition operator for CompGCN, rotate/sub/mult/corr.")
    parser.add_argument("--n_layer", type=int, default=2, help="number of GCN layers.")
    parser.add_argument("--gcn_dropout", type=float, default=0.3, help="dropout for CompGCN.")
    parser.add_argument("--comp_opn", type=str, default="rnn", help="composition operator for relational paths, rnn/add/mult")
    parser.add_argument("--psa_alpha", type=float, default=0.0, help="weight for PSA score")
    parser.add_argument("-rm_p", "--rm_path_idx", type=str, action='append', help="remove path idxs, 0-7")
    parser.add_argument("-add_p", "--add_path_idx", type=str, action='append', help="add path idxs, 0-7")

    args = parser.parse_args()
    print(args)
    K_group = [5, 10, 20]
    dataset = args.dataset
    data_dir = "./data/%s/" % dataset
    # ~~~~~~~~~~~~~~~~~~ mlflow experiment ~~~~~~~~~~~~~~~~~~~~~
    experiment_name = args.exp_name
    mlflow.set_tracking_uri('./mlflow_output')
    client = MlflowClient()
    try:
        EXP_ID = client.create_experiment(experiment_name)
        print('Initial Create!')
    except:
        experiments = client.get_experiment_by_name(experiment_name)
        EXP_ID = experiments.experiment_id
        print('Experiment Exists, Continuing')
    with mlflow.start_run(experiment_id=EXP_ID):
        archive_path = mlflow.get_artifact_uri()
        # ~~~~~~~~~~~~~~~~~ reproduce setting ~~~~~~~~~~~~~~~~~~~~~
        seed = args.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        dgl.random.seed(seed)
        print('Loading data....')
        d = Data(data_dir=data_dir, kg_name=args.kg_name)
        print('Preparing for DGL.graph....')
        k_graph = utils.load_dgl_graph(d, device)
        params = vars(args)
        mlflow.log_params(params)

        experiment = Experiment(batch_size=args.batch_size, lr=args.lr, dr=args.dr, edim=args.edim)
        er_vocab_train, er_vocab_train_valid, er_vocab_train_valid_test = experiment.get_er_vocab(d)
        experiment.train_and_eval()

