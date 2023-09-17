import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm


class Data:
    def __init__(self, data_dir, kg_name):
        self.ents, self.rels, self.ent2id, self.rel2id, self.kg_data = self.load_knowledge_graph(kg_dir=data_dir + kg_name + '.txt')
        self.num_ents, self.num_rels = len(self.ent2id), len(self.rel2id)
        print(self.rel2id)

        self.brand_list, self.region_list = [], []
        with open(data_dir + 'list_brands.txt', 'r') as f:
            for line in f.readlines():
                x = line.strip().split('\t')
                self.brand_list.append(self.ent2id[x[0]])
        with open(data_dir + 'list_regions.txt', 'r') as f:
            for line in f.readlines():
                x = line.strip()
                self.region_list.append(self.ent2id[x])
        self.brand_list = sorted(self.brand_list, key=lambda y: int(y))
        self.region_list = sorted(self.region_list, key=lambda y: int(y))
        self.psa_id, self.psa_rev_id = self.rel2id['rel_placestoreat'], self.rel2id['rel_placestoreat_rev']
        self.kg_id2region_id = dict([(kg_id, i) for i, kg_id in enumerate(self.region_list)])
        self.region_id2kg_id = dict([(i, kg_id) for i, kg_id in enumerate(self.region_list)])
        self.kg_id2brand_id = dict([(kg_id, i) for i, kg_id in enumerate(self.brand_list)])
        self.brand_id2kg_id = dict([(i, kg_id) for i, kg_id in enumerate(self.brand_list)])

        self.train_data, self.valid_data, self.test_data = self.load_store_data(data_dir)
        print('number of kept entities=%d, number of kept relations=%d, #brand=%d, #region=%d' % (self.num_ents, self.num_rels, len(self.brand_list), len(self.region_list)))

    def load_knowledge_graph(self, kg_dir):
        facts_str = []
        print('loading knowledge graph...')
        with open(kg_dir, 'r') as f:
            for line in tqdm(f.readlines()):
                x = line.strip().split('\t')
                facts_str.append([x[0], x[1], x[2]])
        all_rels = sorted(list(set([x[1] for x in facts_str])))
        ents = sorted(list(set([x[0] for x in facts_str] + [x[2] for x in facts_str])), key=lambda y: (y.split('_')[0], int(y.split('_')[1])))
        ent2id, rel2id = dict([(x, i) for i, x in enumerate(ents)]), dict([(x, i) for i, x in enumerate(all_rels)])
        kg_data = sorted([[ent2id[x[0]], rel2id[x[1]], ent2id[x[2]]] for x in facts_str], key=lambda y: y[1])
        return ents, all_rels, ent2id, rel2id, kg_data

    def load_store_data(self, data_dir):
        brand2cate = {}
        for x in self.kg_data:
            if x[1] == self.rel2id['rel_brand2cat1']:
                brand2cate[x[0]] = x[2]

        L = len(self.region_list)
        train_data = []
        print('loading training data..')
        with open(data_dir+'train.txt', 'r') as f:
            for line in tqdm(f.readlines()):
                x = line.strip().split('\t')
                brand = self.ent2id[x[0].split(':')[0]]
                cate1 = brand2cate[brand]
                for i, x_i in enumerate(x[1:]):
                    region_kg_id = self.ent2id[x_i.split(',')[0]]
                    region_ont_id = self.kg_id2region_id[region_kg_id]
                    checkin = float(x_i.split(',')[1])
                    train_data.append([brand, region_kg_id, region_ont_id, checkin, cate1])
        print('loading valid/test data...')
        valid_data, test_data = defaultdict(lambda: [i for i in range(L)]), defaultdict(lambda: [i for i in range(L)])
        for str0 in ['valid', 'test']:
            with open(data_dir+str0 +'.txt', 'r') as f:
                for line in tqdm(f.readlines()):
                    x = line.strip().split('\t')
                    brand = self.ent2id[x[0].split(':')[0]]
                    cate1 = brand2cate[brand]
                    for i, x_i in enumerate(x[1:]):
                        region_kg_id = self.ent2id[x_i.split(',')[0]]
                        region_ont_id = self.kg_id2region_id[region_kg_id]
                        checkin = float(x_i.split(',')[1])
                        if i == 0:
                            relat_score = (L - (i + 1) + 1) / L
                            rank = i + 1
                            ck_last = checkin
                        elif checkin == ck_last:
                            relat_score = (L-rank+1) / L
                        else:
                            ck_last = checkin
                            rank = i+1
                            relat_score = (L-rank+1) / L
                        eval(str0+'_data')[brand][region_ont_id] = [brand, region_kg_id, region_ont_id, checkin, cate1, relat_score]
                    for i, v in enumerate(eval(str0+'_data')[brand]):
                        if type(v) == int:
                            region_kg_id = self.region_id2kg_id[i]
                            eval(str0+'_data')[brand][i] = [brand, region_kg_id, i, 0.0, cate1, 0.0]
        return train_data, valid_data, test_data

