# UrbanKG-KnowSite

 !["The Comparison of Data-driven Paradigm and Knowledge-driven Paradigm for Site Selection"](./fig_comparsion.png)

## Requirements

To install requirements:

```setup
python 3.7.4
pytorch 1.1
dgl 0.5.1
mlflow 1.12.1
```

## Running a model

To train (and evaluate) the model in the paper, run this command:

```
python main.py --dataset beijing --batch_size 128 --edim 64 --lr 0.001 --dr 0.995 --comp_opn rnn --dr 0.995 --freeze true --gcn_dropout 0.3 --kg_name kg_reverse --n_layer 2 --opn rotate --pretrain true --psa_alpha 0.5
python main.py --dataset shanghai --batch_size 128 --edim 64 --lr 0.003 --dr 0.995 --comp_opn rnn --dr 0.995 --freeze true --gcn_dropout 0.1 --kg_name kg_reverse --n_layer 2 --opn rotate --pretrain true --psa_alpha 0.8
```


## Reference

    @inproceddings{liu2023knowsite,
	title 	  = {KnowSite: Leveraging Urban Knowledge Graph for Site Selection},
	author	  = {Liu, Yu and Ding, Jingtao and Li, Yong},
	booktitle = {ACM SIGSPATIAL},
	year      = {2023}}
