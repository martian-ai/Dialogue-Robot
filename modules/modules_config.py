import argparse
import os
import random
import string

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from modules.alpha_nn.modeling_bert import BERT_CLF, BERT_Match
from modules.utils.optimizer.bert import AdamW

# Hyper Parameters
parser = argparse.ArgumentParser(description='Arguments for Bot')

# model
parser.add_argument('--task_tpye', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--bert_type', type=str)

# dataset
parser.add_argument('--dataset', default='sentiment-3', type=str, help='sentiment-3, domain-all')
parser.add_argument('--valset_ratio', default=0.1, type=float, help='set ratio between 0 and 1 for validation support')

# process
parser.add_argument('--max_seq_len', default=64, type=int)

# optimizer
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
parser.add_argument('--max_grad_norm', default=10.0, type=float, help='max grad norm')
parser.add_argument('--l2reg', default=0.05, type=float) # TODO

# initializer
parser.add_argument('--initializer', default='xavier_uniform_', type=str)

# bert
parser.add_argument('--bert_dim', default=768, type=int)
parser.add_argument('--dropout', default=0, type=float)

# train
parser.add_argument('--train_type', type=str, help='try 16, 32, 64 for BERT models')
parser.add_argument('--batch_size', type=int, help='try 16, 32, 64 for BERT models')
parser.add_argument('--epochs', type=int, help='try larger number for non-BERT models')
parser.add_argument('--patience', type=int)
# parser.add_argument('--hops', default=3, type=int)

# clf
parser.add_argument('--clf_label_dim', default=21, type=int)  # network
parser.add_argument('--clf_embed_dim', default=300, type=int)
parser.add_argument('--clf_hidden_dim', default=300, type=int)

# device
parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
parser.add_argument("--local_rank", type=int)


model_params = {
    "task_type": "",
    "target_dir":"",
    "locla_dir":""

}

task_type = {
    'clf':'',
    'match':''
}

model_name = ['bert-match', ]


model_classes = {
    'bert-clf': BERT_CLF,
    'bert-match': BERT_Match,
}

input_colses = {
    'bert-clf': ['indices'],
    'bert-match': ['indices_merge', 'segment', 'mask'],
}

dataset_files = {
    'demo-sentiment-3': {
        'train': '../resources/corpus/demo/clf.txt',
        'val': '../resources/corpus/demo/clf.txt',
        'test': '../resources/corpus/demo/clf.txt'
    },
    'domain-all': {
        'train': '../resources/corpus/domain-slot/iot/train.txt',
        'val': '../resources/corpus/domain-slot/iot/dev.txt',
        'test': '../resources/corpus/domain-slot/iot/test.txt'
    },
    'demo-match': {
        'train': '../resources/corpus/demo/match.txt',
        'val': '../resources/corpus/demo/match.txt',
        'test': '../resources/corpus/demo/match.txt'
    },
    'match-car-2': {
        'train': '../resources/corpus/match/sup_car_ds/train.tsv',
        'val': '../resources/corpus/match/sup_car_ds/dev.tsv',
        'test': '../resources/corpus/match/sup_car_ds/test.tsv'
    },
    'match-lcmqc': {
        'train': '../resources/corpus/match/sup_car_ds/train.tsv',
        'val': '../resources/corpus/match/sup_car_ds/dev.tsv',
        'test': '../resources/corpus/match/sup_car_ds/test.tsv'
    }
}

label_dict = {
    'demo-sentiment-3': {'positive': 0, 'neutral': 1, 'negative': 2},
    'sentiment-3': {'positive': 0, 'neutral': 1, 'negative': 2},
    'domain-all': {'alerts': 0, 'baike': 1, 'calculator': 2, 'call': 3, 'car_limit': 4, 'chat': 5, 'cook_book': 6, 'fm': 7, 'general_command': 8, 'home_command': 9, 'map': 10, 'master_command': 11, 'music': 12, 'news': 13, 'shopping': 14, 'stock': 15, 'time': 16, 'translator': 17, 'travel': 18, 'video': 19, 'weather': 20},
    'demo-match': {'0': 0, '1': 1},
    'match-car-2': {'0': 0, '1': 1},
}

initializers = {
    'xavier_uniform_': torch.nn.init.xavier_uniform_,
    'xavier_normal_': torch.nn.init.xavier_normal_,
    'orthogonal_': torch.nn.init.orthogonal_,
}

optimizers = {
    'adadelta': torch.optim.Adadelta,  # default lr=1.0
    'adagrad': torch.optim.Adagrad,  # default lr=0.01
    'adam': torch.optim.Adam,  # default lr=0.001
    'adamax': torch.optim.Adamax,  # default lr=0.002
    'asgd': torch.optim.ASGD,  # default lr=0.01
    'rmsprop': torch.optim.RMSprop,  # default lr=0.01
    'sgd': torch.optim.SGD, 
    'adamw': AdamW # default lr 2e-5
}

local_bert_dirs = {
    'bert-base-chinese' : '../resources/embedding/bert-base-chinese'
}

local_bert_config_files = {
    'bert-base-chinese' : '../resources/embedding/bert-base-chinese/bert_config.json'
}


opt.model_class = model_classes[opt.model_name]

opt.local_bert_dir = local_bert_dirs[opt.bert_type]
opt.local_bert_config_file = local_bert_config_files[opt.bert_type]

opt.dataset_file = dataset_files[opt.dataset]
opt.label_dict = label_dict[opt.dataset]

opt.inputs_cols = input_colses[opt.model_name]
opt.initializer = initializers[opt.initializer]
opt.optimizer = optimizers[opt.optimizer]

opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)
