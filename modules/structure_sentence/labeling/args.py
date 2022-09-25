import argparse
import os
import random
import numpy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from learner.nn_with_bert.function.modeling import  BERT_CRF

# Hyper Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='bert-base-chinese', type=str)
parser.add_argument('--dataset', default='sentiment-3', type=str, help='sentiment-3, domain-all')
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--initializer', default='xavier_uniform_', type=str)
parser.add_argument('--lr', default=5e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--l2reg', default=0.01, type=float)
parser.add_argument('--num_epoch', default=1, type=int, help='try larger number for non-BERT models')
parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
parser.add_argument('--log_step', default=200, type=int)
parser.add_argument('--embed_dim', default=300, type=int)
parser.add_argument('--hidden_dim', default=300, type=int)
parser.add_argument('--bert_dim', default=768, type=int)
parser.add_argument('--pretrained_bert_name', default='bert-base-chinese', type=str)  # 其他embedding
# parser.add_argument('--pretrained_bert_name', default='roberta-base', type=str) # 其他embedding
parser.add_argument('--max_seq_len', default=64, type=int)
# parser.add_argument('--clf_dim', default=21, type=int)
parser.add_argument('--hops', default=3, type=int)
parser.add_argument('--patience', default=20, type=int)
parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
parser.add_argument('--valset_ratio', default=0.1, type=float, help='set ratio between 0 and 1 for validation support')
# The following parameters are only valid for the lcf-bert model
parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')
opt = parser.parse_args()

if opt.seed is not None:
    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(opt.seed)

model_classes = {
    'bert-base-chinese': BERT_CRF,
}
# dataset_files = {
#     'sentiment-3': {
#         'train': '../../resources/corpus/clf/demo.txt',
#         'val': '../../resources/corpus/clf/demo.txt',
#         'test': '../../resources/corpus/clf/demo.txt'
#     },
#     'domain-all': {
#         'train': '../../resources/corpus/domain-slot/iot/train.txt',
#         'val': '../../resources/corpus/domain-slot/iot/dev.txt',
#         'test': '../../resources/corpus/domain-slot/iot/test.txt'
#     },
}
input_colses = {
    'bert-base-chinese': ['indices'],
}

# label_dict = {
#     'sentiment-3': {'positive': 0, 'neutral': 1, 'negative': 2},
#     'domain-all': {'alerts':0, 'baike':1, 'calculator':2,'call':3, 'car_limit':4,'chat':5, 'cook_book':6,'fm':7, \
#                     'general_command':8,'home_command':9, 'map':10,'master_command':11, 'music':12,'news':13, 'shopping':14, \
#                     'stock':15, 'time':16, 'translator':17, 'travel':18, 'video':19, 'weather':20}
#     }               

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
    'sgd': torch.optim.SGD
}
opt.model_class = model_classes[opt.model_name]
opt.dataset_file = dataset_files[opt.dataset]
opt.inputs_cols = input_colses[opt.model_name]
opt.initializer = initializers[opt.initializer]
opt.optimizer = optimizers[opt.optimizer]
opt.label_dict = label_dict[opt.dataset]
opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)
