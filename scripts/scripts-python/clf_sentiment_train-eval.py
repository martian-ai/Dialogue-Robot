import logging
import os
import sys
from time import localtime, strftime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.append('../../..') 

from modules.alpha_nn.modeling_bert import BERT_CLF, BertConfig
from modules.dataset_bert import CLFDataset
from modules.utils.tokenizer.tokenization import Tokenizer4Bert

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

max_seq_len = 32
tokenizer_type = 'bert-base-chinese'
bert_file = '../../../resources/embedding/bert-base-chinese'
config_file = '../../../resources/embedding/bert-base-chinese/bert_config.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = Tokenizer4Bert(bert_file ,max_seq_len, True)
batch_size = 64
trainset = '../../../resources/dataset/classification/sentiment-3/test.tsv'
testset = '../../../resources/dataset/classification/sentiment-3/test.tsv' 
valset = '../../../resources/dataset/classification/sentiment-3/test.tsv'
label_dict = {'positive': 0, 'neutral': 1, 'negative': 2}


trainset = CLFDataset(trainset, tokenizer, label_dict=label_dict, max_seq_len=max_seq_len)
testset = CLFDataset(testset, tokenizer, label_dict=label_dict, max_seq_len=max_seq_len)
valset = CLFDataset(valset, tokenizer, label_dict=label_dict, max_seq_len=max_seq_len)
train_data_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, drop_last=True)
test_data_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, drop_last=True)
val_data_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, drop_last=True)

class OPT:
    def __init__(self):
        self.dropout = 0.1
        self.bert_dim = 768
        self.clf_dim = 3

opt = OPT()

config = BertConfig(vocab_size_or_config_json_file=config_file)
model = BERT_CLF(config, opt).to(device)
_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(_params, lr=5e-5, weight_decay=0.01)

criterion = nn.CrossEntropyLoss()

num_epoch = 100

def _train(criterion, optimizer, train_data_loader, val_data_loader):
    max_val_acc = 0
    max_val_f1 = 0
    max_val_epoch = 0
    global_step = 0
    path = None
    for i_epoch in range(num_epoch):
        logger.info('>' * 100)
        n_correct, n_total, loss_total = 0, 0, 0
        # switch model to training mode
        model.train()
        for i_batch, batch in enumerate(train_data_loader):
            global_step += 1
            # clear gradient accumulators
            optimizer.zero_grad()
            indice = batch["indice"].to(device)
            mask = batch["mask"].to(device)
            segment = batch["segment"].to(device)
            outputs = model(indice, mask, segment)
            targets = batch['label'].to(device)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_total += len(outputs)
            loss_total += loss.item() * len(outputs)
            # if global_step % 10 == 0:
            #     train_acc = n_correct / n_total
            #     train_loss = loss_total / n_total
            #     logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

        val_acc = _evaluate_clf_acc(val_data_loader)
        logger.info('> val_acc: {:.4f}'.format(val_acc))
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            max_val_epoch = i_epoch
            if not os.path.exists('state_dict'):  # TODO sava path 需要制定
                os.mkdir('state_dict')
            path = 'state_dict/{0}_{1}_val_acc_{2}'.format("BERT-CLF", "zhijian", round(val_acc, 4))
            torch.save(model.state_dict(), path)
            logger.info('>> saved: {}'.format(path))
        if i_epoch - max_val_epoch >= 3:
            print('>> early stop.')
            break

    return path

def _evaluate_clf_acc(data_loader):
    n_correct, n_total = 0, 0
    t_targets_all, t_outputs_all = None, None
    model.eval() # switch model to evaluation mode
    with torch.no_grad():
        for i_batch, t_batch in enumerate(data_loader):
            indice = t_batch["indice"].to(device)
            mask = t_batch["mask"].to(device)
            segment = t_batch["segment"].to(device)
            t_outputs = model(indice, mask, segment)
            t_targets = t_batch['label'].to(device)
            n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
            n_total += len(t_outputs)
            if t_targets_all is None:
                t_targets_all = t_targets
                t_outputs_all = t_outputs
            else:
                t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
    acc = n_correct / n_total
    return acc

print(torch.cuda.is_available())
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

best_model_path = _train(criterion, optimizer, train_data_loader, val_data_loader)