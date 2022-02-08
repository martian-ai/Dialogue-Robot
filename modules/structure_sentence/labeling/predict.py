import argparse
import torch
import sys
import torch.nn as nn

from torch.utils.data import DataLoader
from transformers import BertModel  # TODO
sys.path.append('../..')
from args import opt
from modules.utils_tokenizer.bert import Tokenizer4Bert
from modules.utils_dataset.NERDataset import NERDatasetSingleLine

class NERPredictService():
    def __init__(self, opt, best_model_path):
        self.opt = opt
        self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)  # 此处不是最终的model
        self.model = nn.DataParallel(opt.model_class(bert, opt)).to(opt.device)  # 加载设置特定参数的模型
        self.model.load_state_dict(torch.load(best_model_path))  # TODO

    def predict(self, query):
        predict_data = NERDatasetSingleLine(query, self.tokenizer)
        train_data_loader = DataLoader(dataset=predict_data, batch_size=self.opt.batch_size, shuffle=True)
        self.model.eval()
        with torch.no_grad():
            for _, t_batch in enumerate(train_data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_outputs = self.model(t_inputs)
        return t_outputs


if __name__ == "__main__":

    ner_api = NERPredictService(opt, 'state_dict/bert_sentiment-3_val_acc_0.5556')
    print('最终结果', ner_api.predict('江泽民在中共中央担任总书记, 出生在江苏南京，在莫斯科大学学习生活了四年'))
    print('最终结果', ner_api.predict('新疆人李志华在中共中央担任总书记, 出生在江苏南京，在莫斯科大学学习生活了四年'))
    print('最终结果', ner_api.predict('西城区区委书记张三年多大'))
