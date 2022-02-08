# -*- coding: utf-8 -*-
"""
basic clf model funciton

use modules/alpha_nn/ modeling_*.py 
"""
import torch
from torch import nn
from transformers import BertForSequenceClassification, BertConfig

class BertModel(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, bert_model_path, num_labels):
        """_summary_

        Args:
            bert_model_path (_type_): _description_
            num_labels (_type_): _description_
        """
        super(BertModel, self).__init__()
        print(bert_model_path)
        print(num_labels)
        #self.bert = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels = 2)  # /bert_pretrain/
        self.bert = BertForSequenceClassification.from_pretrained(bert_model_path, num_labels=num_labels)

        self.device = torch.device("cuda")
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        """_summary_

        Args:
            batch_seqs (_type_): _description_
            batch_seq_masks (_type_): _description_
            batch_seq_segments (_type_): _description_
            labels (_type_): _description_

        Returns:
            _type_: _description_
        """
        loss, logits = self.bert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
    
    
class BertModelTest(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, bert_model_path):
        """_summary_

        Args:
            bert_model_path (_type_): _description_
        """
        super(BertModelTest, self).__init__()
        config = BertConfig.from_pretrained(bert_model_path)
        self.bert = BertForSequenceClassification(config) 
        self.device = torch.device("cuda")

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        """_summary_

        Args:
            batch_seqs (_type_): _description_
            batch_seq_masks (_type_): _description_
            batch_seq_segments (_type_): _description_
            labels (_type_): _description_

        Returns:
            _type_: _description_
        """
        loss, logits = self.bert(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities

# "../../resources/model/roberta_joint_sup_training", num_labels = 7
