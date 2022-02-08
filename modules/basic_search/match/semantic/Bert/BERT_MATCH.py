# coding=utf-8
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import truncnorm
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from modules.alpha_learner.nn_with_bert.function.modeling import BertForSequenceClassification


class BERT_Match(nn.Module):
    def __init__(self, opt):
        """
        config : bert 参数
        opt : 分类模型等参数
        """
        super(BERT_Match, self).__init__()

        self.bert = BertForSequenceClassification.from_pretrained(opt.local_bert_dir, num_labels = 2)  # /bert_pretrain/
        self.device = opt.device
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.bert(input_ids = batch_seqs, attention_mask = batch_seq_masks, token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities