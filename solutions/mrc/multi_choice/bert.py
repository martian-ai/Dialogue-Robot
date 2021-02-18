# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BERT(nn.Module):
    def __init__(self, bert, opt):
        super(BERT, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.softmax = nn.Softmax(dim=4)

    def forward(self, inputs):
        a_indices, a_segments = inputs[0], inputs[1]
        b_indices, b_segments = inputs[2], inputs[3]
        c_indices, c_segments = inputs[4], inputs[5]
        d_indices, d_segments = inputs[6], inputs[7]

        _, a_pooled_output = self.bert(a_indices, token_type_ids=a_segments)
        _, b_pooled_output = self.bert(b_indices, token_type_ids=b_segments)
        _, c_pooled_output = self.bert(c_indices, token_type_ids=c_segments)
        _, d_pooled_output = self.bert(d_indices, token_type_ids=d_segments)

        a_pooled_output = self.dropout(a_pooled_output)
        a_logits = self.dense(a_pooled_output)

        b_pooled_output = self.dropout(b_pooled_output)
        b_logits = self.dense(b_pooled_output)

        c_pooled_output = self.dropout(c_pooled_output)
        c_logits = self.dense(c_pooled_output)

        d_pooled_output = self.dropout(d_pooled_output)
        d_logits = self.dense(d_pooled_output)

        return self.softmax([a_logits, b_logits, c_logits, d_logits])