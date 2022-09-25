# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BERT(nn.Module):
    def __init__(self, bert, opt):
        super(BERT, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.softmax = nn.Softmax(dim=1)
        self.dense = nn.Linear(opt.bert_dim, 2)
        self.dense2 = nn.Linear(8, 4)

    def forward(self, inputs):
        a_indices, a_segments = inputs[0], inputs[1]
        b_indices, b_segments = inputs[2], inputs[3]
        c_indices, c_segments = inputs[4], inputs[5]
        d_indices, d_segments = inputs[6], inputs[7]

        #print('a_indices', a_indices)
        #print('b_indices', b_indices)

        _, a_pooled_output = self.bert(a_indices, token_type_ids=a_segments)
        _, b_pooled_output = self.bert(b_indices, token_type_ids=b_segments)
        _, c_pooled_output = self.bert(c_indices, token_type_ids=c_segments)
        _, d_pooled_output = self.bert(d_indices, token_type_ids=d_segments)

        a_ = self.softmax(self.dense(self.dropout(a_pooled_output)))
        b_ = self.softmax(self.dense(self.dropout(b_pooled_output)))
        c_ = self.softmax(self.dense(self.dropout(c_pooled_output)))
        d_ = self.softmax(self.dense(self.dropout(d_pooled_output)))

        #print('a_', a_)
        #print('b_', b_)

        cat = self.dense2(torch.cat((a_, b_, c_, d_), dim=-1))
        softmax = self.softmax(cat)
        return cat, softmax
