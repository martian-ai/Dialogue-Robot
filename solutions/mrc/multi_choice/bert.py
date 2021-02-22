# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BERT(nn.Module):
    def __init__(self, bert, opt):
        super(BERT, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.softmax = nn.Softmax(dim=1)
        self.dense1 = nn.Linear(opt.bert_dim, 1)
        #self.dense2 = nn.Linear(opt.bert_dim, 4)

    def forward(self, inputs):
        a_indices, a_segments = inputs[0], inputs[1]
        b_indices, b_segments = inputs[2], inputs[3]
        c_indices, c_segments = inputs[4], inputs[5]
        d_indices, d_segments = inputs[6], inputs[7]

        _, a_pooled_output = self.bert(a_indices, token_type_ids=a_segments)
        _, b_pooled_output = self.bert(b_indices, token_type_ids=b_segments)
        _, c_pooled_output = self.bert(c_indices, token_type_ids=c_segments)
        _, d_pooled_output = self.bert(d_indices, token_type_ids=d_segments)

        #print('a', a_pooled_output)
        #print('b', b_pooled_output)
        #print('c', c_pooled_output)
        #print('d', d_pooled_output)

        a_ = self.dense1(self.dropout(a_pooled_output))
        b_ = self.dense1(self.dropout(b_pooled_output))
        c_ = self.dense1(self.dropout(c_pooled_output))
        d_ = self.dense1(self.dropout(d_pooled_output))

        cat = torch.cat((a_, b_, c_, d_), 1)
        softmax = self.softmax(cat)
        return softmax
