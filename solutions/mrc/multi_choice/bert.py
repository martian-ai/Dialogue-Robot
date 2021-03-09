# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BERT(nn.Module):
    def __init__(self, bert, opt):
        super(BERT, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.softmax = nn.Softmax(dim=1)
        self.softmax1 = nn.Softmax(dim=1)
        self.dense1 = nn.Linear(opt.bert_dim, 2)
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

        a_ = self.softmax1(self.dense1(self.dropout(a_pooled_output)))
        b_ = self.softmax1(self.dense1(self.dropout(b_pooled_output)))
        c_ = self.softmax1(self.dense1(self.dropout(c_pooled_output)))
        d_ = self.softmax1(self.dense1(self.dropout(d_pooled_output)))

        #print('a', a_)
        #print('b', b_)
        #print('c', c_)
        #print('d', d_)
        
        indices = torch.LongTensor([0]).cuda()
        index_select_a = torch.index_select(a_, 1, indices).cuda()
        #print('select a ', index_select_a)
        index_select_b = torch.index_select(b_, 1, indices).cuda()
        #print('select b ', index_select_b)
        index_select_c = torch.index_select(c_, 1, indices).cuda()
        #print('select c ', index_select_c)
        index_select_d = torch.index_select(d_, 1, indices).cuda()
        #print('select d ', index_select_d)

        #cat = torch.cat((a_, b_, c_, d_), 1)
        cat = torch.cat((index_select_a, index_select_b, index_select_c, index_select_d), 1)
        #print('cat', cat)
        softmax = self.softmax(cat)
        #print('softmax', softmax)
        return cat, softmax
