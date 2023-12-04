#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Martian.AI, All Rights Reserved.

main application interface for Dialogue Robot BotMVP

Authors: apollo2mars(apollo2mars@gmail.com)
"""

import math
from math import sqrt
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

START_TAG = "<START>"
STOP_TAG = "<STOP>"


class MLP(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        super().__init__()

    def forward(self):
        pass


import torch
from torch.nn import Linear, ReLU, Sequential, Dropout, Softmax
import torch.nn.functional as F

class MLP(torch.nn.Module):
    """
    
    默认三层隐藏层，分别有128个 64个 16个神经元

    论文

    Args:
        torch (_type_): _description_
    """
    def __init__(self, input_n, output_n, num_layer=3, layer_list=[128, 64, 16], dropout=0.5):

        """
        :param input_n: int 输入神经元个数
        :param output_n: int 输出神经元个数
        :param num_layer: int 隐藏层层数
        :param layer_list: list(int) 每层隐藏层神经元个数
        :param dropout: float 训练完丢掉多少
        """
        super(MLP, self).__init__()
        self.input_n = input_n
        self.output_n = output_n
        self.num_layer = num_layer
        self.layer_list = layer_list

        # 输入层
        self.input_layer = Sequential(
            Linear(input_n, layer_list[0], bias=False),
            ReLU()
        )

        # 隐藏层
        self.hidden_layer = Sequential()

        for index in range(num_layer-1):
            self.hidden_layer.extend([Linear(layer_list[index], layer_list[index+1], bias=False), ReLU()])

        self.dropout = Dropout(dropout)

        # 输出层
        self.output_layer = Sequential(
            Linear(layer_list[-1], output_n, bias=False),
            Softmax(dim=1),
        )

    def forward(self, x):
        input = self.input_layer(x)
        hidden = self.hidden_layer(input)
        hidden = self.dropout(hidden)
        output = self.output_layer(hidden)
        return output


class DynamicLSTM(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 batch_first=True,
                 dropout=0,
                 bidirectional=False,
                 only_use_last_hidden_state=False,
                 rnn_type='LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bias=bias,
                               batch_first=batch_first,
                               dropout=dropout,
                               bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              bias=bias,
                              batch_first=batch_first,
                              dropout=dropout,
                              bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              bias=bias,
                              batch_first=batch_first,
                              dropout=dropout,
                              bidirectional=bidirectional)

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(
            x, x_len, batch_first=self.batch_first)

        # process using the selected RNN
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        """unsort: h"""
        ht = torch.transpose(
            ht, 0, 1
        )[x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(
                out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(
                    ct, 0, 1
                )[x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)


class AttentionMulitHead(nn.Module):

    def __init__(self,
                 embed_dim,
                 hidden_dim=None,
                 out_dim=None,
                 n_head=1,
                 score_function='dot_product',
                 dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(AttentionMulitHead, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len,
                                                      self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len,
                                                      self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx),
                           dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0),
                           dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class NoQueryAttention(AttentionMulitHead):
    '''q is a parameter'''

    def __init__(self,
                 embed_dim,
                 hidden_dim=None,
                 out_dim=None,
                 n_head=1,
                 score_function='dot_product',
                 q_len=1,
                 dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim, out_dim,
                                               n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(NoQueryAttention, self).forward(k, q)


class AttentionSelf(nn.Module):

    def __init__(self, input_dim, dim_k, dim_v):
        """
        self attetion 
        x(input) : batch_size * seq_len * input_dim
        超详细图解Self-Attention - 伟大是熬出来的的文章 - 知乎 https://zhuanlan.zhihu.com/p/410776234
        """
        super(AttentionSelf, self).__init__()
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        """
        q : batch_size * input_dim * dim_k
        k : batch_size * input_dim * dim_k
        v : batch_size * input_dim * dim_v
        """
        Q = self.q(x)  # Q: batch_size * seq_len * dim_k
        K = self.k(x)  # K: batch_size * seq_len * dim_k
        V = self.v(x)  # V: batch_size * seq_len * dim_v

        atten = nn.Softmax(dim=-1)(
            torch.bmm(Q, K.permute(0, 2, 1))
        ) * self._norm_fact  # Q * K.T() # batch_size * seq_len * seq_len
        output = torch.bmm(atten,
                           V)  # Q * K.T() * V # batch_size * seq_len * dim_v

        return output


class AttentionSeq2Seq(nn.Module):
    """
    Applies an attention mechanism on the query features from the decoder.

    math:
            \begin{array}{ll}
            x = context*query \\
            attn_scores = exp(x_i) / sum_j exp(x_j) \\
            attn_out = attn * context
            \end{array}

    Args:
        dim(int): The number of expected features in the query

    Inputs: query, context
        - **query** (batch, query_len, dimensions): tensor containing the query features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: query, attn
        - **query** (batch, query_len, dimensions): tensor containing the attended query features from the decoder.
        - **attn** (batch, query_len, input_len): tensor containing attention weights.

    Attributes:
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    """

    def __init__(self):
        super(AttentionSeq2Seq, self).__init__()
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    """
        - query   (batch, query_len, dimensions): tensor containing the query features from the decoder.
        - context (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    """

    def forward(self, query, context):
        batch_size = query.size(0)
        dim = query.size(2)
        in_len = context.size(1)
        # (batch, query_len, dim) * (batch, in_len, dim) -> (batch, query_len, in_len)
        attn = torch.bmm(query, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn_scores = F.softmax(attn.view(-1, in_len),
                                dim=1).view(batch_size, -1, in_len)

        # (batch, query_len, in_len) * (batch, in_len, dim) -> (batch, query_len, dim)
        attn_out = torch.bmm(attn_scores, context)

        return attn_out, attn_scores


class Seq2Seq(nn.Module):

    def __init__(self):
        super().__init__()


class BiLSTMCRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTMCRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim // 2,
                            num_layers=1,
                            bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(
                    1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([
            torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags
        ])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


class TextCNN(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 output_size,
                 kernel_dim=100,
                 kernel_sizes=(2, 3, 4),
                 dropout=0.5):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # print(self.embedding)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes
        ])
        '''
        上面是个for循环，不好理解写成下面也是没问题的。
        self.conv13 = nn.Conv2d(Ci, Co, (2, D))
        self.conv14 = nn.Conv2d(Ci, Co, (3, D))
        self.conv15 = nn.Conv2d(Ci, Co, (4, D))
        '''

        # kernal_size = (K,D)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)


#     def init_weights(self, pretrained_word_vectors, is_static=False):
#         self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
#         if is_static:#这里不使用预训练的词向量
#             self.embedding.weight.requires_grad = False

    def forward(self, inputs, is_training=False):
        inputs = self.embedding(inputs).unsqueeze(1)  # (B,1,T,D)
        # print(inputs.shape)
        inputs = [F.relu(conv(inputs)).squeeze(3)
                  for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        # print(inputs[0].shape)

        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2)
                  for i in inputs]  # [(N,Co), ...]*len(Ks)
        '''
        最大池化也可以拆分理解
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        # print(len(inputs))
        concated = torch.cat(inputs, 1)
        # print(concated.shape)
        if is_training:
            concated = self.dropout(concated)  # (N,len(Ks)*Co)
        out = self.fc(concated)
        # print(out.shape)
        return F.log_softmax(out, 1)
