#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Martian.AI, All Rights Reserved.

modules alpha_nn(neural_network) modeling_attention

Authors: apollo2mars(apollo2mars@gmail.com)
"""

import torch
import torch.nn as nn
import math


class GlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. 
    It then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """

    def __init__(self, dim, coverage=False, attn_type="dot"):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type.")
        self.attn_type = attn_type

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None, coverage=None):
        """

        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, source_l_ = coverage.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank += self.linear_cover(cover).view_as(memory_bank)
            memory_bank = self.tanh(memory_bank)

        # compute attention scores, as in Luong et al.
        align = self.score(source, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(1 - mask, -float('inf'))

        # Softmax to normalize attention weights
        align_vectors = self.softmax(align.view(batch*target_l, source_l))
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, source], 2).view(batch*target_l, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, source_l_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            # Check output sizes
            target_l_, batch_, dim_ = attn_h.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            target_l_, batch_, source_l_ = align_vectors.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        return attn_h, align_vectors


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True,
                 topic=False, topic_dim=300, split_noise=False):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if (self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)
        self.use_topic = topic
        self.split_noise = split_noise
        if self.use_topic:
            self.linear_topic_keys = nn.Linear(model_dim,
                                               head_count * self.dim_per_head)
            self.linear_topic_vecs = nn.Linear(topic_dim,
                                               head_count * self.dim_per_head)
            self.linear_topic_w = nn.Linear(head_count * self.dim_per_head * 3, head_count)

    def forward(self, key, value, query, mask=None, layer_cache=None,
                type=None, topic_vec=None, requires_att=False):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        assert self.use_topic == (topic_vec is not None)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), \
                    self.linear_keys(query), \
                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        if self.use_topic:
                            topic_key = self.linear_topic_keys(key)
                            topic_key = shape(topic_key)
                            if self.split_noise:
                                topic_vec_summ = self.linear_topic_vecs(topic_vec[0])
                                topic_vec_noise = self.linear_topic_vecs(topic_vec[1])
                                topic_vec_summ = shape(topic_vec_summ)
                                topic_vec_noise = shape(topic_vec_noise)
                                topic_vec = (topic_vec_summ, topic_vec_noise)
                            else:
                                topic_vec = self.linear_topic_vecs(topic_vec)
                                topic_vec = shape(topic_vec)
                        key = self.linear_keys(key)
                        value = self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        if self.use_topic:
                            topic_key = layer_cache["memory_topic_keys"]
                            if self.split_noise:
                                topic_vec_summ = layer_cache["memory_topic_vecs_summ"]
                                topic_vec_noise = layer_cache["memory_topic_vecs_noise"]
                                topic_vec = (topic_vec_summ, topic_vec_noise)
                            else:
                                topic_vec = layer_cache["memory_topic_vecs"]
                        key = layer_cache["memory_keys"]
                        value = layer_cache["memory_values"]
                    if self.use_topic:
                        layer_cache["memory_topic_keys"] = topic_key
                        if self.split_noise:
                            layer_cache["memory_topic_vecs_summ"] = topic_vec_summ
                            layer_cache["memory_topic_vecs_noise"] = topic_vec_noise
                        else:
                            layer_cache["memory_topic_vecs"] = topic_vec
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    if self.use_topic:
                        topic_key = self.linear_topic_keys(key)
                        topic_key = shape(topic_key)
                        if self.split_noise:
                            topic_vec_summ = self.linear_topic_vecs(topic_vec[0])
                            topic_vec_noise = self.linear_topic_vecs(topic_vec[1])
                            topic_vec_summ = shape(topic_vec_summ)
                            topic_vec_noise = shape(topic_vec_noise)
                            topic_vec = (topic_vec_summ, topic_vec_noise)
                        else:
                            topic_vec = self.linear_topic_vecs(topic_vec)
                            topic_vec = shape(topic_vec)
                    key = self.linear_keys(key)
                    value = self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            if self.use_topic:
                topic_key = self.linear_topic_keys(key)
                topic_key = shape(topic_key)
                if self.split_noise:
                    topic_vec_summ = self.linear_topic_vecs(topic_vec[0])
                    topic_vec_noise = self.linear_topic_vecs(topic_vec[1])
                    topic_vec_summ = shape(topic_vec_summ)
                    topic_vec_noise = shape(topic_vec_noise)
                    topic_vec = (topic_vec_summ, topic_vec_noise)
                else:
                    topic_vec = self.linear_topic_vecs(topic_vec)
                    topic_vec = shape(topic_vec)
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if self.use_topic:
            if self.split_noise:
                topic_vec_summ = topic_vec[0] / math.sqrt(dim_per_head)
                topic_scores_summ = torch.matmul(topic_vec_summ.unsqueeze(3), topic_key.unsqueeze(4)).squeeze_(-1)
                topic_scores_summ = topic_scores_summ.transpose(2, 3).expand_as(scores)

                topic_vec_noise = topic_vec[1] / math.sqrt(dim_per_head)
                topic_scores_noise = torch.matmul(topic_vec_noise.unsqueeze(3), topic_key.unsqueeze(4)).squeeze_(-1)
                topic_scores_noise = topic_scores_noise.transpose(2, 3).expand_as(scores)

                topic_scores = topic_scores_summ - topic_scores_noise
            else:
                topic_vec = topic_vec / math.sqrt(dim_per_head)
                topic_scores = torch.matmul(topic_vec.unsqueeze(3), topic_key.unsqueeze(4)).squeeze_(-1)
                topic_scores = topic_scores.transpose(2, 3).expand_as(scores)

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)
            if self.use_topic:
                topic_scores = topic_scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.

        attn = self.softmax(scores)
        if self.use_topic:
            topic_attn = self.softmax(topic_scores)
            context_raw = torch.matmul(attn, value)
            context_topic = torch.matmul(topic_attn, value)
            p_vec = torch.cat([context_raw, context_topic, query], -1).transpose(1, 2)\
                .contiguous().view(batch_size, -1, head_count * dim_per_head * 3)
            topic_p = torch.sigmoid(self.linear_topic_w(p_vec).transpose(1, 2)).unsqueeze(-1)
            attn = topic_p * attn + (1-topic_p) * topic_attn
            """
            mean_key = torch.sum(topic_key.unsqueeze(2) * (1-mask).float().unsqueeze(-1), dim=3) /\
                torch.sum((1-mask).float(), dim=-1, keepdim=True)
            if self.split_noise:
                mean_topic = torch.sum(torch.cat(topic_vec, -1).unsqueeze(2) * (1-mask).float().unsqueeze(-1), dim=3) /\
                    torch.sum((1-mask).float(), dim=-1, keepdim=True)
                sigma_vec = torch.cat([query, mean_key, mean_topic], -1).transpose(1, 2)\
                    .contiguous().view(batch_size, -1, head_count * dim_per_head * 4)
            else:
                mean_topic = torch.sum(topic_vec.unsqueeze(2) * (1-mask).float().unsqueeze(-1), dim=3) /\
                    torch.sum((1-mask).float(), dim=-1, keepdim=True)
                sigma_vec = torch.cat([query, mean_key, mean_topic], -1).transpose(1, 2)\
                    .contiguous().view(batch_size, -1, head_count * dim_per_head * 3)
            sigma = torch.sigmoid(self.linear_topic_u(torch.tanh(self.linear_topic_w(sigma_vec)))).transpose(1, 2)
            topic_scores = -0.5 * ((1 - torch.sigmoid(topic_scores)) / sigma.unsqueeze(-1)).pow(2)
            scores = scores + topic_scores
            """

        if requires_att:
            required_att = attn.mean(1)
        else:
            required_att = None

        drop_attn = self.dropout(attn)
        if (self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output, required_att
        else:
            context = torch.matmul(drop_attn, value)
            return context, required_att
