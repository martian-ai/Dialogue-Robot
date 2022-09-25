
from __future__ import absolute_import, division, print_function

import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

logger = logging.getLogger(__name__)


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def rnn_factory(rnn_type, **kwargs):
    """ rnn factory, Use pytorch version when available. """
    rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gumbel_softmax(logits, tau=1.0, hard=False, log_mode=True, dim=-1):

    while(True):
        gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        if log_mode:
            y_soft = gumbels.log_softmax(dim)
        else:
            y_soft = gumbels.softmax(dim)
        if torch.sum(torch.isnan(y_soft)).item() < 0.01:
            break

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def gumbel_soft2hard(log_logits, dim=-1):
    y_soft = log_logits.exp()
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(log_logits).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret


class GlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
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


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """

    def detach(self):
        """ Need to document this """
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        """ Need to document this """
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

    def map_batch_fn(self, fn):
        raise NotImplementedError()

################################################################


class RNNDecoder(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`models.NMTModel`.
    .. mermaid::
       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Memory_Bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I
    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [LSTM, GRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, copy_attn=False,
                 dropout=0.0, embeddings=None,
                 reuse_copy_attn=False):
        super(RNNDecoder, self).__init__()
        assert embeddings is not None
        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        input_size = self.embeddings.embedding_dim + self.hidden_size

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = GlobalAttention(
                hidden_size, attn_type=attn_type
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        if rnn_type == "LSTM":
            stacked_cell = StackedLSTM
        else:
            stacked_cell = StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = state.input_feed.squeeze(0)
        input_feed_batch, _ = input_feed.size()
        tgt_batch, _, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # Initialize local and return variables.
        decoder_outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self.training and self._coverage:
            attns["coverage"] = []

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for _, emb_t in enumerate(tgt.transpose(0, 1).split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)

            rnn_output, hidden = self.rnn(decoder_input, hidden)
            decoder_output, p_attn = self.attn(
                rnn_output,
                memory_bank,
                memory_lengths=memory_lengths)

            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            decoder_outputs += [decoder_output]
            attns["std"] += [p_attn]

            # Update the coverage attention.
            if self.training and self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output,
                                              memory_bank)
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
        # Return result.
        return hidden, decoder_outputs, attns

    def forward(self, tgt, memory_bank, state, memory_masks=None,
                step=None):
        # Check
        assert isinstance(state, RNNDecoderState)
        # tgt.size() returns tgt length and batch
        tgt_batch, _ = tgt.size()
        memory_batch, _, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)
        # END
        memory_lengths = memory_masks.sum(dim=1)
        emb = self.embeddings(tgt)
        # Run the forward pass of the RNN.
        decoder_final, decoder_outputs, attns = self._run_forward_pass(
            emb, memory_bank, state, memory_lengths=memory_lengths)

        # Update the state with the result.
        final_output = decoder_outputs[-1]
        coverage = None
        if "coverage" in attns:
            coverage = attns["coverage"][-1].unsqueeze(0)
        state.update_state(decoder_final, final_output.unsqueeze(0), coverage)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: decoder_outputs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(decoder_outputs) == list:
            decoder_outputs = torch.stack(decoder_outputs).transpose(0, 1)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k]).transpose(0, 1)

        return decoder_outputs, state, attns

    def init_decoder_state(self, src, memory_bank, encoder_final):
        """ Init decoder state with last state of the encoder """
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            return RNNDecoderState(self.hidden_size,
                                   tuple([_fix_enc_hidden(enc_hid)
                                          for enc_hid in encoder_final]))
        else:  # GRU
            return RNNDecoderState(self.hidden_size,
                                   _fix_enc_hidden(encoder_final))


class RNNDecoderState(DecoderState):
    """ Base class for RNN decoder state """

    def __init__(self, hidden_size, rnnstate):
        """
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = self.hidden[0].size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = self.hidden[0].data.new(*h_size).zero_() \
                              .unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        """ Update decoder state """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [e.data.repeat(1, beam_size, 1)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]

    def map_batch_fn(self, fn):
        self.hidden = tuple(map(lambda x: fn(x, 1), self.hidden))
        self.input_feed = fn(self.input_feed, 1)


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input_feed, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input_feed, (h_0[i], c_0[i]))
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input_feed, (h_1, c_1)


class StackedGRU(nn.Module):
    """
    Our own implementation of stacked GRU.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input_feed, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input_feed, hidden[0][i])
            input_feed = h_1_i
            if i + 1 != self.num_layers:
                input_feed = self.dropout(input_feed)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)
        return input_feed, (h_1,)


################################
# encoder


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):

        pe = torch.zeros(max_len, dim)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        position = torch.arange(0, max_len).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)

        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None, add_emb=None):
        emb = emb * math.sqrt(self.dim)
        if add_emb is not None:
            emb = emb + add_emb
        if (step):
            pos = self.pe[:, step][:, None, :]
            emb = emb + pos
        else:
            pos = self.pe[:, :emb.size(1)]
            emb = emb + pos
        emb = self.dropout(emb)
        return emb


class DistancePositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        mid_pos = max_len // 2
        # absolute position embedding
        ape = torch.zeros(max_len, dim // 2)
        # distance position embedding
        dpe = torch.zeros(max_len, dim // 2)

        ap = torch.arange(0, max_len).unsqueeze(1)
        dp = torch.abs(torch.arange(0, max_len).unsqueeze(1) - mid_pos)

        div_term = torch.exp((torch.arange(0, dim//2, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim * 2)))
        ape[:, 0::2] = torch.sin(ap.float() * div_term)
        ape[:, 1::2] = torch.cos(ap.float() * div_term)
        dpe[:, 0::2] = torch.sin(dp.float() * div_term)
        dpe[:, 1::2] = torch.cos(dp.float() * div_term)

        ape = ape.unsqueeze(0)
        super(DistancePositionalEncoding, self).__init__()
        self.register_buffer('ape', ape)
        self.register_buffer('dpe', dpe)
        self.dim = dim
        self.mid_pos = mid_pos

    def forward(self, emb, shift):
        device = emb.device
        _, length, _ = emb.size()
        pe_seg = [len(ex) for ex in shift]
        medium_pos = [torch.cat([torch.tensor([0], device=device),
                                 (ex[1:] + ex[:-1]) // 2 + 1,
                                 torch.tensor([length], device=device)], 0)
                      for ex in shift]
        shift = torch.cat(shift, 0)
        index = torch.arange(self.mid_pos, self.mid_pos + length, device=device).\
            unsqueeze(0).expand(len(shift), length) - shift.unsqueeze(1)
        index = torch.split(index, pe_seg)
        dp_index = []
        for i in range(len(index)):
            dpi = torch.zeros([length], device=device)
            for j in range(len(index[i])):
                dpi[medium_pos[i][j]:medium_pos[i][j+1]] = index[i][j][medium_pos[i][j]:medium_pos[i][j+1]]
            dp_index.append(dpi.unsqueeze(0))
        dp_index = torch.cat(dp_index, 0).long()

        dpe = self.dpe[dp_index]
        ape = self.ape[:, :emb.size(1)].expand(emb.size(0), emb.size(1), -1)
        pe = torch.cat([dpe, ape], -1)
        emb = emb + pe
        return emb


class RelativePositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        mid_pos = max_len // 2
        # relative position embedding
        pe = torch.zeros(max_len, dim)

        position = torch.arange(0, max_len).unsqueeze(1) - mid_pos

        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        super(RelativePositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dim = dim
        self.mid_pos = mid_pos

    def forward(self, emb, shift):
        device = emb.device
        bsz, length, _ = emb.size()
        index = torch.arange(self.mid_pos, self.mid_pos + emb.size(1), device=device).\
            unsqueeze(0).expand(bsz, length) - shift.unsqueeze(1)
        pe = self.pe[index]
        emb = emb + pe
        return emb

    def get_emb(self, emb, shift):
        device = emb.device
        index = torch.arange(self.mid_pos, self.mid_pos + emb.size(1), device=device).\
            unsqueeze(0).expand(emb.size(0), emb.size(1)) - shift.unsqueeze(1)
        return self.pe[index]


class RNNEncoder(nn.Module):
    """ A generic recurrent neural network encoder.
    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn = rnn_factory(rnn_type,
                               input_size=embeddings.embedding_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               dropout=dropout,
                               bidirectional=bidirectional,
                               batch_first=True)

    def forward(self, src, mask):

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()
        lengths = mask.sum(dim=1)

        # Lengths data is wrapped inside a Tensor.
        lengths_list = lengths.view(-1).tolist()
        packed_emb = pack(emb, lengths_list, batch_first=True, enforce_sorted=False)

        memory_bank, encoder_final = self.rnn(packed_emb)

        memory_bank = unpack(memory_bank, batch_first=True)[0]

        return memory_bank, encoder_final


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, type='self')
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(TransformerEncoder, self).__init__()
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        x = self.pos_emb(top_vecs)

        for i in range(self.num_inter_layers):
            x = self.transformer[i](i, x, mask)  # all_sents * max_tokens * dim

        output = self.layer_norm(x)

        return output


################################
# decoder


MAX_SIZE = 5000


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, topic=False, topic_dim=300, split_noise=False):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout, topic=topic, topic_dim=topic_dim, split_noise=split_noise)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, previous_input=None,
                layer_cache=None, topic_vec=None, requires_att=False):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`

        """
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                                      :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None

        query, _ = self.self_attn(all_input, all_input, input_norm,
                                  mask=dec_mask,
                                  layer_cache=layer_cache,
                                  type="self")

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, att = self.context_attn(memory_bank, memory_bank, query_norm,
                                     mask=src_pad_mask,
                                     layer_cache=layer_cache,
                                     type="context",
                                     topic_vec=topic_vec,
                                     requires_att=requires_att)
        mid = self.drop(mid) + query

        output = self.feed_forward(mid)

        return output, all_input, att
        # return output

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
       attn_type (str): if using a seperate copy attention
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings=None,
                 topic=False, topic_dim=300, split_noise=False):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = num_layers

        if embeddings is not None:
            self.embeddings = embeddings
            self.pos_emb = PositionalEncoding(dropout, self.embeddings.embedding_dim)

        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout,
                                     topic=topic, topic_dim=topic_dim, split_noise=split_noise)
             for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt, memory_bank, state, init_tokens=None,
                step=None, cache=None, memory_masks=None, tgt_masks=None,
                requires_att=False, topic_vec=None):

        if tgt.dim() == 2:
            tgt_batch, tgt_len = tgt.size()

            # Run the forward pass of the TransformerDecoder.
            emb = self.embeddings(tgt)
            if init_tokens is not None:
                emb = torch.cat([init_tokens.unsqueeze(1), emb[:, 1:, :]], 1)
            assert emb.dim() == 3  # len x batch x embedding_dim

            output = self.pos_emb(emb, step)
        else:
            tgt_batch, tgt_len, _ = tgt.size()
            output = tgt

        if tgt_masks is not None:
            tgt_pad_mask = tgt_masks.unsqueeze(1).expand(tgt_batch, tgt_len, tgt_len)
        else:
            assert tgt.dim() == 2
            padding_idx = self.embeddings.padding_idx
            tgt_pad_mask = tgt.data.eq(padding_idx).unsqueeze(1) \
                .expand(tgt_batch, tgt_len, tgt_len)

        src_memory_bank = memory_bank
        if memory_masks is not None:
            src_batch = memory_masks.size(0)
            src_len = memory_masks.size(-1)
            src_pad_mask = memory_masks.unsqueeze(1).expand(src_batch, tgt_len, src_len)
        else:
            src_batch = memory_bank.size(0)
            src_len = memory_bank.size(1)
            src_pad_mask = tgt_pad_mask.new_zeros([src_batch, tgt_len, src_len])

        if state.cache is None:
            saved_inputs = []

        for i in range(self.num_layers):
            prev_layer_input = None
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
            output, all_input, last_layer_att \
                = self.transformer_layers[i](
                    output, src_memory_bank,
                    src_pad_mask, tgt_pad_mask,
                    previous_input=prev_layer_input,
                    layer_cache=state.cache["layer_{}".format(i)]
                    if state.cache is not None else None,
                    topic_vec=topic_vec,
                    requires_att=False if i < self.num_layers-1 else requires_att)
            if state.cache is None:
                saved_inputs.append(all_input)

        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)

        output = self.layer_norm(output)

        # Process the result and update the attentions.

        if state.cache is None:
            state = state.update_state(tgt, saved_inputs)

        if requires_att and last_layer_att is not None:
            return output, state, {"copy": last_layer_att}
        else:
            return output, state, None

    def init_decoder_state(self, src, memory_bank, enc_hidden=None,
                           with_cache=False):
        """ Init decoder state """
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state


class TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if (self.previous_input is not None
                and self.previous_layer_inputs is not None):
            return (self.previous_input,
                    self.previous_layer_inputs,
                    self.src)
        else:
            return (self.src,)

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers):
        self.cache = {}

        for l in range(num_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None
            }
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.cache["layer_{}".format(l)] = layer_cache

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 0)
        if self.cache is not None:
            _recursive_map(self.cache)
