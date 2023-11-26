import torch
import torch.nn as nn

import sys

sys.path.append("../..")

from modules.alpha_nn.modeling_transformer import (RNNDecoder, RNNEncoder,
                                                   TransformerDecoder,
                                                   TransformerEncoder)
from utils import aeq, gumbel_softmax, rnn_factory

class GenerateRNN(nn.Module):
    """RNN model for Generate.

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, args, device, vocab) -> None:
        """Generate RNN init function."""
        super().__init__()

        self.args = args
        self.device = device
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.beam_size = args.beam_size
        # self.max_length = args.max_length
        # self.min_length = args.min_length
        self.max_length = args.max_seq_len
        self.min_length = 1

        # special tokens
        # self.start_token = vocab['[unused1]']
        # self.end_token = vocab['[unused2]']
        self.pad_token = vocab['[PAD]']
        self.mask_token = vocab['[MASK]']
        self.seg_token = vocab['[SEP]']
        self.cls_token = vocab['[CLS]']
        # self.agent_token = vocab['[unused3]']
        # self.customer_token = vocab['[unused4]']
        self.hidden_size = 100
        #self.hidden_size = args.enc_hidden_size
        self.embeddings = nn.Embedding(self.vocab_size,
                                       self.hidden_size,
                                       padding_idx=0)
        tgt_embeddings = nn.Embedding(self.vocab_size,
                                      self.hidden_size,
                                      padding_idx=0)

        self.encoder = RNNEncoder('LSTM',
                                  bidirectional=True,
                                  num_layers=args.enc_layers,
                                  hidden_size=self.hidden_size,
                                  dropout=args.enc_dropout,
                                  embeddings=self.embeddings)
        self.decoder = RNNDecoder("LSTM",
                                  True,
                                  args.dec_layers,
                                  args.dec_hidden_size,
                                  dropout=args.dec_dropout,
                                  embeddings=tgt_embeddings,
                                  coverage_attn=args.coverage,
                                  copy_attn=args.copy_attn)

    def _fast_translate_batch(self,
                              batch,
                              memory_bank,
                              max_length,
                              memory_mask=None,
                              min_length=2,
                              beam_size=3,
                              hidden_state=None,
                              copy_attn=False):
        """_summary_

        Args:
            batch (_type_): _description_
            memory_bank (_type_): _description_
            max_length (_type_): _description_
            memory_mask (_type_, optional): _description_. Defaults to None.
            min_length (int, optional): _description_. Defaults to 2.
            beam_size (int, optional): _description_. Defaults to 3.
            hidden_state (_type_, optional): _description_. Defaults to None.
            copy_attn (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        batch_size = memory_bank.size(0)

        if self.args.decoder == "rnn":
            dec_states = self.decoder.init_decoder_state(batch.src, memory_bank, hidden_state)
        else:
            dec_states = self.decoder.init_decoder_state(batch.src, memory_bank, with_cache=True)

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn( lambda state, dim: torch.tile(state, beam_size, dim=dim))
        memory_bank = torch.tile(memory_bank, beam_size, dim=0)
        memory_mask = torch.tile(memory_mask, beam_size, dim=0)
        if copy_attn:
            src_map = torch.tile(batch.src_map, beam_size, dim=0)
        else:
            src_map = None

        batch_offset = torch.arange(batch_size,
                                    dtype=torch.long,
                                    device=self.device)
        beam_offset = torch.arange(0,
                                   batch_size * beam_size,
                                   step=beam_size,
                                   dtype=torch.long,
                                   device=self.device)

        alive_seq = torch.full([batch_size * beam_size, 1],
                               self.start_token,
                               dtype=torch.long,
                               device=self.device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (torch.tensor([0.0] + [float("-inf")] *
                                       (beam_size - 1),
                                       device=self.device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = [[] for _ in range(batch_size)]  # noqa: F812

        for step in range(max_length):
            # Decoder forward.
            decoder_input = alive_seq[:, -1].view(1, -1)
            decoder_input = decoder_input.transpose(0, 1)

            if self.args.decoder == "rnn":
                dec_out, dec_states, attn = self.decoder(
                    decoder_input,
                    memory_bank,
                    dec_states,
                    step=step,
                    memory_masks=memory_mask)
            else:
                dec_out, dec_states, attn = self.decoder(
                    decoder_input,
                    memory_bank,
                    dec_states,
                    step=step,
                    memory_masks=memory_mask,
                    requires_att=copy_attn)

            # Generator forward.
            if copy_attn:
                probs = self.generator(
                    dec_out.transpose(0, 1).squeeze(0),
                    attn['copy'].transpose(0, 1).squeeze(0), src_map)
                probs = collapse_copy_scores(
                    probs.unsqueeze(1), batch, self.vocab,
                    torch.tile(batch_offset, beam_size, dim=0))
                log_probs = probs.squeeze(1)[:, :self.vocab_size].log()
            else:
                log_probs = self.generator(dec_out.transpose(0, 1).squeeze(0))

            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            if self.args.block_trigram:
                cur_len = alive_seq.size(1)
                if (cur_len > 3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        if (len(words) <= 3):
                            continue
                        trigrams = [(words[i - 1], words[i], words[i + 1])
                                    for i in range(1,
                                                   len(words) - 1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            log_probs[i] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.args.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0)**alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (topk_beam_index +
                           beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat([
                alive_seq.index_select(0, select_indices),
                topk_ids.view(-1, 1)
            ], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append(
                            (topk_scores[i, j], predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(hypotheses[b],
                                          key=lambda x: x[0],
                                          reverse=True)
                        _, pred = best_hyp[0]
                        results[b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            if memory_bank is not None:
                memory_bank = memory_bank.index_select(0, select_indices)
            if memory_mask is not None:
                memory_mask = memory_mask.index_select(0, select_indices)
            if src_map is not None:
                src_map = src_map.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        results = [t[0] for t in results]
        return results

    def forward(self, batch):
        # src = batch.src
        # tgt = batch.tgt
        # # segs = batch.segs # use in bert
        # mask_src = batch.mask_src

        src = batch['src']
        tgt = batch['tgt']
        # segs = batch.segs # use in bert
        mask_src = batch['mask_src']
        top_vec, hid = self.encoder(src, mask_src)

        if self.training:
            dec_state = self.decoder.init_decoder_state(src, top_vec, hid)
            decode_output, _, attn = self.decoder(tgt[:, :-1],
                                                  top_vec,
                                                  dec_state,
                                                  memory_masks=mask_src)
            summary = None
        else:
            decode_output, attn = None, None
            summary = self._fast_translate_batch(batch,
                                                 top_vec,
                                                 self.max_length,
                                                 memory_mask=mask_src,
                                                 beam_size=self.beam_size,
                                                 hidden_state=hid,
                                                 copy_attn=self.args.copy_attn)

        return decode_output, summary, attn