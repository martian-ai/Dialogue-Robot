# coding:utf-8
# sample convert from origin instances to training instances
# mrc
#
# chat
import os
import collections
import six
import sys
import logging

class SampleFormatConvertor(object):
    def __init__(self, vocab, tokenizer, doc_stride=128):
        self.do_lower_case = True
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.index = 0
        self.doc_stride = doc_stride
<<<<<<< HEAD
=======
        self.turn = 1
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325

    def convert(self, instances, data_type, max_seq_length=512, is_training=True, token_done=True):
        new_instances = []
        for instance in instances:
            if data_type == 'douban':
                new_instance = self.__douban(instance, max_seq_length, is_training, token_done)
<<<<<<< HEAD
                new_instances.extend(new_instance)
            if data_type == 'squad':
                new_instance = self.__squad_bert(instance, max_seq_length, is_training, token_done)
                new_instances.extend(new_instance)
        return new_instances

=======
                new_instances.append(new_instance)
            if data_type == 'squad':
                new_instance = self.__squad_bert(instance, max_seq_length, is_training, token_done)
                new_instances.append(new_instance)
        return new_instances


>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
    def __douban(self, instance, max_seq_length, is_training, token_done):
        if is_training:
            if token_done:
                history_token = instance['history']
                true_utterance_token = instance['true_utterance']
                false_utterance_token = instance['false_utterance']
            else:
                history_token = [ self.tokenizer.tokenize(item)[0] for item in instance['history'] ]
<<<<<<< HEAD
                true_utterance_token = [ self.tokenizer.tokenize(item)[0] for item in instance['true_utterance_token'] ]
                false_utterance_token = [ self.tokenizer.tokenize(item)[0] for item in instance['false_utterance_token'] ]

=======
                true_utterance_token = [ self.tokenizer.tokenize(item)[0] for item in instance['true_utterance'] ]
                false_utterance_token = [ self.tokenizer.tokenize(item)[0] for item in instance['false_utterance'] ]
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
            # word level token cut
            history_token = [ item[0:max_seq_length] if len(item) > max_seq_length else item for item in history_token]
            true_utterance_token = [ item[0:max_seq_length] if len(item) > max_seq_length else item for item in true_utterance_token]
            false_utterance_token = [ item[0:max_seq_length] if len(item) > max_seq_length else item for item in false_utterance_token]
<<<<<<< HEAD

=======
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
            # word level index
            history_idx = [ self.vocab.convert_tokens_to_ids(item) for item in history_token]
            true_utterance_idx = [ self.vocab.convert_tokens_to_ids(item) for item in true_utterance_token]
            false_utterance_idx = [ self.vocab.convert_tokens_to_ids(item) for item in false_utterance_token]
<<<<<<< HEAD

=======
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
            # get sequence length
            history_len, true_utterance_len, false_utterance_len = [], [], []
            for item in history_idx:
                history_len.append(len(item))
            for item in true_utterance_idx:
                true_utterance_len.append(len(item))
            for item in false_utterance_idx:
                false_utterance_len.append(len(item))
<<<<<<< HEAD

=======
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
            # Zero-pad up to the sequence length.
            for item in history_idx:
                while len(item) < max_seq_length:
                    item.append(0)
            for item in true_utterance_idx:
                while len(item) < max_seq_length:
                    item.append(0)
            for item in false_utterance_idx:
                while len(item) < max_seq_length:
                    item.append(0)

            # max turn cut or padding for history
<<<<<<< HEAD
            if len(history_idx) > 5:
                history_idx = history_idx[:5]
                history_len = history_len[:5]
            else:
                [history_idx.append([0]*max_seq_length) for _ in range(5-len(history_idx))]
                [history_len.append(0) for _ in range(5-len(history_idx))]
=======
            if len(history_idx) > self.turn:
                history_idx = history_idx[:self.turn]
                history_len = history_len[:self.turn]
            else:
                [history_idx.append([0]*max_seq_length) for _ in range(self.turn-len(history_idx))]
                [history_len.append(0) for _ in range(self.turn-len(history_idx))]
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325

            new_instance = {}
            new_instance['history_idx'] = history_idx
            new_instance['history_len'] = history_len
            new_instance['true_utterance_idx'] = true_utterance_idx
            new_instance['true_utterance_len'] = true_utterance_len
            new_instance['false_utterance_idx'] = false_utterance_idx
            new_instance['false_utterance_len'] = false_utterance_len

            return new_instance

        else: # not training
            if token_done:
                history_token = instance['history']
                utterance_token = instance['utterance']
            else:
                history_token = [ self.tokenizer.tokenize(item)[0] for item in instance['history'] ]
                utterance_token = [ self.tokenizer.tokenize(item)[0] for item in instance['utterance'] ]

            # word level token cut
            history_token = [ item[0:max_seq_length] if len(item) > max_seq_length else item for item in history_token]
            utterance_token = [ item[0:max_seq_length] if len(item) > max_seq_length else item for item in utterance_token]

            # word level index
            history_idx = [ self.vocab.convert_tokens_to_ids(item) for item in history_token]
            utterance_idx = [ self.vocab.convert_tokens_to_ids(item) for item in utterance_token]

            # get sequence length
            history_len, utterance_len = [], []
            for item in history_idx:
                history_len.append(len(item))
            for item in utterance_idx:
                utterance_len.append(len(item))

            # Zero-pad up to the sequence length.
            for item in history_idx:
                while len(item) < max_seq_length:
                    item.append(0)
            for item in utterance_idx:
                while len(item) < max_seq_length:
                    item.append(0)

            # max turn cut or padding for history
<<<<<<< HEAD
            if len(history_idx) > 5:
                history_idx = history_idx[:5]
                history_len = history_len[:5]
            else:
                [history_idx.append([0]*max_seq_length) for _ in range(5-len(history_idx))]
                [history_len.append(0) for _ in range(5-len(history_len))]
=======
            if len(history_idx) > self.turn:
                history_idx = history_idx[:self.turn]
                history_len = history_len[:self.turn]
            else:
                [history_idx.append([0]*max_seq_length) for _ in range(self.turn-len(history_idx))]
                [history_len.append(0) for _ in range(self.turn-len(history_len))]
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325

            new_instance = {}
            new_instance['history_idx'] = history_idx
            new_instance['history_len'] = history_len
            new_instance['utterance_idx'] = utterance_idx
            new_instance['utterance_len'] = utterance_len

            return new_instance

    def __squad_bert(self, instance, max_seq_length, is_training, token_done, max_query_length=64):

        doc_tokens = instance['context_tokens'] if not self.do_lower_case else [token.lower() for token in instance['context_tokens']]
        question = instance['question'].lower() if self.do_lower_case else instance['question']
        query_tokens = self.tokenizer.tokenize(question)[0]
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]
        tok_to_orig_index = [] # 你们 在 潘家园 [0, 0, 1, 2, 2, 2]
        orig_to_tok_index = [] # 你们 在 潘家园 [0, 2, 3, ?]
        all_doc_tokens = [] # 文档中所有字级别的token
        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self.tokenizer.tokenize(token)[0]
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        
        tok_start_position = None
        tok_end_position = None

        if is_training :
            tok_start_position = orig_to_tok_index[instance['answer_start']]
            if instance['answer_end'] < len(doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[instance['answer_end'] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
                # (tok_start_position, tok_end_position) = _improve_answer_span(all_doc_tokens, tok_start_position, tok_end_position, tokenizer,instance['context'].lower())

        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3 # 最后三个token 是 [CLS], [SEP] ,[SEP]
        # 通过sliding window 的方法处理 长文档的问题
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, self.doc_stride) #Todo
        new_instances = [] 
        for (doc_span_index, doc_span) in enumerate(doc_spans): # 遍历所有截取的片段
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []  # query is 0, answer is 1, mask is 0
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = self.vocab.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids) # 1 mask
            while len(input_ids) < max_seq_length: # 补充0
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            """
            start_position 和 end_position 设置
            dureader robust devset 的 start_positon 为 -1
            """
            start_position = None
            end_position = None
            if is_training:
                # 训练时, 如果当前doc中没有答案, 则跳过
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = -1
                    end_position = -1
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
                if start_position == -1 and end_position == -1:  # 训练时 start_position 和 end_postion 为 -1, 说明标注错误
                    continue
            if not is_training:
                start_position = -2
                end_position = -2

            new_instance = {
                'doc_span_index': doc_span_index,
                'tokens': tokens,
                'token_to_orig_map': token_to_orig_map,
                'token_is_max_context': token_is_max_context,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'start_position': start_position,
                'end_position': end_position,
            }

            for k, v in instance.items():
                if k not in new_instance:
                    new_instance[k] = v
            new_instances.append(new_instance)

        return new_instances

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


# def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,orig_answer_text):
#     """Returns tokenized answer spans that better match the annotated answer."""

#     # The SQuAD annotations are character based. We first project them to
#     # whitespace-tokenized words. But then after WordPiece tokenization, we can
#     # often find a "better match". For example:
#     #
#     #   Question: What year was John Smith born?
#     #   Context: The leader was John Smith (1895-1943).
#     #   Answer: 1895
#     #
#     # The original whitespace-tokenized answer will be "(1895-1943).". However
#     # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
#     # the exact answer, 1895.
#     #
#     # However, this is not always possible. Consider the following:
#     #
#     #   Question: What country is the top exporter of electornics?
#     #   Context: The Japanese electronics industry is the lagest in the world.
#     #   Answer: Japan
#     #
#     # In this case, the annotator chose "Japan" as a character sub-span of
#     # the word "Japanese". Since our WordPiece tokenizer does not split
#     # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
#     # in SQuAD, but does happen.
#     tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

#     for new_start in range(input_start, input_end + 1):
#         for new_end in range(input_end, new_start - 1, -1):
#             text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
#             if text_span == tok_answer_text:
#                 return (new_start, new_end)

#     return (input_start, input_end)
