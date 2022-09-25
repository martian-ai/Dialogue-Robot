# -*- coding: utf-8 -*-

import os
import pickle
import json
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer
from transformers import RobertaTokenizer


def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        #self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_bert_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


# A. A、 A． 等情况删除
def text_clean(text):
    text = text.strip('A.')
    text = text.strip('B.')
    text = text.strip('C.')
    text = text.strip('D.')
    text = text.strip('A、')
    text = text.strip('B、')
    text = text.strip('C、')
    text = text.strip('D、')
    text = text.strip('A．')
    text = text.strip('B．')
    text = text.strip('C．')
    text = text.strip('D．')
    text = text.strip()
    return text


class MultiChoiceDataset(Dataset):
    def __init__(self, fname, tokenizer):
        with open(fname, "r", encoding='utf-8') as reader:
            examples = json.load(reader)
        #examples = examples[:100]
        print("*"*200)
        print('{} examples read done'.format(len(examples)))
        all_data = []
        for entry in tqdm(examples):
            #print(">"*200)
            id = entry['ID']
            content = entry['Content']
            print('content', content)
            print('length of content', len(content))
            content_indices = tokenizer.text_to_sequence(content)
            content_len = np.sum(content_indices != 0 )
            questions = entry['Questions']
            for item in questions:
                question = item['Question']
                question_indices = tokenizer.text_to_sequence(question)
                question_len = np.sum(question_indices != 0 )
                print('question', question)
                print('question length', question_len)
                choice_list = item['Choices']
                try:
                    answer = item['Answer']
                except:
                    print(item)

                q_id = item['Q_id']

                if len(choice_list) == 2:
                    choice_list.append('')
                    choice_list.append('')
                elif len(choice_list) == 3:
                    choice_list.append('')

                # choice a
                choice_a = text_clean(choice_list[0])
                choice_a_indices = tokenizer.text_to_sequence(choice_a)
                choice_a_len = np.sum(choice_a_indices != 0 )
                #concat_a_indices = tokenizer.text_to_sequence('[CLS] ' + content + ' [SEP] ' + question + '[SEP]' + choice_a + '[SEP]' )
                #concat_segments_a_indices = [0] * (content_len + 2) + [1] * ( question_len + choice_a_len + 2)
                concat_a_indices = tokenizer.text_to_sequence('[CLS] ' + choice_a[:90] + question + '[SEP]' + content + '[SEP]' )
                concat_segments_a_indices = [0] * ( question_len + choice_a_len + 2) + [1] * (content_len + 1) 
                concat_segments_a_indices = pad_and_truncate(concat_segments_a_indices, tokenizer.max_seq_len)
                print('choice a string', choice_a)
                print('choice a length', choice_a_len)
                print('choice a indices', concat_a_indices)
                print('choice a segment indices', concat_segments_a_indices)

                # choice b
                choice_b = text_clean(choice_list[1])
                choice_b_indices = tokenizer.text_to_sequence(choice_b)
                choice_b_len = np.sum(choice_b_indices != 0 )
                #concat_b_indices = tokenizer.text_to_sequence('[CLS] ' + content + ' [SEP] ' + question + '[SEP]' + choice_b + '[SEP]' )
                #concat_segments_b_indices = [0] * (content_len + 2) + [1] * ( question_len + choice_b_len + 2)
                concat_b_indices = tokenizer.text_to_sequence('[CLS] ' + choice_b[:90] + question[:38] + '[SEP]' + content[:380] + '[SEP]' )
                concat_segments_b_indices = [0] * ( question_len + choice_b_len + 2) + [1] * (content_len + 1) 
                concat_segments_b_indices = pad_and_truncate(concat_segments_b_indices, tokenizer.max_seq_len)
                print('choice b string', choice_b)
                print('choice b length', choice_b_len)
                print('choice b indices', concat_b_indices)
                print('choice b segment indices', concat_segments_b_indices)

                # choice c
                choice_c = text_clean(choice_list[2])
                choice_c_indices = tokenizer.text_to_sequence(choice_c)
                choice_c_len = np.sum(choice_c_indices != 0 )
                #concat_c_indices = tokenizer.text_to_sequence('[CLS] ' + content + ' [SEP] ' + question + '[SEP]' + choice_c + '[SEP]' )
                #concat_segments_c_indices = [0] * (content_len + 2) + [1] * ( question_len + choice_c_len + 2)
                concat_c_indices = tokenizer.text_to_sequence('[CLS] ' + choice_c[:90] + question[:38] + '[SEP]' + content[:380] + '[SEP]' )
                concat_segments_c_indices = [0] * ( question_len + choice_c_len + 2) + [1] * (content_len + 1) 
                concat_segments_c_indices = pad_and_truncate(concat_segments_c_indices, tokenizer.max_seq_len)
                print('choice c string', choice_c)
                print('choice c length', choice_c_len)
                print('choice c indices', concat_c_indices)
                print('choice c segment indices', concat_segments_c_indices)

                # choice d
                choice_d = text_clean(choice_list[3])
                choice_d_indices = tokenizer.text_to_sequence(choice_d)
                choice_d_len = np.sum(choice_d_indices != 0 )
                #concat_d_indices = tokenizer.text_to_sequence('[CLS] ' + content + ' [SEP] ' + question + '[SEP]' + choice_d + '[SEP]' )
                #concat_segments_d_indices = [0] * (content_len + 2) + [1] * ( question_len + choice_d_len + 2)
                concat_d_indices = tokenizer.text_to_sequence('[CLS] ' +  choice_d[:90] + question[:38] +  '[SEP]' + content[:380] + '[SEP]' )
                concat_segments_d_indices = [0] * ( question_len + choice_d_len + 2) + [1] * (content_len + 1) 
                concat_segments_d_indices = pad_and_truncate(concat_segments_d_indices, tokenizer.max_seq_len)

                print('choice d string', choice_d)
                print('choice d length', choice_d_len)
                print('choice d indices', concat_d_indices)
                print('choice d segment indices', concat_segments_d_indices)


                if answer == 'A':
                    polarity = 0
                elif answer == 'B':
                    polarity = 1
                elif answer == 'C':
                    polarity = 2
                elif answer == 'D':
                    polarity = 3

                data = {
                    'concat_a_indices':concat_a_indices,
                    'concat_segments_a_indices':concat_segments_a_indices,
                    'concat_b_indices':concat_b_indices,
                    'concat_segments_b_indices':concat_segments_b_indices,
                    'concat_c_indices':concat_c_indices,
                    'concat_segments_c_indices':concat_segments_c_indices,
                    'concat_d_indices':concat_d_indices,
                    'concat_segments_d_indices':concat_segments_d_indices,
                    'polarity' : polarity
                }
                all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
