# -*- coding=utf-8 -*-
from tqdm import tqdm
from collections import Counter, OrderedDict
import numpy as np
import logging
import json
import os

class Vocabulary(object):
    def __init__(self, do_lowercase=True):

        self.do_lowercase = do_lowercase
        self.length = None
        """
        word level operation
        """
        self.word_vocab = None
        self.word_vocab = ["<PAD>"]
        self.word2idx = None
        self.word_counter = Counter()
        """
        char level operation
        """
        self.char_vocab = None
        self.char_vocab = ["<PAD>"]
        self.char2idx = None
        self.char_counter = Counter()
<<<<<<< HEAD
        self.__build_index_mapper()

    def __build_index_mapper(self):
=======

    def build_index_mapper(self):
        logging.info("build index mapper")
<<<<<<< Updated upstream
>>>>>>> a4c61043139df7181725262524186164bbf898dc
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
=======
        self.special_tokens = None
>>>>>>> Stashed changes
        self.word2idx = dict(zip(self.word_vocab, range(len(self.word_vocab))))
        self.char2idx = dict(zip(self.char_vocab, range(len(self.char_vocab))))
        self.idx2word = dict(zip(range(len(self.word_vocab)), self.word_vocab))
        self.idx2char = dict(zip(range(len(self.char_vocab)), self.char_vocab))

<<<<<<< HEAD
=======
        logging.info(" length of self.word2idx is : %s", len(self.word2idx))
        logging.info(" length of self.char2idx is : %s", len(self.char2idx))
        logging.info(" length of self.idx2word is : %s", len(self.idx2word))
        logging.info(" length of self.idx2char is : %s", len(self.idx2char))
        print(self.idx2word[0])
        print(self.word2idx['<PAD>'])

>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
    def build_ner_vacab(self):
        pass

    def build_clf_vocab(self):
        pass

    def build_absa_vocab(self):
        pass

    def build_retrieval_vocab(self):
        pass

    def build_mrc_vocab(self, instances, min_word_count=-1, min_char_count=-1):
        logging.info("Building mrc vocabulary.")
        for instance in tqdm(instances):
            for token in instance['context_tokens']:
                for char in token:
                    self.char_counter[char] += 1
                token = token.lower() if self.do_lowercase else token
                self.word_counter[token] += 1
            for token in instance['question_tokens']:
                for char in token:
                    self.char_counter[char] += 1
                token = token.lower() if self.do_lowercase else token
                self.word_counter[token] += 1
        for w, v in self.word_counter.most_common():
            if v >= min_word_count:
                self.word_vocab.append(w)
        for c, v in self.char_counter.most_common():
            if v >= min_char_count:
                self.char_vocab.append(c)
<<<<<<< Updated upstream
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
=======
>>>>>>> Stashed changes
        self.__build_index_mapper()

    def build_chatbot_vocab(self, instances, min_word_count=10, min_char_count=50):
        pass

    def convert_tokens_to_ids(self, items):	
        output = []	
        for item in items:	
            output.append(self.word2idx.setdefault(item, 0)) # TODO 默认补0，字典里没有的key	
        return output	

    def convert_ids_to_tokens(self, items):	
        output = []	
        for item in items:	
            output.append(self.idx2word.setdefault(item, '<UNK>')) # 超过字典范围的idx 补充为<UNK> 	
        return output	
<<<<<<< Updated upstream
<<<<<<< HEAD
=======
=======
=======
>>>>>>> Stashed changes
        self.build_index_mapper()

    def input_chatbot_vocab(self, vocab_list):
        logging.info("Input chatbot vocabulary.")
        for tmp_vocab in tqdm(vocab_list):
            self.word_vocab.append(tmp_vocab)
        self.length = len(self.word_vocab)
        self.build_index_mapper()

#    def build_chatbot_vocab(self, instances, min_word_count=10, min_char_count=50):
#        logging.info("Building chatbot vocabulary.")
#        for instance in tqdm(instances):
#            tokens = []
#            for item in instance['history']:
#                tokens.extend(item) 
#            for item in instance['true_utterance']:
#                tokens.extend(item)
#            for item in instance['false_utterance']:
#                tokens.extend(item)
#            for token in tokens:
#                for char in token:
#                    self.char_counter[char] += 1
#                token = token.lower() if self.do_lowercase else token
#                self.word_counter[token] += 1
#        for w, v in self.word_counter.most_common():
#            if v >= min_word_count:
#                self.word_vocab.append(w)
#        for c, v in self.char_counter.most_common():
#            if v >= min_char_count:
#                self.char_vocab.append(c)
#        self.length = len(self.char_vocab) + len(self.word_vocab)
#        self.build_index_mapper()

    def set_vocab(self, word_vocab):
        self.word_vocab += word_vocab
        self.length = len(self.word_vocab)
        self.build_index_mapper()

    #def set_vocab(self, word_vocab, char_vocab):
    #    self.word_vocab += word_vocab
    #    self.char_vocab += char_vocab
    #    self.length = len(self.char_vocab) + len(self.word_vocab)
    #    self.build_index_mapper()
<<<<<<< Updated upstream
>>>>>>> a4c61043139df7181725262524186164bbf898dc
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
=======
>>>>>>> Stashed changes

    def get_word_pad_idx(self):
        return self.word2idx['<PAD>']

    def get_char_pad_idx(self):
        return self.char2idx['<PAD>']

    def get_word_vocab(self):
        return self.word_vocab

    def get_char_vocab(self):
        return self.char_vocab

    def get_word_counter(self):
        return self.word_counter

    def get_vocab_size(self):
        return self.length

<<<<<<< Updated upstream
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
=======
>>>>>>> Stashed changes
    def convert_by_vocab(self, items):	
        output = []	
        for item in items:
            output.append(self.word2idx.setdefault(item, 0)) # TODO 默认补0，字典里没有的key	
        return output	

    def convert_by_inv_vocab(self, items):	
        output = []	
        for item in items:	
            output.append(self.idx2word.setdefault(item, '<UNK>')) # 超过字典范围的idx 补充为<UNK> 	
        return output	

    def convert_tokens_to_ids(self, tokens):	
        return self.convert_by_vocab(tokens)	

    def convert_ids_to_tokens(self, ids):	
        return self.convert_by_inv_vocab(ids)

<<<<<<< Updated upstream
>>>>>>> a4c61043139df7181725262524186164bbf898dc
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
=======
>>>>>>> Stashed changes
    def save(self, file_path):
        logging.info("Saving vocabulary at {}".format(file_path))
        with open(file_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def load(self, file_path):
        logging.info("Loading vocabulary at {}".format(file_path))
        with open(file_path) as f:
            vocab_data = json.load(f)
            self.__dict__.update(vocab_data)
