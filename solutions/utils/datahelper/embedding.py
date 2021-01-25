<<<<<<< HEAD

#    def make_word_embedding(self, embedding_file, init_scale=0.02):
#         if self.word_vocab is None or self.word2idx is None:
#             raise ValueError("make_word_embedding must be called after build_vocab/set_vocab")

#         # 1. Parse pretrained embedding
#         embedding_dict = dict()
#         with open(embedding_file) as f:
#             for line in f:
#                 if len(line.rstrip().split(" ")) <= 2: continue
#                 word, vector = line.rstrip().split(" ", 1)
#                 embedding_dict[word] = np.fromstring(vector, dtype=np.float, sep=" ")

#         # 2. Update word vocab according to pretrained word embedding
#         new_word_vocab = []
#         special_tokens_set = set(self.special_tokens if self.special_tokens is not None else [])
#         for word in self.word_vocab:
#             if word == "<PAD>" or word in embedding_dict or word in special_tokens_set:
#                 new_word_vocab.append(word)
#         self.word_vocab = new_word_vocab
#         self._build_index_mapper()

#         # 3. Make word embedding matrix
#         embedding_size = embedding_dict[list(embedding_dict.keys())[0]].shape[0]
#         embedding_list = []
#         for word in self.word_vocab + self.char_vocab:
#             if word == "<PAD>":
#                 embedding_list.append(np.zeros([1, embedding_size], dtype=np.float))
#             elif word in special_tokens_set:
#                 embedding_list.append(np.random.uniform(-init_scale, init_scale, [1, embedding_size]))
#             else:
#                 embedding_list.append(np.reshape(embedding_dict[word], [1, embedding_size]))

#         # To be consistent with the behavior of tf.contrib.lookup.index_table_from_tensor,
#         # <UNK> token is appended at last
#         embedding_list.append(np.random.uniform(-init_scale, init_scale, [1, embedding_size]))

#         return np.concatenate(embedding_list, axis=0)
=======
import numpy as np

def make_word_embedding(vocab, embedding_file, init_scale=0.02):
    if vocab.word_vocab is None or vocab.word2idx is None:
        raise ValueError("make_word_embedding must be called after build_vocab/set_vocab")

    # 1. Parse pretrained embedding
    embedding_dict = dict()
    with open(embedding_file) as f:
        print("*"*20)
        print('read pretrain embedding begin')
        for line in f:
            if len(line.rstrip().split(" ")) <= 2: continue
            word, vector = line.rstrip().split(" ", 1)
            embedding_dict[word] = np.fromstring(vector, dtype=np.float, sep=" ")
        print("*"*20)
        print('read pretrain embedding done')

    # 2. Update word vocab according to pretrained word embedding
    #new_word_vocab = []
    special_tokens_set = set(vocab.special_tokens if vocab.special_tokens is not None else [])
    #for word in vocab.word_vocab:
    #    if word == "<PAD>" or word in embedding_dict or word in special_tokens_set:
    #        new_word_vocab.append(word)
    #vocab.word_vocab = new_word_vocab
    #vocab.build_index_mapper()

    # 3. Make word embedding matrix
    embedding_size = embedding_dict[list(embedding_dict.keys())[0]].shape[0]
    embedding_list = []
    for word in vocab.word_vocab:
        if word in special_tokens_set:
            embedding_list.append(np.random.uniform(-init_scale, init_scale, [1, embedding_size]))
        elif word in embedding_dict:
            embedding_list.append(np.reshape(embedding_dict[word], [1, embedding_size]))
        elif word == "<PAD>":
            embedding_list.append(np.zeros([1, embedding_size], dtype=np.float))
        else: # 当前词不在tentcent  字典里 
            print(word)
            #embedding_list.append(np.zeros([1, embedding_size], dtype=np.float))
            embedding_list.append(np.random.uniform(-init_scale, init_scale, [1, embedding_size]))

    # To be consistent with the behavior of tf.contrib.lookup.index_table_from_tensor,
    # <UNK> token is appended at last
    # embedding_list.append(np.random.uniform(-init_scale, init_scale, [1, embedding_size]))

    print("embedding list build done")

    return np.concatenate(embedding_list, axis=0)
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325


# def __set_embedding_info(self):
# """
# :return: embedding files dict
# """
# embedding_files = {
#     'Static':{
#         "random":"",
#         "word2vec":"",
#         "glove":"",
#         "tencent":"/Users/sunhongchao/Documents/Tencent_AILab_ChineseEmbedding.txt"
#     },
#     'Dynamic':{
#         "BERT":"",
#         "ELMo":"",
#         "ERINE":"",
#         "GPT-2-Chinese":"",
#         "BERT-WWW":""
#     }
# }

# self.embedding_files = embedding_files

# def __set_embedding(self):

# def get_word2vec():
#     word2vec = {}
#     fin = open(self.embedding_files['Static']['tencent'], 'r', encoding='utf-8', newline='\n', errors='ignore')
#     for line in fin:
#         tokens = line.rstrip().split(' ') 
#         if tokens[0] in self.word2idx.keys():
#             word2vec[tokens[0]] = np.asarray(tokens[1:], dtype='float16')
#     fin.close()

#     return word2vec

# if self.emb_type == 'random':
#     embedding_matrix = np.zeros((len(self.word2idx) + 2, 300)) # TODO 改成random
# elif self.emb_type == 'tencent':
#     word2vec = get_word2vec()
#     embedding_matrix = []
#     # embedding_matrix.append(np.random.rand(200))
#     for key, val in self.word2idx.items(): # 遍历当前语料构建的词典
#         if key == '<PAD>':
#             embedding_matrix.append(np.zeros(200))
#         else:
#             if key in word2vec.keys(): # 在 预训练embedding 的词典中
#                 embedding_matrix.append(word2vec[key])
#             else: # 不 在 预训练embedding 的词典中
#                 embedding_matrix.append(np.random.rand(200))

# elif self.emb_type == 'bert':
#     pass

<<<<<<< HEAD
# self.embedding_matrix = embedding_matrix
=======
# self.embedding_matrix = embedding_matrix
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
