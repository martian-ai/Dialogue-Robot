import sys, os
import random
import itertools
import tensorflow as tf
import pickle as pkl
sys.path.append('..')

from solutions.ranker.smn import SMN
from utils.data_helper.chatbot_batch_generator import BatchGenerator
from utils.data_helper.ranker_helper import RankerDataHelper
from utils.dataset.douban import DoubanReader, DoubanEvaluator
from utils.data_helper.chatbot_vocabulary import Vocabulary

random.seed(1234)
os.environ["CUDA_VISIBLE_DEVICES"] = str(2)
warmup_proportion = 0.1
epochs = 20 
batch_size = 512
learning_rate = 1e-3

data_folder = '../corpus/corpus/chatbot/douban'
train_file = os.path.join(data_folder, "train.txt") # TODO
dev_file = os.path.join(data_folder, "valid.txt") # TODO
test_file = os.path.join(data_folder, "test.txt") # TODO 测试集  R10_1 这种评测处理
response_file = os.path.join(data_folder, "responses.txt")
tencent_embedding_file = 'Tencent_AILab_ChineseEmbedding.txt'
tencent_vocab_file = '../resources/tencent_vocab.pkl'
douban_vocab_file = '../resources/douban_vocab.txt'
douban_embedding_file = '../resources/douban_embedding.pkl'
save_folder = './chatbot-smn-0526'

print('gpu card :', '2')
print('epochs :', epochs)
print('batch size :', batch_size)
print('data folder :', data_folder)
print('vocab save and load file :', douban_vocab_file)
print('embedding save and load file :', douban_embedding_file)
print('save floder :', save_folder)

# read Douban data
reader = DoubanReader()
reader.read_response(response_file)
train_data = reader.read(train_file)
random.shuffle(train_data)
eval_data = reader.read(dev_file)
test_data = reader.read(test_file)

# TODO reader Other Data

# 定义统一的数据格式进行 convert

print('train data 0')
print(train_data[0])

# build vocab
vocab = Vocabulary()
embeddings = []
if os.path.exists(douban_vocab_file) and os.path.exists(douban_embedding_file):
    vocab.load(douban_vocab_file)
    with open(douban_embedding_file, 'rb') as f:
        embeddings = pkl.load(f)['word_embedding']
    print('*'*100)
    print(vocab.get_word_vocab()[:100])
    print('*'*100)
    print(vocab.get_char_vocab()[:100])
    print('*'*100)
    print(len(embeddings))
    print(embeddings[100])
else:
    vocab.build_vocab(train_data + eval_data + test_data, min_word_count=20, min_char_count=50)
    
    # 使用tencent 的embedding, 只能使用vocab.set_vocab() 这种方式
    with open(tencent_vocab_file, 'rb') as f:
        tencent_vocab = pkl.load(f)
        tencent_word_vocab, tencent_char_vocab = tencent_vocab['char_vocab'], tencent_vocab['word_vocab'] 
        tmp_word_vocab_set, tmp_char_vocab_set = set(vocab.get_word_vocab()), set(vocab.get_char_vocab())

        tmp_word_vocab = [tmp for tmp in tencent_word_vocab if tmp in tmp_word_vocab_set]
        tmp_char_vocab = [tmp for tmp in tencent_char_vocab if tmp in tmp_char_vocab_set]

        vocab.set_vocab(tmp_word_vocab, tmp_char_vocab)
        vocab.save(douban_vocab_file)

        # TODO 当前未使用bert 
        embeddings = vocab.make_word_embedding(tencent_embedding_file)
        with open(douban_embedding_file, 'wb') as f:
            pkl.dump({'word_embedding':embeddings}, f)


# data convert and generator
bert_data_helper = RankerDataHelper(vocab)
train_data = bert_data_helper.convert(train_data,data='douban',max_seq_length=16) # TODO 16 可能有点短
eval_data = bert_data_helper.convert(eval_data,data='douban',max_seq_length=16)
# TODO 其他数据格式的convert 方式

print('train data 0 after evaluate')
print(train_data[0])

train_batch_generator = BatchGenerator(vocab,train_data,training=True,batch_size=batch_size)
eval_batch_generator = BatchGenerator(vocab,eval_data,training=False,batch_size=batch_size)

# model train and evaluate
model = SMN(vocab=vocab, sequence_len=16, hidden_unit=128) 
num_train_steps = int(len(train_data) / batch_size * epochs) # 一共运行多少个 step # TODO 非bert方式， warmup 是否必须 
num_warmup_steps = int(num_train_steps * warmup_proportion) # 优化之前 运行多个step 作为warmup
model.compile(learning_rate,num_train_steps, num_warmup_steps) 

evaluator = DoubanEvaluator(dev_file)
model.train_and_evaluate(train_batch_generator, eval_batch_generator, evaluator, embeddings, epochs, eposides=15, save_dir=save_folder)

# Code Review
