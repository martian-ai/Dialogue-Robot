import sys, os
import random
import itertools
import tensorflow as tf
import pickle as pkl
sys.path.append('..')

from solutions.search.rank.selector.smn import SMN
from solutions.utils.datahelper.dataset.douban import DoubanReader, DoubanEvaluator
from solutions.utils.datahelper.vocabulary import Vocabulary
from solutions.utils.datahelper.embedding import make_word_embedding
from solutions.utils.datahelper.generator import BatchGenerator
from solutions.utils.datahelper.convertor import SampleFormatConvertor
from solutions.utils.datahelper.tokenizer import JieBaTokenizer 

random.seed(1234)
os.environ["CUDA_VISIBLE_DEVICES"] = str(3)
warmup_proportion = 0.1
epochs = 3000 
batch_size = 4096
learning_rate = 1e-3
sequence_len = 20

data_folder = '../resources/corpus'
#
train_file = os.path.join(data_folder, 'chitchat_smn', "release_format_v5_smn_1-1.txt") # TODO
#train_file = os.path.join(data_folder, 'chitchat_1112', "test_all_smn_top_10w.txt") # TODO
dev_file = os.path.join(data_folder, 'chitchat_1112', "test_all_smn_top_10w.txt") # TODO
response_file = os.path.join(data_folder, "responses.txt")
#
douban_embedding_file = '../resources/embedding_vocab_all_1st.pkl'
save_folder = './chatbot-train-all-test-10w-1nd-small' # 如果文件夹不为空，会默认使用上一次训练的模型, 继续训练
#
tencent_vocab_file = '../resources/tencent_vocab.pkl'
tencent_embedding_file = '/export/home/sunhongchao1/Tencent_AILab_ChineseEmbedding.txt'

# read Douban data
reader = DoubanReader()
reader.read_response(response_file)
train_data = reader.read(train_file)
random.shuffle(train_data)
print("### train data read done ")
eval_data = reader.read(dev_file)
print("### eval data read done ")
#test_data = reader.read(test_file)
print("### test data read done ")

print('train data 0 before evaluate')
print(train_data[0])

# build vocab
vocab = Vocabulary()

with open('vocab_word_all.txt', mode='r', encoding='utf-8') as f:
    lines = f.readlines()
    vocab_lines = [ item.strip() for item in lines ]

# vocab_lines = vocab_lines[:100000]

print(vocab_lines[:10])
vocab.input_chatbot_vocab(vocab_lines)
print('vocab load done')


embeddings = []
# if os.path.exists(douban_vocab_file) and os.path.exists(douban_embedding_file):
if os.path.exists(douban_embedding_file):
    with open(douban_embedding_file, 'rb') as f:
        embeddings = pkl.load(f)['word_embedding']
else:
    # 使用tencent 的embedding, 只能使用vocab.set_vocab() 这种方式
    with open(tencent_vocab_file, 'rb') as f:
        tencent_vocab = pkl.load(f)
        tencent_word_vocab  = tencent_vocab['word_vocab'] 
        # tencent_word_vocab, tencent_char_vocab = tencent_vocab['char_vocab'], tencent_vocab['word_vocab'] 
        tmp_word_vocab_set = set(vocab.get_word_vocab())

        tmp_word_vocab = [tmp for tmp in tencent_word_vocab if tmp in tmp_word_vocab_set]
        print(len(tmp_word_vocab))
        # tmp_char_vocab = [tmp for tmp in tencent_char_vocab if tmp in tmp_char_vocab_set]

    embeddings = make_word_embedding(vocab, tencent_embedding_file)
    with open(douban_embedding_file, 'wb') as f:
        pkl.dump({'word_embedding':embeddings}, f)

#else:
#    vocab.build_chatbot_vocab(train_data + eval_data + test_data, min_word_count=20, min_char_count=50)
#    
#    # 使用tencent 的embedding, 只能使用vocab.set_vocab() 这种方式
#    with open(tencent_vocab_file, 'rb') as f:
#        tencent_vocab = pkl.load(f)
#        tencent_word_vocab, tencent_char_vocab = tencent_vocab['char_vocab'], tencent_vocab['word_vocab'] 
#        tmp_word_vocab_set, tmp_char_vocab_set = set(vocab.get_word_vocab()), set(vocab.get_char_vocab())
#
#        tmp_word_vocab = [tmp for tmp in tencent_word_vocab if tmp in tmp_word_vocab_set]
#        tmp_char_vocab = [tmp for tmp in tencent_char_vocab if tmp in tmp_char_vocab_set]
#
#        vocab.set_vocab(tmp_word_vocab, tmp_char_vocab)
#        vocab.save(douban_vocab_file)
#
#        print('chart length')
#        print(len(vocab.char_vocab))
#        print('word vocab')
#        print(len(vocab.word_vocab))
#
#        # TODO 当前未使用bert 
#        embeddings = make_word_embedding(vocab, tencent_embedding_file)
#        with open(douban_embedding_file, 'wb') as f:
#            pkl.dump({'word_embedding':embeddings}, f)
#



tokenizer = JieBaTokenizer() 

# data convert and generator
data_helper = SampleFormatConvertor(vocab, tokenizer)
train_data = data_helper.convert(train_data,'douban',max_seq_length=sequence_len, is_training=True, token_done=False) # TODO 16 可能有点短
eval_data = data_helper.convert(eval_data,'douban',max_seq_length=sequence_len, is_training=True, token_done=False)
# TODO 其他数据格式的convert 方式

print('train data 0')
print(train_data[0])

train_batch_generator = BatchGenerator(vocab,train_data,training=True,batch_size=batch_size)
print('train data generator done')
eval_batch_generator = BatchGenerator(vocab,eval_data,training=True,batch_size=batch_size)
print('eval data generator done')

# model train and evaluate
model = SMN(vocab=vocab, sequence_len=sequence_len, hidden_unit=256) 
num_train_steps = int(len(train_data) / batch_size * epochs) # 一共运行多少个 step # TODO 非bert方式， warmup 是否必须 
# num_warmup_steps = int(num_train_steps * warmup_proportion) # 优化之前 运行多个step 作为warmup
model.compile(learning_rate, num_train_steps , 0) 

evaluator = DoubanEvaluator(dev_file, 'p_1')
model.train_and_evaluate(train_batch_generator, eval_batch_generator,
                         evaluator, embeddings, epochs, eposides=1, save_dir=save_folder)
