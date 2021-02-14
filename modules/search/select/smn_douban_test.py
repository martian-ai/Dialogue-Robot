import sys, os
import random
from tqdm import tqdm
import itertools
import tensorflow as tf
import pickle as pkl
import numpy as np
import jieba
import os 
sys.path.append('..')
from solutions.selector.smn import SMN
from solutions.utils.datahelper.dataset.douban import DoubanReader, DoubanEvaluator
from solutions.utils.datahelper.vocabulary import Vocabulary
from solutions.utils.datahelper.generator import BatchGenerator
from solutions.utils.datahelper.convertor import SampleFormatConvertor
from solutions.utils.datahelper.tokenizer import JieBaTokenizer 

random.seed(1234)
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
batch_size = 1024
learning_rate = 1e-3

def ComputeR(scores,count, tmp=1):
    total = 0
    correct = 0
    labels = [1, 0, 0, 0, 0, 0, 0,0,0,0] * int(len(scores)//count)

    print("A"*100)

    print(len(labels))
    print(len(scores))

    if len(labels) != len(scores):
        raise Exception
        
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total+1
            sublist = scores[i:i+count]
            if scores[i] in sorted(sublist, reverse=True)[:tmp]:
                correct += 1 
    return float(correct)/ total


def get_data(test_data):
    all_data = []
    all_label = []
    for instance in test_data:
        inference_data = []
        tmp_label = []
        for item in instance['true_utterance']:
            tmp_data = {}
            tmp_data['history'] = instance['history']
            tmp_data['utterance'] = [item]
            inference_data.append(tmp_data)
            tmp_label.append('1')
        for item in instance['false_utterance']:
            tmp_data = {}
            tmp_data['history'] = instance['history']
            tmp_data['utterance'] = [item]
            inference_data.append(tmp_data)
            tmp_label.append('0')
        all_data.append(inference_data)
        all_label.append(tmp_label)

    return all_data

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    data_folder = '../resources/corpus/'
    embedding_file = '../resources/embedding_top_10w_vocab_1st.pkl'
    load_folder = './chatbot-smn-1120-sampling-4nd-small/best_weights'
    vocab_file = 'vocab_word_sort_uniq.txt'
    response_file = os.path.join(data_folder, "responses.txt")
    test_file = os.path.join(data_folder, 'chitchat_1112', "test_all_smn_top_10w.txt") 

    # maker vocab
    vocab = Vocabulary()
    with open(vocab_file, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        vocab_lines = [ item.strip() for item in lines ]
        print('vocab top 10 is : ', vocab_lines[:10])

        vocab.input_chatbot_vocab(vocab_lines)
    # make data
    reader = DoubanReader()
    reader.read_response(response_file)
    test_data = reader.read(test_file) # list of instance

    print('### test data length is : ', len(test_data))
    all_data = get_data(test_data)
    print('### all data length is : ', len(test_data))


    # build tokenizer
    tokenizer = JieBaTokenizer() 

    # data convert
    data_helper = SampleFormatConvertor(vocab, tokenizer)

    # load model
    model = SMN(vocab=vocab, sequence_len=20, hidden_unit=256) 
    model.load(load_folder)

    # inference data

    inference_data = []
    for tmp in all_data:
        tmp_result =data_helper.convert(tmp, data_type='douban',max_seq_length=20, is_training=False,token_done=False)
        #print("tmp_result")
        #print(tmp)
        #print(tmp_result)
        inference_data.extend(tmp_result)

    print(len(inference_data))
    inference_batch_generator = BatchGenerator(vocab, inference_data, training=False, batch_size=batch_size)
    results = model.inference(inference_batch_generator)
    print(results[:10])
    results = [item.tolist() for item in results ]
    print(results[:10])
    results = [ round(item[1], 4) for item in results]
    print(results[:10])
    print(len(results))

    print(ComputeR(results, 10, 1)) # R10@1
    print(ComputeR(results, 10, 2)) # R10@2
    print(ComputeR(results, 10, 5)) # R10@5
