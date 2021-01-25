import json
import logging
import re
import string
from collections import OrderedDict, Counter
from tqdm import tqdm
from itertools import groupby
import numpy as np
<<<<<<< HEAD
from utils.tokenizer import JieBaTokenizer
from utils.dataset.base_dataset import BaseReader, BaseEvaluator
=======
from solutions.utils.datahelper.tokenizer import JieBaTokenizer
from solutions.utils.datahelper.dataset.base_dataset import BaseReader, BaseEvaluator
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325


class DoubanReader(BaseReader):
    def __init__(self,fine_grained = False):
        self.tokenizer = JieBaTokenizer()

    def read_response(self, file_path):
<<<<<<< HEAD
        logging.info('Reading response file ad %s', file_path)
=======
        logging.info('### Begin : Reading response file at %s', file_path)
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
        response_dict = {}
        with open(file_path, mode='r', encoding='utf-8') as f:
            for line in f.readlines():
                cut_line = line.strip().split('\t')
                if len(cut_line) == 2:
                    response_dict[cut_line[0]] = cut_line[1]
<<<<<<< HEAD
=======
        logging.info('### Done : Reading response file at %s', file_path)
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
        self.response_dict = response_dict

    def read(self, file_path):
        logging.info("Reading file at %s", file_path)
        instances = self._read(file_path)
        #instances = [instance for instance in instances]
        instances = [instance for instance in tqdm(instances)]
        return instances

    def _read(self, file_path):
        with open(file_path, encoding='utf-8', mode='r') as dataset:
            for line in dataset.readlines():
                line = line.strip().split('\t', 1)[1]
                tokens = line.split('_EOS_')

<<<<<<< HEAD
                history = [ item.strip().split(' ') for item in tokens[:-1] ]
=======
                history = [ item.strip() for item in tokens[:-1] ]
                # history = [ item.strip().split(' ') for item in tokens[:-1] ]
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
                utterance = tokens[-1].split('\t')
                true_index_list = utterance[1].split('|')
                false_index_list = utterance[2].split('|')
                # print('*'*40)
                # print('true_index_list', true_index_list)
                # print('false_index_list', false_index_list)
                instance = None
<<<<<<< HEAD
                if all(['NA' not in true_index_list, 'NA' not in false_index_list]): # TODO 1. NA 个数统计 2. 有NA 的情况是否可以使用
                    true_utterance = [ self.response_dict.get(index).strip().split(' ') for index in true_index_list]
                    false_utterance = [ self.response_dict.get(index).strip().split(' ') for index in false_index_list]
                    instance = self._make_instance(history, true_utterance, false_utterance)
=======

                try:
                    if all(['NA' not in true_index_list, 'NA' not in false_index_list]): # TODO 1. NA 个数统计 2. 有NA 的情况是否可以使用

                        true_utterance = [ self.response_dict.get(index).strip() for index in true_index_list]
                        false_utterance = [ self.response_dict.get(index).strip() for index in false_index_list]
                        # true_utterance = [ self.response_dict.get(index).strip().split(' ') for index in true_index_list]
                        # false_utterance = [ self.response_dict.get(index).strip().split(' ') for index in false_index_list]
                        # print("&"*20)
                        # print(true_utterance)
                        # print(false_utterance)
                        instance = self._make_instance(history, true_utterance, false_utterance)
                except:
                    continue
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
                
                if instance is None:
                    print('douban.py _make_instance results is None, skip this qas')
                    continue
<<<<<<< HEAD
=======
                
                # print("Z"*100)
                # print(instance)
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325

                yield instance

    def _make_instance(self, history, true_utterance, false_utterance):
        # 返回数据处理后的结果
        return OrderedDict({
            "history": history,
            "true_utterance": true_utterance,
            "false_utterance": false_utterance,
        })

class DoubanEvaluator(BaseEvaluator):
<<<<<<< HEAD
    def __init__(self, file_path, monitor='r2'): # TODO 其他评价指标添加
=======
    def __init__(self, file_path, monitor): # TODO 其他评价指标添加
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
        self.ground_dict = dict()
        self.monitor = monitor

    def get_monitor(self):
        return self.monitor

    def get_score(self, scores, instances):
        labels = [ tmp['y_true'] for tmp in instances]
<<<<<<< HEAD
        result_r2 = self.computeR2_1(scores, labels)
        # result_r10 = self.computeR10_1(scores, labels)

        return {'r10':1.0, 'r2':result_r2} # TODO 当前 R10 默认为1，后续需要改变，根据任务类型来判断，比如 eval 用r2，test 用r10

    def computeR10_1(self, scores, labels, count = 10):
        """
        scores : [0.1, 0.3, 0.2, 0.4, 0.5, 0.3, 0.2, 0.3, 0.7, 0.1] * batch_size
        labels : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] * batch_size
        """
        total = 0
        correct = 0
        for i in range(len(labels)):
            if labels[i] == 1:
                total = total+1
                sublist = scores[i:i+count]
                if max(sublist) == scores[i]:
                    correct = correct + 1
        return float(correct)/total 
 
    def computeR2_1(self, scores, labels, count = 2):
=======
        result_p_1 = self.computeP_1(scores, labels, count=10)
        #result_p_2_1 = self.computeP_1(scores, labels, count=2)
        #return {'p5_1': result_p_2_1, 'p2_1', result_p_2_1}
        return {'p_1':result_p_1}
 
    def computeP_1(self, scores, labels, count):
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
        """
        scores : [0.1, 0.3] * batch_size
        labels : [1, 0] * batch_size
        """
<<<<<<< HEAD
=======
        print(scores[:20])
        print(labels[:20])
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
        total = 0 # 统计有多少个1
        correct = 0 # 统计有多少个正确的
        for i in range(len(labels)):
            if labels[i] == 1:
                total = total+1
                sublist = scores[i:i+count]
<<<<<<< HEAD
=======
                if sublist[0] == sublist[1]:
                    continue
>>>>>>> 0dea377a37ecbf1591f782cc5893be4d0f877325
                if max(sublist) == scores[i]:
                    correct = correct + 1
        return float(correct)/total
