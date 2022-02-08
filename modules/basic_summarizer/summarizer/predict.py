# -*- coding: utf-8 -*-
import json
import os
from typing import List
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import re

# 本文件测试的时候要使用
import sys
sys.path.append('../..')
print(sys.path)

from modules.alpha_learner.nn_base.sum_cnn_rnn import CNN_RNN
from modules.utils.vocab.Vocab import Vocab
from modules.discourse.sum.summarizer.dataset import SumDataset

def text_segment(text):
    tmp = re.split('[!?。！？;；\\n]', text)
    if len(tmp[-1]) == 0:
        return tmp[:-1]
    else:
        return tmp

class SumPredictService():
    def __init__(self, resources_dir='./'):
        self.device = 'cpu'
        embedding_file_path = os.path.join(resources_dir, "embedding/embedding.npz")
        vocab_file_path =  os.path.join(resources_dir, "vocab/word2id.json")
        model_path = os.path.join(resources_dir, 'models/sum/summarunner/CNN_RNN.pt')

        self.embed = torch.Tensor(np.load(embedding_file_path)['arr_0'])
        with open(vocab_file_path) as f:
            self.word2id = json.load(f)
        self.vocab = Vocab(self.embed, self.word2id)
        self.checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.checkpoint['args'].device = None

        # print(self.checkpoint['args'])
        # print(self.checkpoint['args'].model)

        #self.net = getattr(models, self.checkpoint['args'].model)(self.checkpoint['args'])
        self.net = CNN_RNN(self.checkpoint['args'])
        self.net.load_state_dict(self.checkpoint['model'])
        self.net.eval()

    def predict(self, text, topK=3):
        # 参数校验处理
        if text is None or text.isspace():
            raise Exception('text参数为空或空白')
        if not isinstance(text, str):
            raise Exception('text不是字符串类型')
        if topK < 1:
            raise Exception('topk请求参数不合法')

        text = text.strip()
        print(text)
        cut_text = text_segment(text)

        print(cut_text)

        if len(cut_text) == 1:
            return [] # TODO 文档切分后只有一个切分结果，直接返回，有可能长度大于512
        cut_text = [ item for item in cut_text if '：' not in item ]
        cut_text = [ item for item in cut_text if len(item) > 5 ] # 限制句子长度必须大于 5 

        if len(cut_text) == 0:
            return []

        labels = ["0"] * len(cut_text)
        print(cut_text)
        examples = [{"doc": '\n'.join(cut_text), "labels": '\n'.join(labels), "summaries": cut_text[0]}]
        predict_dataset = SumDataset(examples)
        predict_iter = DataLoader(dataset=predict_dataset, batch_size=1, shuffle=False)

        all_hyp = []
        for batch in predict_iter:
            features, _, summaries, doc_lens = self.vocab.make_features(batch)
            probs = self.net(Variable(features).to(self.device), doc_lens)
            start = 0
            for doc_id, doc_len in enumerate(doc_lens):
                stop = start + doc_len
                print(probs)
                if doc_len == 1:
                    prob = torch.tensor([probs.tolist()])
                else:
                    prob = probs[start:stop]
                    
                topk = min(topK, doc_len)
                topk_indices = prob.topk(topk)[1].cpu().data.numpy()
                topk_indices.sort()
                doc = batch['doc'][doc_id].split('\n')[:doc_len]
                hyp = [doc[index] for index in topk_indices]
                all_hyp.append(hyp)
        # 返回值是一个字符串数组
        result = []
        if all_hyp[0] and len(all_hyp[0]) > 0:
            result = all_hyp[0]
        return result


if __name__ == '__main__':
    # import sys
    # sys.path.append('../..')
    # print(sys.path)
    summary_api = SumPredictService(resources_dir='/Users/sunhongchao/Documents/Bot/resources')
    text001 = '市民反映无法享受最低生活保障问题。市民反映，自己哥哥名叫潘福增，身份证号：110104195911240056，已经62岁，是智力残疾人，有残疾证，户籍地：西城区荣光胡同10号，归天桥街道办事处管，使用低保完全无法承担就医费用，自己哥哥独居，没有父母和儿女，应该可以享受最低收入，没有向街道申请过，来电希望帮助解决无法享受最低生活保障问题。注：请及时向来电人反馈办理情况。'
    text001 = '范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机\n1963年被任为主教；1990年被擢升为天主教河内总教区宗座署理；1994年被擢升为总主教，同年年底被擢升为枢机；2009年2月离世\n范廷颂于1919年6月15日在越南宁平省天主教发艳教区出生；童年时接受良好教育后，被一位越南神父带到河内继续其学业\n范廷颂于1940年在河内大修道院完成神学学业\n范廷颂于1949年6月6日在河内的主教座堂晋铎；及后被派到圣女小德兰孤儿院服务\n1950年代，范廷颂在河内堂区创建移民接待中心以收容到河内避战的难民\n1954年，法越战争结束，越南民主共和国建都河内，当时很多天主教神职人员逃至越南的南方，但范廷颂仍然留在河内\n翌年管理圣若望小修院；惟在1960年因捍卫修院的自由、自治及拒绝政府在修院设政治课的要求而被捕\n1963年4月5日，教宗任命范廷颂为天主教北宁教区主教，同年8月15日就任；其牧铭为「我信天主的爱」\n由于范廷颂被越南政府软禁差不多30年，因此他无法到所属堂区进行牧灵工作而专注研读等工作\n范廷颂除了面对战争、贫困、被当局迫害天主教会等问题外，也秘密恢复修院、创建女修会团体等\n1990年，教宗若望保禄二世在同年6月18日擢升范廷颂为天主教河内总教区宗座署理以填补该教区总主教的空缺\n1994年3月23日，范廷颂被教宗若望保禄二世擢升为天主教河内总教区总主教并兼天主教谅山教区宗座署理；同年11月26日，若望保禄二世擢升范廷颂为枢机\n范廷颂在1995年至2001年期间出任天主教越南主教团主席\n2003年4月26日，教宗若望保禄二世任命天主教谅山教区兼天主教高平教区吴光杰主教为天主教河内总教区署理主教；及至2005年2月19日，范廷颂因获批辞去总主教职务而荣休；吴光杰同日真除天主教河内总教区总主教职务\n范廷颂于2009年2月22日清晨在河内离世，享年89岁；其葬礼于同月26日上午在天主教河内总教区总主教座堂举行'
    print("*"*100) 
    print('原文', text001)
    print("*"*100) 
    print('句子选择结果')
    print(summary_api.predict(text001, topK=2))
    print("*"*100)
    
