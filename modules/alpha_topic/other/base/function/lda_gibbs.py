from __future__ import print_function
import time
import pickle

import numpy as np
from scipy.special import gammaln
from six.moves import xrange

from .base import BaseGibbsParamTopicModel
from .formatted_logger import formatted_logger
from .utils import sampling_from_dist

logger = formatted_logger('GibbsLDA')


class GibbsLDA(BaseGibbsParamTopicModel):
    """
    Latent dirichlet allocation,
    Blei, David M and Ng, Andrew Y and Jordan, Michael I, 2003
    
    Latent Dirichlet allocation with collapsed Gibbs sampling

    Attributes
    ----------
    topic_assignment:
        list of topic assignment for each word token

    """

    def __init__(self, n_doc=10, n_voca=1000, n_topic=10, alpha=0.1, beta=0.01, **kwargs):
        super(GibbsLDA, self).__init__(n_doc=n_doc, n_voca=n_voca, n_topic=n_topic, alpha=alpha, beta=beta, **kwargs)

    def random_init(self, docs):
        """

        Parameters
        ----------
        docs: list, size=n_doc

        """
        for di in range(len(docs)):
            doc = docs[di]
            topics = np.random.randint(self.n_topic, size=len(doc)) # 产生长度为 len(doc) 的主题数组 [2, 4, 0,...., 1], 为每个词找一个随机的主题
            self.topic_assignment.append(topics)
            # 统计 TW, sum_T, DT 的随机初始值
            for wi in range(len(doc)): # 遍历当前文档的所有词
                topic = topics[wi] # 当前词的主题
                word = doc[wi] # 当前词
                self.TW[topic, word] += 1 # 主题topic和词word的共现关系+1
                self.sum_T[topic] += 1 # 主题topic 出现次数+1
                self.DT[di, topic] += 1 # 文档di 与 主题的共现关系+1

    def fit(self, docs, max_iter=100):
        """ Gibbs sampling for LDA

        Parameters
        ----------
        docs
        max_iter: int
            maximum number of Gibbs sampling iteration

        """
        self.random_init(docs)

        for iteration in xrange(max_iter):
            #prev = time.clock()
            prev = time.process_time()

            for di in xrange(len(docs)):
                doc = docs[di]
                for wi in xrange(len(doc)):
                    word = doc[wi]
                    old_topic = self.topic_assignment[di][wi]
                    self.TW[old_topic, word] -= 1
                    self.sum_T[old_topic] -= 1
                    self.DT[di, old_topic] -= 1
                    # compute conditional probability of a topic of current word wi
                    prob = (self.TW[:, word] / self.sum_T) * (self.DT[di, :]) # 得到先验分布
                    new_topic = sampling_from_dist(prob)
                    self.topic_assignment[di][wi] = new_topic
                    self.TW[new_topic, word] += 1 
                    self.sum_T[new_topic] += 1  
                    self.DT[di, new_topic] += 1
        
            if self.verbose:
                logger.info('[ITER] %d,\telapsed time:%.2f,\tlog_likelihood:%.2f', iteration, time.process_time() - prev, self.log_likelihood(docs))

    def log_likelihood(self, docs):
        """
        likelihood function
        评估主题模型的效果
        """
        ll = len(docs) * gammaln(self.alpha * self.n_topic) # Gamma函数绝对值的对数
        ll -= len(docs) * self.n_topic * gammaln(self.alpha)
        ll += self.n_topic * gammaln(self.beta * self.n_voca)
        ll -= self.n_topic * self.n_voca * gammaln(self.beta)

        for di in xrange(len(docs)):
            ll += gammaln(self.DT[di, :]).sum() - gammaln(self.DT[di, :].sum())
        for ki in xrange(self.n_topic):
            ll += gammaln(self.TW[ki, :]).sum() - gammaln(self.TW[ki, :].sum())

        return ll
    
    def inference(self, docs, max_iter=10):
        """
        有了 LDA 的模型，对于新来的文档 doc, 我们只要认为 Gibbs Sampling 公式中的 主题-词 的分布  是稳定不变的，是由训练语料得到的模型提供的，所以采样过程中我们只要估计该文档的 topic (文档-主题分布)分布 就好了. 具体算法如下：
        1. 对当前文档中的每个单词w, 随机初始化一个topic编号z;
        2. 使用Gibbs Sampling公式，对每个词w, 重新采样其topic；
        3. 重复以上过程，知道Gibbs Sampling收敛；
        4. 统计文档中的topic分布，该分布就是 \vec{\theta}
        """

        topic_assignment = [] # 所有文档的主题分布
        DT = np.zeros([len(docs), self.n_topic]) 
        sum_T = np.zeros(self.n_topic) + 1 
        DT += self.alpha
        sum_T += self.beta * self.n_voca

        for di in range(len(docs)):
            doc = docs[di]
            topics = np.random.randint(self.n_topic, size=len(doc)) # 产生长度为 len(doc) 的主题数组 [2, 4, 0,...., 1], 为每个词找一个随机的主题
            topic_assignment.append(topics)
            for wi in range(len(doc)): # 遍历当前文档的所有词
                topic = topics[wi] # 当前词的主题
                word = doc[wi] # 当前词
                # self.TW[topic, word] += 1 # 主题topic和词word的共现关系+1
                sum_T[topic] += 1 # 主题topic 出现次数+1
                DT[di, topic] += 1 # 文档di 与 主题的共现关系+1\

        for iteration in xrange(max_iter):
            for di in xrange(len(docs)):
                doc = docs[di]
                # print('doc', doc)
                for wi in xrange(len(doc)):
                    word = doc[wi]
                    old_topic = topic_assignment[di][wi]
                    # print('wrod', word)
                    # print('old topic', old_topic)
                    sum_T[old_topic] -= 1
                    DT[di, old_topic] -= 1
                    # print(self.TW[:, word])
                    # print(sum_T)
                    # print(DT[di, :])
                    # print(self.TW[:, word])
                    prob = (self.TW[:, word] / sum_T) * (DT[di, :]) # 得到先验分布
                    new_topic = sampling_from_dist(prob)
                    # print('new topic', new_topic)
                    topic_assignment[di][wi] = new_topic
                    sum_T[new_topic] += 1  
                    DT[di, new_topic] += 1
            # print(DT)
        return DT

    def save(self, fname):
        with open(fname, mode='wb') as f:
            pickle.dump(self, f)
            
    def load(self, fname):
        with open(fname, mode='rb') as f:
            return pickle.load(f)