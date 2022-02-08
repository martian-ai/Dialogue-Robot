import re
import jieba 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

def cut_sentence(sentence):
    """
        分句
    :param sentence:str
    :return:list
    """
    re_sen = re.compile('[:;!?。：；？！\n\r]') #.不加是因为不确定.是小数还是英文句号(中文省略号......)
    sentences = re_sen.split(sentence)
    sen_cuts = []
    for sen in sentences:
        if sen and str(sen).strip():
            sen_cuts.append(sen)
    return sen_cuts

class LDASum:
    def __init__(self):
        self.stop_words = []
        # self.stop_words = stop_words.values()
        self.algorithm = 'lda'

    def summarize(self, text, num=100, topic_min=3, judge_topic=None):
        """
        :param text: str
        :param num: int
        :return: list
        """
        # 切句
        if type(text) == str:
            self.sentences = cut_sentence(text)
        elif type(text) == list:
            self.sentences = text
        else:
            raise RuntimeError("text type must be list or str")
        len_sentences_cut = len(self.sentences)
        # 切词
        sentences_cut = [[word for word in list(jieba.cut(sentence)) if word.strip()] for sentence in self.sentences]
        # sentences_cut = [[word for word in jieba_cut(extract_chinese(sentence)) if word.strip()] for sentence in self.sentences]
        # 去除停用词等
        self.sentences_cut = [list(filter(lambda x: x not in self.stop_words, sc)) for sc in sentences_cut]
        self.sentences_cut = [" ".join(sc) for sc in self.sentences_cut]
        print(sentences_cut)
        # 计算每个句子的tf
        vector_c = CountVectorizer(ngram_range=(1, 2), stop_words=self.stop_words)
        tf_ngram = vector_c.fit_transform(self.sentences_cut)
        # 主题数, 经验判断
        topic_num = min(topic_min, int(len(sentences_cut) / 2))  # 设定最小主题数为3
        print('topic_num', topic_num)
        lda = LatentDirichletAllocation(n_components=topic_num, max_iter=32,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=2019)
        res_lda_u = lda.fit_transform(tf_ngram.T)
        res_lda_v = lda.components_
        print('res_lda_v', res_lda_v) # 各个主题在各个文档上分配的概率

        if judge_topic:
            ### 方案一, 获取最大那个主题的k个句子
            ##################################################################################
            topic_t_score = np.sum(res_lda_v, axis=-1)
            print('topic_t_score', topic_t_score)
            # 对每列(一个句子topic_num个主题),得分进行排序,0为最大
            res_nmf_h_soft = res_lda_v.argsort(axis=0)[-topic_num:][::-1]
            # 统计为最大每个主题的句子个数
            exist = (res_nmf_h_soft <= 0) * 1.0
            factor = np.ones(res_nmf_h_soft.shape[1])
            topic_t_count = np.dot(exist, factor)
            # 标准化
            topic_t_count /= np.sum(topic_t_count, axis=-1)
            topic_t_score /= np.sum(topic_t_score, axis=-1)
            # 主题最大个数占比, 与主题总得分占比选择最大的主题
            topic_t_tc = topic_t_count + topic_t_score
            topic_t_tc_argmax = np.argmax(topic_t_tc)
            # 最后得分选择该最大主题的
            res_nmf_h_soft_argmax = res_lda_v[topic_t_tc_argmax].tolist()
            res_combine = {}
            for l in range(len_sentences_cut):
                res_combine[self.sentences[l]] = res_nmf_h_soft_argmax[l]
            score_sen = [(rc[1], rc[0]) for rc in sorted(res_combine.items(), key=lambda d: d[1], reverse=True)]
            #####################################################################################
        else:
            ### 方案二, 获取最大主题概率的句子, 不分主题
            res_combine = {}
            for i in range(len_sentences_cut):
                res_row_i = res_lda_v[:, i]
                res_row_i_argmax = np.argmax(res_row_i)
                res_combine[self.sentences[i]] = res_row_i[res_row_i_argmax]
            score_sen = [(rc[1], rc[0]) for rc in sorted(res_combine.items(), key=lambda d: d[1], reverse=True)]
        num_min = min(num, len(self.sentences))
        return score_sen[0:num_min]


if __name__ == '__main__':
    lda = LDASum()

    doc = "PageRank算法简介。" \
           "是上世纪90年代末提出的一种计算网页权重的算法! " \
           "当时，互联网技术突飞猛进，各种网页网站爆炸式增长。 " \
           "业界急需一种相对比较准确的网页重要性计算方法。 " \
           "是人们能够从海量互联网世界中找出自己需要的信息。 " \
           "Google把从A页面到B页面的链接解释为A页面给B页面投票。 " \
           "Google根据投票来源甚至来源的来源，即链接到A页面的页面。 " \
           "一个高等级的页面可以使其他低等级页面的等级提升。 " \
           "具体说来就是，PageRank有两个基本思想，也可以说是假设。 " \
           "总的来说就是一句话，从全局角度考虑，获取重要的信。 " 

    sum = lda.summarize(doc)
    for i in sum:
        print(i)