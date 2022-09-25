"""
基于主题模型的抽取式文本摘要
Abstract text summarization based on topic model
Topic model based text abstractor
"""

import pickle
import numpy as np


class AbstractSum(object):
    def __init__(self, tm, lda_path):
        self.tm = tm.load(lda_path)

    def summarize(self, lines, datas, judge_topic=False):
        """
        
        """
        DT  = self.tm.inference(datas, max_iter=10) # 显示前10条主题结果
        TD = DT.T
        # cluster_results = []
        # for topic, line in zip(topic_results, sentences):
        #     topic = topic.tolist()
        #     cluster_results.append((topic.index(max(topic)), line))
        # print(sorted(cluster_results, key=lambda k:k[0]))

        if judge_topic:
            ### 方案一, 获取最大那个主题的k个句子
            ##################################################################################
            topic_t_score = np.sum(TD, axis=-1)
            # 对每列(一个句子topic_num个主题),得分进行排序,0为最大
            res_nmf_h_soft = TD.argsort(axis=0)[-self.tm.n_topic:][::-1]
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
            res_nmf_h_soft_argmax = TD[topic_t_tc_argmax].tolist()
            res_combine = {}
            for l in range(len(lines)):
                res_combine[lines[l].strip()] = res_nmf_h_soft_argmax[l]
            score_sen = [(rc[1], rc[0]) for rc in sorted(res_combine.items(), key=lambda d: d[1], reverse=True)]
            #####################################################################################
        else:
            ### 方案二, 获取最大主题概率的句子, 不分主题
            res_combine = {}
            # print(sentences)
            for i in range(len(datas)):
                res_row_i = TD[:, i]
                res_row_i_argmax = np.argmax(res_row_i)
                res_combine[lines[i]] = res_row_i[res_row_i_argmax]
            score_sen = [(rc[1], rc[0]) for rc in sorted(res_combine.items(), key=lambda d: d[1], reverse=True)]
        # num_min = min(num, len(self.sentences))
        # return score_sen[0:num_min]
        return score_sen