# TextRank
## https://blog.csdn.net/asialee_bird/article/details/96894533
## PageRank算法根据网页之间的链接关系构造网络，TextRank算法根据词之间的共现关系构造网络；
## PageRank算法构造的网络中的边是有向无权边，TextRank算法构造的网络中的边是无向有权边。
# coding=utf-8
import jieba.analyse
import numpy as np
import pandas as pd
from textrank4zh import TextRank4Keyword, TextRank4Sentence


#关键词抽取
def keywords_extraction(text):
    tr4w = TextRank4Keyword(allow_speech_tags=['n', 'nr', 'nrfg', 'ns', 'nt', 'nz'])
    # allow_speech_tags   --词性列表，用于过滤某些词性的词
    tr4w.analyze(text=text, window=2, lower=True, vertex_source='all_filters', edge_source='no_stop_words',
                 pagerank_config={'alpha': 0.85, })
    # text    --  文本内容，字符串
    # window  --  窗口大小，int，用来构造单词之间的边。默认值为2
    # lower   --  是否将英文文本转换为小写，默认值为False
    # vertex_source  -- 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点
    #                -- 默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'
    # edge_source  -- 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来构造pagerank对应的图中的节点之间的边
    #              -- 默认值为`'no_stop_words'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。边的构造要结合`window`参数
 
    # pagerank_config  -- pagerank算法参数配置，阻尼系数为0.85
    keywords = tr4w.get_keywords(num=6, word_min_len=2)
    # num           --  返回关键词数量
    # word_min_len  --  词的最小长度，默认值为1
    return keywords
 
#关键短语抽取
def keyphrases_extraction(text):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=text, window=2, lower=True, vertex_source='all_filters', edge_source='no_stop_words',
                 pagerank_config={'alpha': 0.85, })
    keyphrases = tr4w.get_keyphrases(keywords_num=6, min_occur_num=1)
    # keywords_num    --  抽取的关键词数量
    # min_occur_num   --  关键短语在文中的最少出现次数
    return keyphrases
 
#关键句抽取
def keysentences_extraction(text, topK, minLen):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text, lower=True, source='all_filters')
    # text    -- 文本内容，字符串
    # lower   -- 是否将英文文本转换为小写，默认值为False
    # source  -- 选择使用words_no_filter, words_no_stop_words, words_all_filters中的哪一个来生成句子之间的相似度。
    # 		  -- 默认值为`'all_filters'`，可选值为`'no_filter', 'no_stop_words', 'all_filters'
    # sim_func -- 指定计算句子相似度的函数
 
    # 获取最重要的num个长度大于等于sentence_min_len的句子用来生成摘要
    keysentences = tr4s.get_key_sentences(num=topK, sentence_min_len=minLen)
    return keysentences
 
 
def keywords_textrank(text):
    keywords = jieba.analyse.textrank(text, topK=6)
    return keywords

if __name__ == '__main__':
    text = ''
    keywords=keywords_extraction(text)
    print(keywords)

    keyphrases=keyphrases_extraction(text)
    print(keyphrases)

    keysentences=keysentences_extraction(text)
    print(keysentences)