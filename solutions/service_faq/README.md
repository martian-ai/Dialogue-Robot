# Frequently Asked Questions

## Introduction

+ 问答系统相关介绍参考 Q&A System.md

## Framwork

### 问题改写

+ query 改写

  + retall
  + 同义词
  + NMT
  + RL
  + Doc2Query
  + GAN
+ 错词纠正
+ 同义词替换

### 召回

+ 目标：召回率要高，准确率可以比较低，可以召回不重要的信息
+ 字面召回
  + 分词
  + 停用词
  + 核心词

    + ~~最简单的实现方式是用 IDF 逆文档频率算出~~
      + ~~即看这个词在语料里的出现次数，如果次数较少，则认为其包含的信息量是相对较多~~
      + ~~这个方法的局限是无法根据 query 上下文语境动态适应，因为其在不同的上下文中 weight 是一致的~~
    + ~~通过统计点击数据中进行调整~~
      + ~~如果用户的 query 包含 a，b 两个 term，并且点击了 Doc1 和 Doc4，其中 Doc1 包含 a，b， Doc4 只包含a，即 a 出现2次，b 出现1次。一个朴素的想法就是 a 的权重就是1，b 是0.5，这样就可以从历史日志里把每个 query 里的 term 权重统计出来~~
  + 同义词
  + 多词搜索
+ 语义召回
  + 主题模型
  + 深度模型
    + dssm
    + sentence bert
  + 向量召回
    + faiss
    + annoy
    + Proxima
    + ES 7.x dense vector

第一种，基于词汇计数（Lexical term counting）的方法。大家都很熟悉这类方法，它基于字面匹配，好处在于很简单，对长尾的词有很好的鲁棒性，只要在标准问里有出现过，做匹配的时候一定可以召回。但是它的缺点很明显，它基于符号，没有语义层面的理解，所以很难处理字面不同语义相近的表述。

第二种，基于语言模型，主要思想是用概率的方法来判断知识库里面的 FAQ 和用户问询在哪一种在概率上更为接近。它的实战表现更好一些，但是它对语言模型参数的优化非常敏感，所以要做很多平滑实验。

第三种，基于向量化的方法。我把用户的问询投射到这样的向量空间里去，把知识库的 FAQ 也投射到这样的向量空间里去，在向量空间里用距离的方法去做度量。目前存在很多种投射方案，比如基于矩阵的分解，可以把向量拿出来，还可以基于一些其他方法做向量化，向量空间算距离的时候也有很多种方法，比如用平均求和来算这两个点之间的距离。

## 匹配

+ 参考 Match.md
+ 字面
  + jaccard
    + n-gram
    + word
    + jaccard + word2vec
      + https://www.jsjkx.com/EN/article/openArticlePDF.jsp?id=198
    + 广义
      + https://blog.csdn.net/xceman1997/article/details/8600277
    + weighted
  + LCS
  + common word
+ 语义
  + representation-based method
  + interaction-based method
    + match pramid
  + BERT-Flow
  + SimBERT
  + ConsBERT
  + ESIM
  + 模型小型化
    + ernie-tiny
    + tiny-bert
    + albert_tiny
      + 「模型介绍」模型albert_tiny，参数量仅为1.8M，是bert的六十分之一；模型大小仅16M，是二十五分之一；训练和预测提速约10倍；序列长度64的分类任务上，单核cpu的每秒处理能力即qps达到20
      + https://zhuanlan.zhihu.com/p/87517511

## 重排序

+ 参考ReRank.md

## 数据集

+ FAQIR (Karan and Snajder, 2016)
+ StackFAQ (Karan and Snajder , 2018).12
+ 数据集构建
  + Embedding-based Retrieval in Facebook Search
  + https://zhuanlan.zhihu.com/p/152570715

## 评测方式

## 优缺点

+ 基于检索的模型不会产生新的回答，只能从预先定义的“回答集”中挑选出一个较为合适的回答。
+ 缺点

  + 检索式对话系统不会产生新的回复，其能够回复类型与内容都由语料库所决定。一旦用户的问话超脱了语料库的范围，那么对话系统将无法准确回答用户
+ 优点

  + 相对严谨可靠，可控性强，不会回复令人厌恶或违法法规的文本。

## 依赖

+ 倒排索引
+ learn to rank
+ 集成学习/xgboost

## REF

+ https://www.its203.com/article/Ding_xiaofei/81557004
+ 腾讯知文问答系统




# View IR

## query

+ 查询、问询改写，错词纠正，同义词替换

## 召回

+ 召回率要很高，准确性可以比较低，可以召回不那么相关的信息

### 字面

+ 然而它的缺点也很明显，文本是具有语义的、是有语法结构的，倒排索引忽略了语句的语法结构，同时也无法解决一词多义和同义词的问题，也就它无法对 query 进行语义层面的召回
+ 第一种，基于词汇计数（Lexical term counting）的方法。大家都很熟悉这类方法，它基于字面匹配，好处在于很简单，对长尾的词有很好的鲁棒性，只要在标准问里有出现过，做匹配的时候一定可以召回。但是它的缺点很明显，它基于符号，没有语义层面的理解，所以很难处理字面不同语义相近的表述。

### 潜语义

+ 常见的有LDA、LSI、PLSA等，这些都是基于概率和统计的算法，他们通过文档中词语的共现情况来对文档、词语进行向量化表达，能够做到一定的语义层面的相似度计算。而且也有开源工具来方便进行建模学习，以 LSI 模型为例，我们可以使用gensim 来对所有（question，answer）中的 question 进行主题建模，但是这面临着一个问题，即我们需要选择一个主题数量 k 作为训练参数，这个参数的选择完全看运气；另外这类模型对长尾分布的question不能很好的进行表示，在语义层面也只能做浅层的语义表达。LSI是个无监督算法，这是它的优势，我们通常将其作为文本分类或文本相似性任务中给数据打标签的辅助工具
+ 第二种，基于语言模型，主要思想是用概率的方法来判断知识库里面的 FAQ 和用户问询在哪一种在概率上更为接近。它的实战表现更好一些，但是它对语言模型参数的优化非常敏感，所以要做很多平滑实验。
+ 第三种，基于向量化的方法。我把用户的问询投射到这样的向量空间里去，把知识库的 FAQ 也投射到这样的向量空间里去，在向量空间里用距离的方法去做度量。目前存在很多种投射方案，比如基于矩阵的分解，可以把向量拿出来，还可以基于一些其他方法做向量化，向量空间算距离的时候也有很多种方法，比如用平均求和来算这两个点之间的距离。
  + WMD 是 2015 年的工作，它用了一些更加新的方法来算这种距离，这样的方法比简单的平均化求距离要更好一些。但存在一个问题，这种方法对多义性的解决不太好。

### 深度语义

## 排序

### Match

### Ranking

+ https://blog.csdn.net/pearl8899/article/details/102920628

# Ref

+ [基于向量的深层语义相似文本召回？你需要bert和faiss](https://zhuanlan.zhihu.com/p/197708027)
+ [zhiyi simbert](https://github.com/ZhuiyiTechnology/simbert)
+ [鱼与熊掌兼得：融合检索和生成的SimBERT模型](https://kexue.fm/archives/7427)
+ [如何构建一个问答机器人（FAQ问答机器人）](https://blog.csdn.net/Ding_xiaofei/article/details/81557004)

# Data IR

## 召回数据集的构建

+ 我们尝试构建一个通用领域的召回数据集，格式为（q，d+，d-）的三元组
  + 这里我们借鉴了论文Embedding-based Retrieval in Facebook Search中的思路，d-负样本包括easy和hard两类
  + https://zhuanlan.zhihu.com/p/349993294
  + https://aistudio.baidu.com/aistudio/competition/detail/157/0/datasets

# Plan IR

## 三路召回

+ query
+ query rewrite
+ embedding
  + 基于embedding的检索，已经有很多成熟的方案，包括：Annoy、Faiss、Elasticsearch (dense_vector)等
  + ES
    + https://link.zhihu.com/?target=https%3A//www.elastic.co/cn/blog/text-similarity-search-with-vectors-in-elasticsearch
  + sphinx
  + xapian
