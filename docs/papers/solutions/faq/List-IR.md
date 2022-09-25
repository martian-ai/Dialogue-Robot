# View IR

## query  
+ 查询、问询改写，错词纠正，同义词替换

## 召回
+ 召回率要很高，准确性可以比较低，可以召回不那么相关的信息

### 字面
+ 然而它的缺点也很明显，文本是具有语义的、是有语法结构的，倒排索引忽略了语句的语法结构，同时也无法解决一词多义和同义词的问题，也就它无法对 query 进行语义层面的召回
+ 第一种，基于词汇计数（Lexical term counting）的方法。大家都很熟悉这类方法，它基于字面匹配，好处在于很简单，对长尾的词有很好的鲁棒性，只要在标准问里有出现过，做匹配的时候一定可以召回。但是它的缺点很明显，它基于符号，没有语义层面的理解，所以很难处理字面不同语义相近的表述

### 潜语义
+ 常见的有LDA、LSI、PLSA等，这些都是基于概率和统计的算法，他们通过文档中词语的共现情况来对文档、词语进行向量化表达，能够做到一定的语义层面的相似度计算。而且也有开源工具来方便进行建模学习，以 LSI 模型为例，我们可以使用gensim 来对所有（question，answer）中的 question 进行主题建模，但是这面临着一个问题，即我们需要选择一个主题数量 k 作为训练参数，这个参数的选择完全看运气；另外这类模型对长尾分布的question不能很好的进行表示，在语义层面也只能做浅层的语义表达。LSI是个无监督算法，这是它的优势，我们通常将其作为文本分类或文本相似性任务中给数据打标签的辅助工具
+ 第二种，基于语言模型，主要思想是用概率的方法来判断知识库里面的 FAQ 和用户问询在哪一种在概率上更为接近。它的实战表现更好一些，但是它对语言模型参数的优化非常敏感，所以要做很多平滑实验。
+ 第三种，基于向量化的方法。我把用户的问询投射到这样的向量空间里去，把知识库的 FAQ 也投射到这样的向量空间里去，在向量空间里用距离的方法去做度量。目前存在很多种投射方案，比如基于矩阵的分解，可以把向量拿出来，还可以基于一些其他方法做向量化，向量空间算距离的时候也有很多种方法，比如用平均求和来算这两个点之间的距离
    + WMD 是 2015 年的工作，它用了一些更加新的方法来算这种距离，这样的方法比简单的平均化求距离要更好一些。但存在一个问题，这种方法对多义性的解决不太好

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
        + 
    + xapian
        + 

