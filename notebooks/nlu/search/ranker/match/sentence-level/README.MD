# Sentence Similarity

### Metric

+ 衡量句子相似性的指标
+ 数据集



### Reference

+ https://www.zhihu.com/question/29978268/answer/156714617

+ Arora, Sanjeev, et al. "A latent variable model approach to pmi-based word embeddings." *Transactions of the Association for Computational Linguistics* 4 (2016): 385-399.

+ A Simple but Tough-to-Beat Baseline for Sentence Embeddings

  + https://github.com/PrincetonML/SIF


# Embedding 

### Paper : A simple but tough-to-beat baseline for sentence embeddings

![](https://pic4.zhimg.com/80/v2-329aa77939ef2376aaa68ca11318ae9b_hd.png)

![](https://pic2.zhimg.com/80/v2-376f7539767aac571600ea2a780022b1_hd.png)

+ weighted average word embedding
  + weight = a/(a+p(w))
+ PCA
+ 无监督方法
  + 对句子中所有词的word vector求平均，获得sentence embedding
  + 以每个词的tf-idf为权重，对所有词的word vector加权平均，获得sentence embedding
  + 以smooth inverse frequency[1]（简称SIF)为权重，对所有词的word vector加权平均，最后从中减掉principal component，得到sentence embedding
  + 通过Word Mover’s Distance[2]（简称WMD），直接度量句子之间的相似度
+ 有监督
  + 分类任务，例如训练一个CNN的文本分类器[3]，取最后一个hidden layer的输出作为sentence embedding，其实就是取分类器的前几层作为预训练的encoder
  + sentence pair的等价性/等义性判定（[4][5]），这种方法的好处是不仅可以得到sentence embedding，还可以直接学习到距离度量函数里的参数

+ Yves Peirsman的这篇博客[6]里比较了常见方法在计算句子相似句上的效果：

  + ![](https://pic3.zhimg.com/80/v2-5c14c164e4c718720b12e5cfdcad6cc6_hd.jpg)

  + 从图中可以看到在这几个评估数据上：

    1）WMD方法（WMD-W2V）相对于其他无监督方法并没有明显的优势

    2）简单加权的词向量平均（AVG-W2V和AVG-W2V-TFIDF）已经可以作为一个较好的baseline，但是考虑SIF加权的词向量平均（SIF-W2V)通常可以得到更好的效果

    3）这里比较了两种预训练的encoder（InferSent/INF和Google's Sentence Encoder/GSE），相对而言GSE效果更好一些，但要注意它的性能并不一定在所有的数据集上都稳定

    另外，从实践中的经验来看，如果要在无标注的数据上从零开始实现相似度的计算，可以综合几种方法来得到更好的效果。一种可能的步骤如下：

    1）使用某种无监督方法，对于句子集合做简单归类和标注

    2）通过1中得到的标注数据，训练分类模型或者句子对模型，从而得到相似度模型

    3）评估模型效果，适当引入新的标注数据，重复步骤1）2）

    Reference：

    [1] Sanjeev Arora, et al. 2017. A Simple but Tough-to-Beat Baseline for Sentence Embeddings

    [2] Matt J. Kusner, et al. 2015. From Word Embeddings To Document Distances

    [3] Yoon Kim. 2014. Convolutional Neural Networks for Sentence Classification

    [4] Jonas Mueller, et al. 2016. Siamese Recurrent Architectures for Learning Sentence Similarity

    [5] Paul Neculoiu, et al. 2016. Learning Text Similarity with Siamese Recurrent Networks

    [6] Yves Peirsman. 2018. Comparing Sentence Similarity Methods 

# LDA + RNN Language Model



# From Word Embeddings To Document Distances

+ Word Mover’s Distance

+ 文章里面定义了Word Mover’s Distance，词-词 的相似度用word2vec结果算欧式距离，句子-句子 的相似度通过求解一个transportation的优化问题得到。文章里面证明了词向量求平均算欧式距离是Word Mover’s Distance的下界，这样在求解最近邻集合的时候，就可以先用词向量平均对候选集快速进行pruning


