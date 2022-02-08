# Paradigm

## Adversial
+ 见对应文件夹

## ensemble
+ 见对应文件夹

## reinforce
+ 见对应文件夹

## Unsupervised Learning
+ 聚类
+ 奇异值分解
+ 主成分分析
+ 潜在语义分析
+ 概率潜在语义分析
+ 马尔可夫蒙特卡洛方法
+ 潜在狄利克雷分配
+ PageRank


## Semi Supervised Learning

+ motivation
  + 无标签的样本虽然未直接包含标记信息，但如果他们与有标记得样本是从同样得数据源独立同分布采样而来，则他们所包含的关于数据分布得信息对建立模型将有很大益处
  + 类似聚类与流形学习，都是相似得样本有相似得输出

+ Theory Branch
+ 纯半监督 pure semi supervised learning
  + 开发世界假设
  + train(有标记数据 + 未标记数据A)， predict(预测数据B)
  + A is not B
+ 直推学习 transductive learning
  + 封闭世界假设
  + train(有标记数据 + 未标记数据C)， predict(未标记数据D)
  + C is D

+ Application
  + 半监督文本分类
    + https://zhuanlan.zhihu.com/p/45207077
    + https://www.jianshu.com/p/5882cffc3100
    + https://www.jianshu.com/p/a21815a81890


+ Solutions
  + active learning
    + <https://www.jiqizhixin.com/articles/2016-06-20-14>

  + co-training
    + Trick
      + 从Labeled 的数据中训练两个不同的分类器
      + 然后用这两个分类器对Unlabeled中的数据进行分类
      + 把可信度最高的加入到Labeled中
      + 循环直到U中没有数据或者达到循环最大次数

    + Tri-Train
      + 使用三个分类器，对一个无标签样本，如果有两个分类器的判断一致，则将该样本进行标记
      + 将其纳入另一个分类器的训练样本
      + 如此重复迭代，直到所有样本都被标记或者三个分类器不再有变化
      + self-learning

  + self learning 
    + https://blog.csdn.net/lujiandong-1/article/details/52596654 
    + https://zhuanlan.zhihu.com/p/45207077

    + Trick
      + 两个样本集，Labeled 和 Unlabeled， 执行算法如下
      + 使用Labeled，生成分类策略F
      + 用F分类Unlabeled样本，计算误差
      + 选取Unlabeled 中误差小的子集u，加入到Labeled的集合

  + shortage
    + self-training 和 co-training 都是获取正样本，需要获取负样本的方式
    + 常规方法 ： 非正样本随机抽取， 但效果不好
    + Pu-Learning
    ![](http://www.flickering.cn/wp-content/uploads/2013/01/training_data_acquisition1.png)
    + 随着训练的进行，自动标记的样本产生的噪声会不断积累

## MulitTask Learning

+ Define
  + 只要有多个loss 就叫MTL
  + 别名(joint learning, learning to learn, learning with auxiliary task)
  + 通过权衡主任务与辅助的相关任务中的训练信息来提升模型的**泛化性**与表现
  + 从机器学习的视角来看，**MTL可以看作一种inductive transfer**（先验知识），**通过提供inductive bias**（某种对模型的先验假设）**来提升模型效果**。比如，**使用L1正则，我们对模型的假设模型偏向于sparse solution**（参数要少）
  + **在MTL中，这种先验是通过auxiliary task来提供**，更灵活，告诉模型偏向一些其他任务，最终导致模型会泛化得更好

+ View
  + [ A survey on multi-task learning](https://arxiv.org/pdf/1707.08114.pdf)
  + [Learing t Multitask]([http://papers.nips.cc/paper/7819-learning-to-multitask.pdf](http://papers.nips.cc/paper/7819-learning-to-multitask.pdf))
  + Feature-based approach
    + Use data features as the media to share knowledge among task and it usually learns a common feature representaion for all tasks
    + Two categories
      + Shallow approach
      + deep approach
    + Parameter-based approach
      + Links different task by placing regularizers or Bayesian priors on model parameters to achieve knowledge transfer among tasks
      + Four catefories
        + Low-rank approach
        + task clustering approach
        + Task relation learning approach
        + Decomposition approach

  + Unified Formulation for Multitask Learning
    + ![WX20190513-180908](https://ws4.sinaimg.cn/large/006tNc79ly1g2zupspcljj30wt0h6qbl.jpg)

  + [An Overview of Multi-Task Learningin Deep Neural Networks](https://arxiv.org/pdf/1706.05098.pdf)
    + Two MLT methods for Deep Learning
      + Hard parameter shareing
        ![](https://ws2.sinaimg.cn/large/006tNc79ly1g2zwn1rd9qj30m00fawes.jpg)
      + Soft parameter sharing
        ![](https://ws1.sinaimg.cn/large/006tNc79ly1g2zwnkeolzj30ws0cwglz.jpg)
      + Why work ?
        + Implicit data augmentation
        + Attention focusing
        + Eavesdropping
        + Representation bias
        + Regularization

  + Weighted Multitask Learning
    + 一般方法是各个任务进行
      + 加权求和
      + 统一优化
    + 各个任务之间互不竞争，各个任务重复优化，但是多任务学习容易造成某些任务占主导作用，其他任务无法充分优化
      + [multi-task learning using uncertainty to weight loss for scene geometry and semantics]()
        + 在贝叶斯建模中，主要为了解决两种不确定性：认知的不确定与偶然的不确定性
        + 认知的不确定主要由数据不足产生，因此提供更多的训练数据
        + 偶然的不确定性是不能仅从数据看到解释，又分为两类：
          + 数据依赖型（不同方差）
            + 数据依赖型是说相同输入可能会产生不同输出
          + 任务依赖型（相同方差）
            + 任务依赖型是对相同输入会有相同输出，但对不同任务却有不同的输出
        + 该论文主要从任务依赖型的不确定性角度解决多任务学习的权重问题
          ![](https://ws1.sinaimg.cn/large/006tNc79ly1g2zw4b9cesj30ar0luad1.jpg)
      + [Multi-Task Learning as Multi-Objective Optimization]([http://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf](http://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf))
        + NIPS 2018 Intel Lab
        + 多任务学习中，如何使某些任务尽可能优化，但其他任务却不受影响，这是一个帕累托最优的过程(多目标优化的问题)


## Transformer Learning

+ Multi-Source Domain Adaptation with Mixture of Experts
  + MIT EMNLP2018
  + 多源迁移学习的无监督训练方法
  + 通过将所有source分为meta-source 和 meta-target 自动构建训练集
  + 显示的学习 source set 和 target example 之间的匹配度
  + 很好的避免了negative transfer

+  Fast.ai推出NLP最新迁移学习方法「微调语言模型」，可将误差减少超过20%！
  + Fine-tuned Language Models for Text Classification

+ https://www.jiqizhixin.com/articles/2017-06-23-5)

+ A comprehensive Survey on Transfer Learning
   + 中科院，40种迁移学习方法


## Meta Learning

## Online Learning



