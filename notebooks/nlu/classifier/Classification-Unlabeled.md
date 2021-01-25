
# Unbalance Classification Solutions

## Data Augement at Preprocessin

+ 见[workspace-of-preprocessing/data-augement](https://github.com/Apollo2Mars/Workspace-of-Preprocessing)

## change weight of loss

### weight loss

“”“
    class_weights = tf.constant([1.0, 10.0, 15.0, 1.0])
    self.loss = tf.nn.weighted_cross_entropy_with_logits(logits=tf.cast(logits, tf.float64), targets=tf.cast(self.input_y, tf.float64), pos_weight=tf.cast(class_weights, tf.float64))
    loss = tf.reduce_mean(self.loss)
”“”

### Focal Loss

+ <https://blog.csdn.net/u014535908/article/details/79035653>

### Learning weight

+ NIPS 2019 : Mata-weight-net
  + https://github.com/xjtushujun/Meta-weight-net_class-imbalance
  + https://github.com/xjtushujun/meta-weight-net

+ Learning to Reweight Examples for Robust Deep Learning
  + https://github.com/richardaecn/class-balanced-loss

+ CVPR 2019 :  Class-Balanced Loss Based on Effective Number of Samples
  + https://github.com/danieltan07/learning-to-reweight-examples

## EDA

+ https://towardsdatascience.com/these-are-the-easiest-data-augmentation-techniques-in-natural-language-processing-you-can-think-of-88e393fd610
+ https://arxiv.org/abs/1901.11196
+ https://github.com/jasonwei20/eda_nlp

## UDA

+ https://github.com/google-research/uda
+ https://github.com/google-research/bert
+ Unsupervised Data Augmentation for Consistency Training [pdf](https://arxiv.org/abs/1904.12848)

## 集成学习

### 有监督的集成学习

+ 使用采样的方法建立K个平衡的训练集，每个训练集单独训练一个分类器，对K个分类器取平均
+ 一般在这种情况下，每个平衡的训练集上都需要使用比较简单的分类器（why？？？）， 但是效果不稳定

### 半监督集成学习

+ https://www.zhihu.com/question/59236897

## 异常检测

### 无监督的异常检测

+ 从数据中找到异常值，比如找到spam
+ 前提假设是，spam 与正常的文章有很大不同，比如欧式空间的距离很大
+ 优势，不需要标注数据
  + https://www.zhihu.com/question/280696035/answer/417091151
  + https://zhuanlan.zhihu.com/p/37132428

### 结合 有监督集成学习 和 无监督异常检测 的思路

+ 在原始数据集上使用多个无监督异常方法来抽取数据的表示，并和原始的数据结合作为新的特征空间
+ 在新的特征空间上使用集成树模型，比如xgboost，来进行监督学习
+ 无监督异常检测的目的是提高原始数据的表达，监督集成树的目的是降低数据不平衡对于最终预测结果的影响。这个方法还可以和我上面提到的主动学习结合起来，进一步提升系统的性能
+ 运算开销比较大，需要进行深度优化。