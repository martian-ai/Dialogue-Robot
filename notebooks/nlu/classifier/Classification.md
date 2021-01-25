# Text Classification

## Solutions

| Model         | Tips                          |
| ------------- | ----------------------------- |
| KNN | |
| SVM | |
| Decision Tree and Ensemble Learning| |
|Navie Bayesian||
| Feature Engineer + NBSVM [paper](http://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf) [code](https://github.com/mesnilgr/nbsvm) | 可解释性 |
| topic model| 主题模型+短文本分类 <https://www.jiqizhixin.com/articles/2018-10-23-6> |
| TextCNN [paper](https://arxiv.org/abs/1408.5882) | 短文本                        |
| RNNs + Attention | 长文本                        |
| RCNN | Recurrent Convolutional Neural Networks for Text Classification|
| Fastext [website](https://fasttext.cc/) | 多类别，大数据量              |
| Capsule       | scalar to vector， 训练较慢   |
| Bert + NNs   | 效果最好， 模型较大，延时较长 |
| Seq2Seq with Attention |  |
| RCNN [paper](https://arxiv.org/abs/1609.04243) [code](https://github.com/jiangxinyang227/textClassifier) | RNN + Max-pooling 降维 |
| Transformer [paper](https://arxiv.org/abs/1706.03762) [code](https://github.com/jiangxinyang227/textClassifier) |                               |
| HAN [paper](https://www.aclweb.org/anthology/N16-1174) [code](https://github.com/lc222/HAN-text-classification-tf) | 层次注意力机制，长文本，{词向量, 句子向量， 文档向量} |
| Attention based CNN [paper](https://arxiv.org/pdf/1512.05193.pdf) |                               |
| DMN [paper](https://arxiv.org/pdf/1506.07285.pdf) [code](https://github.com/brightmart/text_classification) | Memory-Based |
| EntityNetwork [source-code](https://github.com/siddk/entity-network) [code](https://github.com/brightmart/text_classification) | Memory-Based |
| Adversial-LSTM [paper](https://arxiv.org/abs/1605.07725) [blog](https://www.cnblogs.com/jiangxinyang/p/10208363.html) | 对抗样本，正则化，避免过拟合 |
| VAT [paper](https://arxiv.org/abs/1605.07725) [blog](https://zhuanlan.zhihu.com/p/66389797) |  |

### TextCNN

![tcnn1.png](https://i.loli.net/2019/10/21/4o3k8W6h9XHCsUi.png)

### Capsule

+ 标量变向量
+ 前向传播
+ https://www.cnblogs.com/CZiFan/p/9803067.html

### Fasttext

+ 与word2vec 中的 cbow 结构类似
+ 区别
  + 输出层:为映射到具体类别，多分类的情况可能使用 层次softmax
  + 输入层:英文的情况是可能使用 char-level n-gram 特征记性词向量表示， 中文的情况是直接使用当前句子中每个词的词向量作为输入 (#CHECK)

![20200626211701](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/nlu/20200626211701.png)

## Metric

+ https://www.machinelearningplus.com/machine-learning/evaluation-metrics-classification-models-r/

### AUC_ROC

### mean Average Precesion （mAP）

+ 指的是在不同召回下的最大精确度的平均值

### Precision@Rank k

+ 假设共有*n*个点，假设其中*k*个点是少数样本时的Precision。这个评估方法在推荐系统中也常常会用

### confusion matrix

+ 观察混淆矩阵，找到需要重点增强的类别

## Application

### Intent Detection

### Sentiment Polarity Detection

### Anomaly Detection

+ Kaggle
+ [http://www.cnblogs.com/fengfenggirl/p/iForest.html](http://www.cnblogs.com/fengfenggirl/p/iForest.html)
+ https://github.com/yzhao062/anomaly-detection-resources

## Advance Research

+ 领域相关性研究
  + 跨领域时保持一定的分类能力
+ 数据不平衡研究
  + 有监督
    + 将多的类进行内部聚类
    + 在聚类后进行类内部层次采样，获得同少的类相同数据规模得样本
    + 使用采样样本，并结合类的中心向量构建新的向量，并进行学习
  + 不平衡数据的半监督问题
    + Heterogeneous Graph Attention Networks for Semi-supervised Short Text Classification
  + 不平衡数据的主动学习问题
  + 不平衡数据的无监督问题

### Unsupervised Classification
  + Step 1. self learning / co learning
  + Step 2. 聚类
  + Step 3. Transfer Learning
  + Step 4. Open-GPT Tranasformer

## Reference

## Papers

+ Pang, G., Cao, L., Chen, L. and Liu, H., 2018. Learning Representations of Ultrahigh-dimensional Data for Random Distance-based Outlier Detection. arXiv preprint arXiv:1806.04808.
    + 高维数据的半监督异常检测

+ Do we need hundreds of classifiers to solve real world classification problems.Fernández-Delgado, Manuel, et al. J. Mach. Learn. Res 15.1 (2014)
+ An empirical evaluation of supervised learning in high dimensions.Rich Caruana, Nikos Karampatziakis, and Ainur Yessenalina. ICML '08
+ Man vs. Machine: Practical Adversarial Detection of Malicious Crowdsourcing WorkersWang, G., Wang, T., Zheng, H., & Zhao, B. Y. Usenix Security'14
+ http://www.win-vector.com/dfiles/LogisticRegressionMaxEnt.pdf

## Links

- 各种机器学习的应用场景分别是什么？
  - https://www.zhihu.com/question/26726794
- 机器学习算法集锦：从贝叶斯到深度学习及各自优缺点
  - https://zhuanlan.zhihu.com/p/25327755
- 各种机器学习的应用场景分别是什么？例如，k近邻,贝叶斯，决策树，svm，逻辑斯蒂回归和最大熵模型
  - https://www.zhihu.com/question/26726794

+ 文本分类的整理
  + <https://zhuanlan.zhihu.com/p/34212945>

+ SGM:Sequence Generation Model for Multi-label Classification
  + Peking Unversity COLING 2018
  + **利用序列生成的方式进行多标签分类, 引入标签之间的相关性**

## Projects

- <https://github.com/jiangxinyang227/textClassifier>
- <https://github.com/brightmart/text_classification>
