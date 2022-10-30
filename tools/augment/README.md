
# Data Augmentation

## Invotation

- NLP领域的数据属于离散数据, 小的扰动会改变含义，因此可以用数据增强的方式增强模型的泛化能力

- 常见的问题
  - Imbalance Label
  - Corrupter Label
  - Imbalance Diversity (数据多样性不平衡)

## Supervised Solutions

### 语义相似词替换

- 随机的选一些词并用它们的同义词来替换这些词, 例如，将句子“我非常喜欢这部电影”改为“我非常喜欢这个影片”，这样句子仍具有相同的含义，很有可能具有相同的标签
- 但这种方法可能没什么用，因为同义词具有非常相似的词向量，因此模型会将这两个句子当作相同的句子，而在实际上并没有对数据集进行扩充。

### EDA

- Random Insertion
- Random Swap
  - code：<https://github.com/dupanfei1/deeplearning-util/blob/master/nlp/augment.py>
- Random  Deletion
  - code：<https://github.com/dupanfei1/deeplearning-util/blob/master/nlp/augment.py>

### 回译(相对好用)

- 用机器翻译把一段英语翻译成另一种语言，然后再翻译回英语。
- 已经成功的被用在Kaggle恶意评论分类竞赛中
- 反向翻译是NLP在机器翻译中经常使用的一个数据增强的方法， 其本质就是快速产生一些不那么准确的翻译结果达到增加数据的目的
- 例如，如果我们把“I like this movie very much”翻译成俄语，就会得到“Мне очень нравится этот фильм”，当我们再译回英语就会得到“I really like this movie” ，回译的方法不仅有类似同义词替换的能力，它还具有在保持原意的前提下增加或移除单词并重新组织句子的能力
- 参考：https://github.com/dupanfei1/deeplearning-util/tree/master/nlp

### 文档剪辑（长文本）

- 新闻文章通常很长，在查看数据时，对于分类来说并不需要整篇文章。 文章的主要想法通常会重复出现。将文章裁剪为几个子文章来实现数据增强，这样将获得更多的数据

### 文本生成

- 生成文本
- [Data Augemtation Generative Adversarial Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1711.04340)
- [Triple Generative Adversarial Nets](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1703.02291)
- [Semi-Supervised QA with Generative Domain-Adaptive Nets](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1702.02206)
- ULMFIT
- Open-AI transformer
- BERT

### 文本更正

- 中文如果是正常的文本多数都不涉及，但是很多恶意的文本，里面会有大量非法字符，比如在正常的词语中间插入特殊符号，倒序，全半角等。还有一些奇怪的字符，就可能需要你自己维护一个转换表了
- 文本泛化
- 表情符号、数字、人名、地址、网址、命名实体等，用关键字替代就行。这个视具体的任务，可能还得往下细化。比如数字也分很多种，普通数字，手机号码，座机号码，热线号码、银行卡号，QQ号，微信号，金钱，距离等等，并且很多任务中，这些还可以单独作为一维特征。还得考虑中文数字阿拉伯数字等
- 中文将字转换成拼音，许多恶意文本中会用同音字替代
- 如果是英文的，那可能还得做词干提取、形态还原等，比如fucking,fucked -> fuck
- 去停用词

### 基于上下文的数据增强

- 方法论文：Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations
- 方法实现代码：使用双向循环神经网络进行数据增强
- 该方法目前针对于英文数据进行增强，实验工具：spacy（NLP自然语言工具包）和chainer（深度学习框架）

### 扩句-缩句-句法

- 1.句子缩写，查找句子主谓宾等
- 有个java的项目，调用斯坦福分词工具(不爱用)，查找主谓宾的
- 地址为:（主谓宾提取器）hankcs/MainPartExtractor
- 2.句子扩写
- 3.句法

### HMM-marko（质量较差）

- HMM生成句子原理: 根据语料构建状态转移矩阵，jieba等提取关键词开头，生成句子
- 参考项目:takeToDreamLand/SentenceGenerate_byMarkov

## Semi-Supervised Solution

- 对于分类任务，将有标签数据和无标签数据分别进行向量表示，对有标签数据进行聚类
- 对每一个无标签数据，将其标记为与之最近的聚类中心所表示的类别(该聚类中所有样本中最多的类别表示该聚类的类别)

## Metrics

- 判断两个样本集的分布是否一致
- 判断生成后的数据与原始数据的分布是否一致

## Reference

### Links

- https://www.reddit.com/r/MachineLearning/comments/12evgi/classification_when_80_of_my_training_set_is_of/
- https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
- 知乎:中文自然语言处理中的数据增强方式
    - https://www.zhihu.com/question/305256736
- https://zhuanlan.zhihu.com/p/112877845

### Tools

- SMOTE
- imblance learn
- scikit learn
- synonyms
