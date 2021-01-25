
# ERNIE - 百度

- https://zhuanlan.zhihu.com/p/76757794
- https://cloud.tencent.com/developer/article/1495731

# Ernie 1.0
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/f08f9ca48196c3b9bd23279ee6f219c2.png)

[ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/pdf/1904.09223) 是百度在2019年4月的时候，基于BERT模型，做的进一步的优化，在中文的NLP任务上得到了state-of-the-art的结果。它主要的改进是在mask的机制上做了改进，它的mask不是基本的word piece的mask，而是在pretrainning阶段增加了外部的知识，由三种level的mask组成，分别是basic-level masking（word piece）+ phrase level masking（WWM style） + entity level masking。在这个基础上，借助百度在中文的社区的强大能力，中文的ernie还是用了各种异质(Heterogeneous)的数据集。此外为了适应多轮的贴吧数据，所有ERNIE引入了DLM (Dialogue Language Model) task。

百度的论文看着写得不错，也很简单，而且改进的思路是后来各种改进模型的基础。例如说Masking方式的改进，让BERT出现了WWM的版本，对应的中文版本（[Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/pdf/1906.08101)），以及 [facebook的SpanBERT](https://arxiv.org/pdf/1907.10529)等都是主要基于masking方式的改进。

但是不足的是，因为baidu ernie1.0只是针对中文的优化，导致比较少收到国外学者的关注，另外百度使用的是自家的paddle paddle机器学习框架，与业界主流tensorflow或者pytorch不同，导致受关注点比较少。

## Knowlege Masking
intuition:
模型在预测未知词的时候，没有考虑到外部知识。但是如果我们在mask的时候，加入了外部的知识，模型可以获得更可靠的语言表示。
>例如：
哈利波特是J.K.罗琳写的小说。
单独预测 `哈[MASK]波特` 或者 `J.K.[MASK]琳` 对于模型都很简单，但是模型不能学到`哈利波特`和`J.K. 罗琳`的关系。如果把`哈利波特`直接MASK掉的话，那模型可以根据作者，就预测到小说这个实体，实现了知识的学习。

需要注意的是这些知识的学习是在训练中隐性地学习，而不是直接将外部知识的embedding加入到模型结构中（[ERNIE-TsingHua](https://arxiv.org/pdf/1905.07129.pdf)的做法），模型在训练中学习到了更长的语义联系，例如说实体类别，实体关系等，这些都使得模型可以学习到更好的语言表达。

首先我们先看看模型的MASK的策略和BERT的区别。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/d06bf008c89f4d80d5f2f1125011e798.png)

ERNIE的mask的策略是通过三个阶段学习的，在第一个阶段，采用的是BERT的模式，用的是basic-level masking，然后在加入词组的mask(phrase-level masking), 然后在加入实体级别entity-level的mask。
如下图

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/04fa80065afaf21dcd464514a45c8ee2.png)

- basic level masking
在预训练中，第一阶段是先采用基本层级的masking就是随机mask掉中文中的一个字。

- phrase level masking
第二阶段是采用词组级别的masking。我们mask掉句子中一部分词组，然后让模型预测这些词组，在这个阶段，词组的信息就被encoding到word embedding中了。

- entity level masking
在第三阶段， 命名实体，例如说 人命，机构名，商品名等，在这个阶段被mask掉，模型在训练完成后，也就学习到了这些实体的信息。

不同mask的效果
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/15edb52ec5bfb11009bd44958a2993fa.png)
## Heterogeneous Corpus Pre-training
训练集包括了
- Chinese Wikepedia
- Baidu Baike
- Baidu news
- Baidu Tieba
注意模型进行了繁简体的转化，以及是uncased

## DLM (Dialogue Language Model) task
对话的数据对语义表示很重要，因为对于相同回答的提问一般都是具有类似语义的，ERNIE修改了BERT的输入形式，使之能够使用多轮对话的形式，采用的是三个句子的组合`[CLS]S1[SEP]S2[SEP]S3[SEP]` 的格式。这种组合可以表示多轮对话，例如QRQ，QRR，QQR。Q：提问，R：回答。为了表示dialog的属性，句子添加了dialog embedding组合，这个和segment embedding很类似。
- DLM还增加了任务来判断这个多轮对话是真的还是假的

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/8a80c9b4901bb7a2c1a203196e3079ae.png)

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/3753a88ecda2db42f3cbdad3c4da53ad.png)

## NSP+MLM
在贴吧中多轮对话数据外都采用的是普通的NSP+MLM预训练任务。
NSP任务还是有的，但是论文中没写，但是git repo中写了用了。

最终模型效果对比bert
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/aec47b7b605317ce824a2ea18dac3249.png)