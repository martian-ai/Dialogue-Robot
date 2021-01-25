# Overview
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Overview](#overview)
- [Pruning](#pruning)
	- [Why Pruning](#why-pruning)
	- [Weight Pruning](#weight-pruning)
	- [Neuron Pruning](#neuron-pruning)
	- [Bert Pruning](#bert-pruning)
- [Knowledge Distillation](#knowledge-distillation)
	- [Theory](#theory)
	- [DistilBert](#distilbert)
		- [Knowledge distillation](#knowledge-distillation)
		- [architecture choice and initialization](#architecture-choice-and-initialization)
		- [Performance and Ablation](#performance-and-ablation)
	- [Bert-PKD](#bert-pkd)
		- [Patient Knowledge Distillation](#patient-knowledge-distillation)
		- [architecture choice](#architecture-choice)
	- [TinyBert](#tinybert)
		- [general distillation](#general-distillation)
		- [task-speciﬁc distillation](#task-specic-distillation)
		- [Ablation Studies](#ablation-studies)
		- [summary](#summary)
- [Parameter Quantization](#parameter-quantization)
- [Architecture Design](#architecture-design)
	- [Matrix Factorization](#matrix-factorization)
	- [Albert](#albert)
		- [Matrix Factorization](#matrix-factorization)
		- [Cross-layer Parameter Sharing](#cross-layer-parameter-sharing)
		- [Sentence Order Prediction（SOP）](#sentence-order-predictionsop)
		- [Ablation Studies](#ablation-studies)
		- [Factors affecting model performance](#factors-affecting-model-performance)
- [Dynamic Computation](#dynamic-computation)
- [ref](#ref)
- [appendix](#appendix)
	- [Depthwise Separable Convolution](#depthwise-separable-convolution)

<!-- /TOC -->
![mind-map](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/71cc8b5e3b76dc938b2e26d52658d609.png)

本文中重要的bert模型压缩的论文概要

| name                   | paper                                                                                                                                                                                   | code                                                                                                | explanation | issue time |
|------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-------------|------------|
| Knowledge Distillation | [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)                                                                                                    |                                                                                                     |   很好的论文也很经典 | 2015.03    |
| DistilBERT             | [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/pdf/1910.01108.pdf)                                                                   | [repo](https://github.com/huggingface/transformers/tree/master/examples/distillation)               |将KD很好的应用在Bert上 | 2019.10    |
| Bert-PKD               | [Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/pdf/1908.09355.pdf)                                                                                       | [repo](https://github.com/intersun/PKD-for-BERT-Model-Compression)                                  | 提出了中间层的输出拟合 | 2019.08    |
| TinyBERT               | [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/pdf/1909.10351v1.pdf)                                                                                  | [repo](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)               |2phase的KD以及针对transformer的KD训练| 2019.09    |
| Albert                 | [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)]                                                                       | [repo](https://github.com/google-research/ALBERT)  & [repo-zh](https://github.com/brightmart/albert_zh) |非常好的模型结构压缩以及模型效果影响因素的分析| 2019.09    |
| KD Analysis                 | [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/pdf/1908.08962)                                                                      | | KD的分析，初始化以及预训练的重要性      | 2019.08    |


为什么需要模型的压缩？主要就是因为模型训练太慢了，当然现在提出了一些解决的针对的训练方式（LAMB以及mix precision training），但是过大的模型，特别是现在热门的语言模型，动辄上亿的参数量，导致了很难在工业中进行实际应用，需要耗费很大的机器资源。对于不一些不联网的应用场景，模型往往需要装在装置中，这些便携式装置的容量有限，过大的模型不仅导致预测时间过慢，占用空间过大，极大降低了导致用户体验感。

![lm_size](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/32129bf4050ca4eb9d0631f522f3ae99.png)

该图现在语言模型的大小趋势图，参数量单位为百万级(millions)
# Pruning
有理论说明，我们的模型往往是Over-parameterized，往往模型需要的参数远超过需要的量。如下图所示，我们人脑的神经元也是随着时间的变化，先从少到多，然后到了稳定之后，所需要的神经元往往少于婴幼儿时期的神经元size。

![](https://www.bioserendipity.com/wp-content/uploads/2019/03/Neuronal_Synapse_Pruning.jpeg)

这时我们需要的是网络的pruning。

那么什么是网络的pruning呢？
- 1. 预训练一个大的网络
- 2. evaluate 网络中参数/神经元的重要性
  - 1. 参数的重要性： L1， L2 loss
  - 2. 神经元的重要性：在训练集中该神经元不为0的次数
- 3. 删除不重要的参数/神经元（少量）
- 4. 在训练数据上fine-tune
- 5. 评测新的模型效果
  - 1. 如果效果好，并且模型大小满足需求停止
  - 2. 否则继续loop from a）

![pruning_model](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/f3f6ae0330c3feb906b7e78b2d5c4f84.png)
注意：
1. pruning之后，accuracy会降低
2. 需要再在training set上 finetune
3. 不要一次prune太多的参数/神经元，否则很难得到好的performance

## Why Pruning
为什么需要pruning，不直接训练小的网络呢？这是因为小的网络比大的网络更难训练，[大的网络比较容易optimize](https://www.youtube.com/watch?v=_VuWvQUMQVk)，这边有一个大乐透假设[Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)。

这篇文章中指出，我们训练一个大的网络然后进行模型剪枝，得到一个小的网络，这样的效果很好。但是如果我们直接拿小的网络的结构去训练，随机初始化，最后的结果往往不好。但是如果还是用这个小的网络，以大的网络的初始化参数初始，这样可以得到好的performance。

文章提出的理论是，大的网络是由多个小的sub-networks组成的，只要有一个sub-network得到好的结果，那这个网络表现就好，然后这个subnet的初始化参数可以算是一个好的初始化参数，是大乐透的头奖，但是直接初始化小的网络，它初始化的参数往往很重要，初始化不好的话，很难得到好的performance。

![lottery_t_h](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/177947a63ad437c539836d0cb2a016d0.png)

当然有另外一篇[文章](https://arxiv.org/abs/1810.05270)直接打脸了这篇文章, 表示小的网络也可以得到好的结果，具体作者没怎么研究。

## Weight Pruning
参数的pruning，就是对不重要的参数进行删除，这个原理很简单但是在实际操作上并不可行，这是因为参数剪枝之后，神经元的链接变得不是规则的矩阵操作，虽然参数量降低了，但是这导致不能进行GPU的并行矩阵操作，往往速度变得更慢。

![wgt_prun](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/fcdc15f386a02e5eeed84551fb78b4ec.png)

实际中，参数的剪枝往往是通过将权重替代成0表示的，这就导致模型大小并没有减少，而精度却降低了。

## Neuron Pruning
神经元的剪枝往往是一个实际的办法，它的做法是将不重要神经元进行直接删除，衡量神经元重要程度的依据就是根据这个神经元在训练数据中的不为0的次数（当然肯定还会有其他的方式），这种做法其实和droupout的效果很像，是一种regularization。

![neu_prun](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/82518ef3829ca7eecfd1c437c4a64f1d.png)

## Bert Pruning
![](https://blog.rasa.com/content/images/2019/08/Pruning-1.png)
首先我们观察bert的结构
![bert_arc](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/e3bc4c4f3cbf83f29b2d03b9607fa70e.png)

占坑：参考
https://blog.rasa.com/pruning-bert-to-accelerate-inference/

# Knowledge Distillation
知识蒸馏（[Knowledge Distillation](https://arxiv.org/pdf/1503.02531.pdf)）的概念由Hinton在2015年提出的，它主要的思想是让老师传授知识给学生。小的学生模型去mimic大的老师模型的输出(soft output)，这样子得到一个压缩的模型（distlbert）。具体的理论解释会在下面说明。但是这有一个问题，老师只教导给学生解题的答案，这往往导致学生在遇到新的题目的时候会出现较大的失误，这时候有人提出了一个patient有耐心的老师的概念（bert-pkd），这位老师不仅教导学生解题答案，并且把解题思路也一并教导，授人以鱼不如授人以渔。在这些的基础上，针对模型的知识蒸馏也被相应提出，例如tinybert，提出了针对transformer的知识蒸馏的结构。写下来我们跟着这个思路来具体了解一下。
![](https://blog.rasa.com/content/images/2019/08/Knowledge-distillation-2.jpg)
Knowledge distillation refers to the idea of model compression by teaching a smaller network, step by step, exactly what to do using a bigger already trained network. The ‘soft labels’ refer to the output feature maps by the bigger network after every convolution layer. The smaller network is then trained to learn the exact behavior of the bigger network by trying to replicate it’s outputs at every level (not just the final loss).

https://blog.rasa.com/compressing-bert-for-faster-prediction-2/
## Theory

[Knowledge Distillation](https://arxiv.org/pdf/1503.02531.pdf)指出大的网络例如通过多个网络进行emsemble的结果，往往能比直接用相同数据训练小模型generalize效果更好。

我们通过小的模型去学习大的模型的直接输出，往往能够得到更好的结果，这是因为大的模型的信息熵比onehot的结果往往更多，小的模型能获得的信息量更大。

例如：在MNIST的图片预测中，1被教师模型预测的概率（soft label）为1:0.7，7:0.2，9:0.1，小的模型如果直接学习onehot（hard label）表示（那么1:1 其他概率都为0），这样子模型不能够学习到1和7很像以及和9也有点像的事实，获取的信息量不多。有论文指出，比如在学生模型在没有7的训练集上可以正确预测出7，这就是因为学到了教师模型的概率表示。（这种知识也被称为黑知识dark knowledge）

![soft_label](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/546a160622f06ebc2997ab71ed85e01c.png)

那么如何学习这些输出？$L_{KD}$
- 一种方法是直接对模型的输出logits（最后一层softmax的输入）进行拟合，然后计算MSE（KD没采用这种）
- 另外一种是拟合softmax的输出结果，但是同样会出现一个问题。

```python
>>> a = tf.nn.softmax([100.0,10,1])
>>> b = tf.nn.softmax([1.0,0.1,0.01])
>>> with tf.Session() as sess:
...     sess.run([a,b])
...
[array([0., 0., 1.], dtype=float32), array([0.5623834 , 0.22864804, 0.20896856], dtype=float32)]
```

我们可以看到，例如我们的logits是`[100，10，1]`，但是通过`softmax`之后，输出的结果变味了`[1, 0, 0]`，这就和直接学习one hot一样，失去了学习的意义，所以论文中提出了一种解决方法，加入了temperature $T$的机制，这样子能够使得softmax之后继续保持着教师模型的输出信息量。值得注意的是，在训练的时候，教师和学生模型都需要加入相同的$T$（为了更好的拟合教师模型的原始输出值）, 而在预测的时候，学生模型的$T=1$

![softmax-t](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/299a112dda3089197ad905f7af008ceb.png)

出了mimic soft lebel获取$L_{DS}$ 误差之外论文中增加了模型拟合hard laebl（one hot的原始正确预测值）$L_{CE}$ 进行cross entropy 误差计算。
总结来说就是通过
- distillation loss $L_{DS}$ （拟合soft label with temperature） 权重比较大
- 训练的目标 cross entropy loss $L_{CE}$ （拟合hard label） 权重较小

总的知识蒸馏误差 $L_{KD} =\alpha L_{DS} +(1-\alpha) L_{CE} $
往往来说都是通过蒸馏误差+训练目标loss组成，当然这边的训练目标loss可以根据不同的模型指定

## DistilBert
DistilBert 的作者团队最初是在Medium上提出自己的想法以及实作，可以参考这篇文章--[Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT](https://medium.com/huggingface/distilbert-8cf3380435b5)，作为论文的引入点。
其实论文[DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/pdf/1910.01108.pdf) 很简单也很好懂，只要理解KD的机制，就可以轻松的理解好论文。

### Knowledge distillation

这篇文章提出，模型最后的输出的由distillation loss   $L_{ce}$ 以及训练误差，分别是  Mask language  modeling loss $L_{mlm}$ 和cosine embedding loss $L_{cos}$的线性加和组成， 即：

  \begin{equation}
  L_{loss}= \alpha_{ce}*L_{ce} + \alpha_{lml}*L_{mlm} + \alpha_{cos}*L_{cos}
  \end{equation}

  其中 $\alpha_{xx}$ 为权重
- distillation loss $L_{ce}$
  - $L_{ce} = \sum _i t_i * log(s_i)$ 其中 $t_i$ （ $s_i$ ）分别是老师模型（学生模型）的概率输出，具体的公式用的是Hinton的softmax-temperature：$t_i = \frac {exp(z_i/T)} { \sum _j exp(z_j/T)}$, 训练的时候两个模型所用的Temperature $T$ 都是一样的，预测是学生模型用的 $T=1$。
  - 上面的公式主要是拟合两个的模型输出概率分布，我们理想当然都是用cross entropy来解决，但是在看源码的时候，我们发现这实现中用的是Kullback-Leibler loss

  ![kl](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/160605c4061194d530471d64b36d3cbe.png)

  其中p是老师的输出分布，q是学生的输出分布。

```python
  nn.KLDivLoss(F.log_softmax(s_logits_slct / self.temperature, dim=-1),
  F.softmax(t_logits_slct / self.temperature, dim=-1))
```

  刚开始看的时候并没有很理解为什么要用KL距离，但是经过查阅资料，发现其实KL距离和cross entropy其实是等价的，都是拟合两个概率分布，使得最大似然。KL diversion 代表的是两个分布的距离，越大 代表分布越不像，越小=0 代表两个分布一样
  > 详细的KL推导参考
  > Theory [pdf](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/GANtheory%20(v2).pdf),[pptx](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2018/Lecture/GANtheory%20(v2).pptx),[video](https://youtu.be/DMA4MrNieWo) (2018/05/11)

  ![kl2](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/e1414925881a8f5038b25d197ed2ca4b.png)

  > Theory Unsupervised Learning: Deep Generative Model [pdf](https://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/VAE%20(v5).pdf),[video (part 1)](https://www.youtube.com/watch?v=YNUek8ioAJk),[video (part 2)](https://www.youtube.com/watch?v=8zomhgKrsmQ) (2016/12/02)

  ![kl3](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/142bea590438c8c9f2ffa07c3b11cfc9.png)
  > [cross entropy，logistic loss 和 KL-divergence的关系和区别](https://blog.csdn.net/u012223913/article/details/75112246)
  >- cross entropy和KL-divergence作为目标函数效果是一样的，从数学上来说相差一个常数。
  >- logistic loss 是cross entropy的一个特例


- Mask language  modeling loss（跟Bert 一致）$L_{mlm}$
- 首层的embedding的cosine embedding loss $L_{cos}$

### architecture choice and initialization
- DistilBert将token-type embeddings以及pooler层去掉了 (used for the next sentence classification task) ，这边我不是很理解，因为pooler层对应的是[CLS]的输出，去掉了怎么做分类任务？可能需要再研究一下。但是我看distilbert 的[应用](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)的时候，是有[CLS]层的。

- 学生模型layer数是老师模型的一般，但是hidden dim是一致的，文章中指出学生参数初始化是直接复制老师模型的layers，具体的操作是skip的方式，例如12层的教师模型，学生模型6层，初始化用的是`for teacher_idx in [0, 2, 4, 7, 9, 11]:`


### Performance and Ablation
蒸馏模型的效果取决于三个方面，一个是模型大小，一个是模型效果，以及预测速度。文章中对比了distlbert以及教师模型Bert-Base，得出了结论，可以得到97%的bert的效果，大小减少了近40%，预测时间提高了60%。可以说还是非常好的。它也提出了，它的学生模型可以在iphone7上直接运行。具体见下图。

![distilbert-ab](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/047f3eff474b5e3f4dd11a5d512987bb.png)

文章中国主要提出了三种loss加权以及权重初始化的方式，这四者是不是都是有用的？哪个其效果最好？下面的分析可以看出，这四者都是有用的（当然哈，要不然也不会放上来），但是其最大作用的是distillation loss 以及参数初始化，作用最小的是训练误差 mask language model loss。

![distb-loss-ab](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/159580a299e32a599ba44121d3131b61.png)

这个模型学习了教师模型的输入 $L_{cos}$ 和输出 $L_{ce}$，并且在训练目标上进行了finetune $L_{mlm}$, 还提出了参数初始化的思想，可以说是很不错了，这些方法可以transfer到不同的模型中去，但是模型中间层的仅靠参数初始化够吗？下面会给出答案。

## Bert-PKD
Bert-PKD，在KD的基础上，提出了一种改进，叫做Patient Knowledge Distillation。在这种方法下，得到了和教师模型相同效果的压缩模型。这种方法地提出就是因为我们在普通地KD中，只学习了教师模型地输出，而不注重中间层的参数方式。直观地理解来说，我们怎么可以只学习题目的答案而错过中间的解题过程？耐心的老师会教大家中间的过程！那么中间层的选择怎么选呢，本文[Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/pdf/1908.09355.pdf)提出了两种方式，一种是PKD-last（只学习最后几层的输出），一种是PKD-skip（学习中间每隔一层的输出），这两种方式的选择上，PKD-skip略胜一筹。

### Patient Knowledge Distillation
PKD的loss 主要由三个部分组成
\begin{equation}
L_{PKD} = (1-\alpha)L^s_{CE}+\alpha L_{DS}+\beta L_{PT}
\end{equation}
其中$\alpha$， $\beta$都是参数

通常的DS都是由distillation loss以及training loss组成，这里也不例外，distillation loss 这边由常规的softmax-soft loss $L_{DS}$ 以及 特制话化的中间层误差 $L_{PT}$ 组成，除此之外，还有一个就是训练误差用来finetune模型，$L^s_{CE}$，这边和DistilBert不一样的是，它没有采用Mask Language Model的loss 而是采用了具体任务的分类误差作为loss。下面会进行详细说明。

- softmax-soft loss $L_{DS}$

\begin{equation}
L_{DS} = -\sum _{i \in[N] } \sum_{c \in C} [P^t(y_i=c | x_i;\hat \theta ^t) \cdot logP^s(y_i=c | x_i; \theta ^s)]
\end{equation}

其中

\begin{equation}
\begin{split}
P^t(y_i=c | x_i) &= softmax(\frac {Wh_i} {T}) \\
&=softmax( \frac{W \cdot BERT^{12}(x_i; \hat \theta ^t)} {T})
\end{split}
\end{equation}
详细的就是都是softmax-soft的公式表示，详见之前的原理部分。

- taskspeciﬁc cross-entropy loss $L^s_{CE}$
\begin{equation}
L^s_{CE} = -\sum _{i \in[N] } \sum_{c \in C} [1 [y_i= c] \cdot logP^s(y_i=c | x_i; \theta ^s)]
\end{equation}
这边的拟合的目标是one hot表示的，具体的目标是task specific的。只有当 $y_i = c$时，目标为1，其余时候都是0。

- Patient Teacher Loss $L_{PT}$

\begin{equation}
L_{PT} = -\sum _{i=1}^N \sum_{j=1} ^{M}  ||\frac{h_{i,j} ^s}{||h^s _{i,j}||^2_2} - \frac{h_{i,I_{pt}(j)} ^t}{||h^t _{i,I_{pt}(j)}||^2_2}||^2_2
\end{equation}

N:代表的是**training sample**的个数
M:代表的是**number of layers in student model** 学生模型的中间block的个数
所以这边中间隐藏层的表示维度两个模型是一致的。
这个公式咋看很复杂，但是其实就是求学生模型的中间隐藏层的归一化后的向量（normalized hidden states）表示的MSE（均方误差）。

$h_i$：代表的是[CLS] token 的第$i$隐藏层输出，$h_i= [h_{i,1}, h_{i,2}, . . . , h_{i,k}]$ 每个隐藏层有k维
$I_{pt}$ 表示的是模型中间隐藏层的对应关系，例如我们有teacher 模型12层，学生模型6层，学生的每个隐藏层的输出要怎么对应教师的隐藏层，由于我们在softmax-soft中已经显性地学习了最后一层的输出，所以我们只需要对应5个隐藏层的结果就行了。本文提出了两种思路，一种是：
  - 1. PKD-skip：12层的话，$I_{pt}= \{ 2, 4, 6, 8, 10 \}$
  - 2. PKD-last：12层的话，$I_{pt}= \{ 7, 8, 9, 10, 11 \}$

（需要特别注意的是，我们这边提到的隐藏层，是指的BERT里面的一个block为单位，也就是encoder级别的）。
下面这张图很好的表示三种loss以及两种策略。
![bertpkd-ill](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/2a3bd2b5440d80ff045949295069ca42.png)

### architecture choice
- 该论文没有用基础的BERT模型做教师，而是直接用fine-tune后的结果来做教师模型
- 对于学生模型（block数目是老师的一半）的初始化，他直接使用的是是BERT_base前k层的参数进行初始化

btw，它的ablation studies写的真差

Well-Read Students Learn Better: On the Importance of Pre-training Compact Models TODO：这篇文章好像提到了pretraining的重要度以及初始化的重要度，需要补充进来

## TinyBert
![tnb-mindmap](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/62c7ca7a4cc5f4e603013ea2d4060952.png)

[TinyBert](https://arxiv.org/pdf/1909.10351v1.pdf)作为华为出品的知识蒸馏论文，在笔者写时还在under review的状态，我特别注意到已经提交了第三版了。当时还纳闷，但是读完了文章之后，觉得思路以及工作上还是不错的，但是文笔以及文章的结构就一般，以及loss的设计和模型初始化的选择都需要改进（个人愚见xp）。

在我们之前介绍的两种模型上，我们已经看到了学习输入的embedding，学习初始化, 学习中间层，学习输出层，以及特定任务fine-tune，这些思路都是能够很好的提升模型的效果的。那么我们还能怎么提升呢？

TinyBert 提出了，虽然这些思路很不错，但是都不是针对transformer的KD方法，我可以在更细粒度的学习transformer的中间表示，例如虽然Bert-PKD提出了中间block输出学习，但是这仅仅是学习到了这个block的输出，但是这个block是由self-attention+dense组成的，作为学生模型，我学习到了最后的输出结果，但是更中间的结果我没学到呀，我要学得更深！！！针对这个思路，我认为这个也可以叫做VPKD，very patient knowledge distillation 哈哈。因为老师模型在教做题的时候，每个步骤都讲得很仔细，没有一个步骤跳过或者省略。

当然模型的效果也是很重要的，虽然tinyBert生成自己取得了state of the art 在模型压缩上，但是它论文中说在GLUE 任务上达到了BERT_base 96% 的效果，但是和DistilBert（97%）以及Bert-PKD（100%）相矛盾。（这边需要再探讨一下呀)。但是这个模型的压缩效果很不错，只有28% 的参数量，预测速度也只需要31%的时间，确实是一个很大的提升。

本文提出的方法是完全针对的pre-training-then-ﬁne-tuning paradigm，也是提出了**明确**提出两步走的策略，Transformer Distillation 由两步组成，一步是general distillation，还有就是task-speciﬁc distillation（虽然之前所有的都是这样子做的 - -），但是还是有些创新的。

这个思路的提出其实结合现实例子也好理解，我们的上大学过程中，首先我们大一的时候接受的是通识教育（general distillation）是我们具备现代的大学生需要的人文 政经 理化的知识，然后在大二之后，我们可以自己选择专业（specific task phrase），这个阶段我们学习的更多是本专业的内容，然后对接受更多本专业材料的学习（Data augumentation），最后学生毕业了～

Tiny Bert的使用说明参见[ref](https://mp.weixin.qq.com/s/cqYWllVCgWwGfAL-yX7Dww)
### general distillation
在genneral distillation的阶段，没有finetune的original BERT 作为老师（这个应该是很潮了吧，之前的应该不是用没有finetune的，这个可以check一下）。这个时候学生学习的是老师模型general domain的知识，然后再在下游任务上面进行finetune。

具体如何学习呢？

文章提出了针对transformer的KD方法，叫做**Transformer Distllation**，具体就是因为每个transformer由input + encoder*N+ output组成，我们也就针对每个都学习，但是这样子好像大家都会呀，所以把encoder又细分成self-attention+dense，分别在学习这个结果，名曰attention loss。

![transformer-ds](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/394d5756b37e0b34491410842f3a21b9.png)
- Embedding-layer Distillation
 其实embedding layer的学习在distilBERT已经提到了，它学习的是embedding的cosine距离，但是这边提出的的方式用的是MSE，这种方式可能有待商榷（个人觉得embedding 用cosine distance更合理一点）。
\begin{equation}
  L_{embd} = MSE(E^SW_e, E^T)
\end{equation}
其中$E^S$ 表示的是学生的embedding，$E^T$表示的是老师的embedding，为什么需要乘以$W_e$ 呢，这是因为做一个线性映射, $W_e$ 是一个可学习的矩阵，目的是把学生模型的特定向量映射到对应的老师模型的向量空间去吗，因为我们**不要求**两个的维度一致。

- Transformer-layer Distillation
  - attention based distillation
  attention 的权重可以获取很多的语言学的知识，所以不能够忽视这些信息。文章中定义每一层的attention loss $L_{attn}$ :
  \begin{equation}
  L_{attn} = \frac 1 h MSE(A^S_i, A^T_i)
  \end{equation}
  这边的$h$ 表示的是attention heads的个数。

  - hidden states based distillation
    除了mimic attention 的权重之外，我们还需要mimic每个encoder的hidden states的输出，$L_{hidn}$:
    \begin{equation}
    L_{hidn} = MSE(H^SW_h, H^T)
    \end{equation}

    其中$H^S$ 表示的是学生的某一个block的hidden states的output，$H^T$表示的是老师的对应block的hidden states的output，为什么需要乘以$W_h$ 呢，这是因为做一个线性映射,$W_h$是一个可学习的矩阵，目的是把学生模型的特定向量映射到对应的老师模型的向量空间去吗，因为我们**不要求**两个的维度一致。
- Prediction-Layer Distillation
  最后的这个就是典型的softmax-soft loss了 $L_{pred}$
  \begin{equation}
  L_{pred} = −softmax(\frac{z^T}{T} ) · log_softmax(\frac {z^S} {T}),
  \end{equation}
  > 这边原论文中$−softmax(z^T)$ 我没看到有做temperature的引入，我觉得是错的，如果我说的不对可以联系我哈

### task-speciﬁc distillation
在task-speciﬁc distillation阶段，首先我们进行了数据增强，对特定任务提高更多的数据，然后在再这些数据上重复transformer distallation的操作。
- step 1: 数据增强，Glove替换相近的词等方式
- step 2: 训练特定任务的Bert，然后继续使用特定任务的bert作为老师，重复general distllation的方式。

### Ablation Studies
![tnb-ab](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/61eca85044f707e0a3b09da60d8c1626.png)
- learning procedures
  - TD (Task-speciﬁc Distillation)：比较重要
  - GD (General Distillation): 相对作用比较小，所以fine-tune和数据是大头
  - DA (Data Augmentation)：比较重要
- distillation tasks
  - $L_{embd}$：这个只是一般重要，但是这个在DistilBert中还是很重要的，所以我更加觉得这边的loss设计有点问题了。应该使用的是cosine distance loss
  - $L_{attn}$：很重要
  - $L_{hidn}$：很重要
  - $L_{pred}$


### summary
本文的loss 组成是有一下三大部分组成
![tnb-ls](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/a036bf3ab6fa215fe74b47866978e27e.png)

对比了TinyBert和DistilBert以及BERT-PKD的distillation方法。这张图非常棒哈！!
- BertPKD 就直接学习finetune后的教师模型了，它的参数初始化都是用BERT_base的前几层进行初始化的，同时它学习了block的中间输出和最后的预测输出和具体任务的结果cross entropy
- DistilBERT 用的是教师模型的skip layer作为参数初始化，（而且ablation studies指出，这还相当重要），然后它还预测了语言模型的MLM loss和最后的KD loss和任务loss
- TinyBERT两阶段中，都增加了很多的loss，但是它没有进行参数的初始化，我觉得这个是一个**很重要**需要改进的地方

![tnb-ab](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/c86221c72b54aa14793c7e743f1a35a3.png)

整个two-phase的架构设计就是这样，先学习general的student bert 然后在学习specific的bert。结合我提出的本科教育体制就很好理解了。
![tnb-train](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/4cc46d7dadf0b3c010691590093c45ba.png)


# Parameter Quantization
parameter quantization使用的是少的bits表示一个value，具体的话，可以使用k-means的方法将权重进行clustering，例如下图所示，我们可以将一个模型的权重聚类成4种，然后每一种都使用2bit表示。如果使用Huffman code表示，能减少更多的空间。
![quant-1](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/02c090d7c8eb530afff7e4a1a9cf4ddc.png)

![](https://blog.rasa.com/content/images/2019/08/Quantization-1.jpg)

占坑：参考[Compressing BERT for faster prediction](https://blog.rasa.com/compressing-bert-for-faster-prediction-2/)


# Architecture Design
通过模型结构的改变，我们可以使得模型参数大大减少。
## Matrix Factorization
例如通过matrix factorization，一个$M * N$ 的矩阵，可以分解成 $M*K$ 和 $K*N$ 的矩阵相乘。通常这边的 $K$ 是远小于 $M$ 和 $N$ 的。
![mf](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/a53906dd11b75b2daf5f95db95dfaf47.png)
当然这边的矩阵分解的方法，会限制其的一些表达区间，因为乘积出来的结果的秩$<= M$ 也小于$<= N$，而$W$矩阵没有这个限制。（这个不是特别理解）

（see appendix for application in CNN）

## Albert
![albert-im](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/5cbc5b1206c7c643338e6d2fca90476a.png)
google提出的Albert（顺便提一下爱因斯坦也是albert）可以说是一个很轰动的模型了，它的参数量甚至少于TinyBert，但是它的效果却能够达到SOTA！

不愧是google大法，它采用不同于之前的方式，它利用的是改变bert模型结构的方式，来取得模型压缩的效果，所以说之前的KD研究集合在albert作为老师模型，可以取得更好的模型压缩效果。

本文除了在模型压缩上提出了两个非常重要的方式之外
- Matrix Factorization
-  Parameter sharing

还提出了为什么NSP的方式不好的理由，简单的说，就是NSP这种对于BERT来说毫无难度，不如说就是一个鸡肋任务～，之前Robeta以及XLNET，SpanBERT都尝试着解释为什么NSP没有效果，但是都没有这边的解释合理。既然我已经知道了自己的不足了，google团队又提出了新的自监督loss，句子顺序的预测sentence-order prediction (SOP)。结合这些Albert简直就是一颗新星。

此外，本篇文章还分析模型的宽度和深度对模型效果的作用，以及还发现了bert其实是没有under trained，所以add&norm中的dropout应该要去掉（这是首次提出），他也对更多数据的作用做出了分析。本篇文章觉得收到Roberta的影响比较重, 很多相关的研究，此外，他还用了spanBERT的spanMask的作为MLM loss。

稳重训练使用了更大的batch size（4096 bert原始使用256），采用的是很潮的LAMB optimizer。

顺便说一下，[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) 这篇文章很短，也很简单，非常值得阅读，但是需要先了解阅读的文献也很多。源码也只是在BERT的源码基础上做了简单的修改，很适合研究，其中我觉得[albert-zh](https://github.com/brightmart/albert_zh)的code写得更好一些。

总的来说，此文处处有惊喜！

### Matrix Factorization
我刚开始在看的时候，没有注意到问什么我们embedding 的hidden size要和encoder 的每个token的hidden output size一致，后面一想，也对因为我们的每个block都是一样的结构，前一层的输出是作为下一层的输入的，所以我们必须要保证embedding作为第0层输出也要和第一层的输出一样的维度！

BERT, XLNET,ROBERTA 的WordPiece的 embedding size E 都是等于 hidden layer size H, i.e., E ≡ H.

词嵌入向量参数的因式分解 Factorized embedding parameterization，albert 采用不是直接将one-hot 映射到hidden space of size H， 这边采用的是先将模型映射到一个地位的embedding space of size E，然后在投射到高为的Hidden space of size H。这样做好处是将参数量从$O(V * H)$ 减少到$O(V * E + E * H)$

如以ALBert_xxlarge为例，V=30000, H=4096, E=128

 那么原先参数为V * H= 30000 * 4096 = 1.23亿个参数，现在则为V * E + E * H = 30000*128+128*4096 = 384万 + 52万 = 436万，

词嵌入相关的参数变化前是变换后的28倍。
以下是code snippet，有删减节选，具体详见 [modelling.py](https://github.com/brightmart/albert_zh/blob/master/modeling.py)

```python
def embedding_lookup_factorized(...):
    embedding_table = tf.get_variable(  # [vocab_size, embedding_size]
        name=word_embedding_name,
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range))
    ...
    flat_input_ids = tf.reshape(input_ids, [-1])  # one rank. shape as (batch_size * sequence_length,)
    if use_one_hot_embeddings:
        one_hot_input_ids = tf.one_hot(flat_input_ids,depth=vocab_size)  # one_hot_input_ids=[batch_size * sequence_length,vocab_size]
        output_middle = tf.matmul(one_hot_input_ids, embedding_table)  # output=[batch_size * sequence_length,embedding_size]
    else:
        output_middle = tf.gather(embedding_table,flat_input_ids)  # [vocab_size, embedding_size]*[batch_size * sequence_length,]--->[batch_size * sequence_length,embedding_size]

    # 2. project vector(output_middle) to the hidden space
    project_variable = tf.get_variable(  # [embedding_size, hidden_size]
        name=word_embedding_name+"_2",
        shape=[embedding_size, hidden_size],
        initializer=create_initializer(initializer_range))
    output = tf.matmul(output_middle, project_variable) # ([batch_size * sequence_length, embedding_size] * [embedding_size, hidden_size])--->[batch_size * sequence_length, hidden_size]
    ...
    return (output, embedding_table, project_variable)
```

### Cross-layer Parameter Sharing
Cross-layer Parameter Sharing 防止了模型的参数量随着block的增加而增加

- all-shared strategy(ALBERT-style)
all-shared strategy指的是，share self-attention以及FFN layer 的参数
- shared-attention
 shared-attention strategy指的是，share self-attention 的参数
- shared-FFN
 shared-FFN strategy指的是，share self-FFN 的参数
- not-shared
  什么都不share是原始bert的模式

对于我来说，之前从来都没接触过parameter sharing，所以我仔细看了一下模型文件[modelling.py](https://github.com/brightmart/albert_zh/blob/master/modeling.py)发现其实很简单，我们只需要把需要parameters sharing的部分，加上 `with tf.variable_scope(scope的ID, reuse=tf.AUTO_REUSE):` 就可以重复使用这个scope里面的参数了，那具体如何选择我们sharing的力度呢，这边的解决方案是定义一个dict，来对应share的layer名称。具体看code

share strategy的implementation
```python

def layer_scope(idx, shared_type):
		if shared_type == 'all':
			tmp = {
				"layer":"layer_shared",
				'attention':'attention',
				'intermediate':'intermediate',
				'output':'output'
			}
		elif shared_type == 'attention':
			tmp = {
				"layer":"layer_shared",
				'attention':'attention',
				'intermediate':'intermediate_{}'.format(idx),
				'output':'output_{}'.format(idx)
			}
		elif shared_type == 'ffn':
			tmp = {
				"layer":"layer_shared",
				'attention':'attention_{}'.format(idx),
				'intermediate':'intermediate',
				'output':'output'
			}
		else:
            # 不共享？
			tmp = {
				"layer":"layer_{}".format(idx),
				'attention':'attention',
				'intermediate':'intermediate',
				'output':'output'
			}

		return tmp
```
具体scope的层次
```python

for layer_idx in range(num_hidden_layers):

	idx_scope = layer_scope(layer_idx, shared_type)

	with tf.variable_scope(idx_scope['layer'], reuse=tf.AUTO_REUSE):
		layer_input = prev_output

		with tf.variable_scope(idx_scope['attention'], reuse=tf.AUTO_REUSE):
			attention_heads = []

			with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
				layer_input_pre = layer_norm(layer_input)

			with tf.variable_scope("self"):
```

### Sentence Order Prediction（SOP）
在训练Albert的时候，使用了改进版的masked language modeling（MLM） loss 之外，放弃了next-sentence prediction(NSP)的使用，使用了全新的 Sentence Order Prediction（SOP）。
\begin{equation}
  L_{loss} = L_{ngram-MLM} +L_{SOP}
\end{equation}
- ngram-MLM:
最初的bert使用的是mask wordpiece，但是后面ernie1.0发现，这样子做没能获得完整词的knoeledge，所以有了whole word masking（WWM），只要一个词的其中一个wordpiece被mask了，整个词都会被mask，而之后spanBERT发现随机mask连续的span的词效果能得到更好，所以这边albert也使用spanbert的方式，当然采样的方式不一样。这边词的长度设为最长为3，例如White House correspondents，词长度n的采样公式为：
\begin{equation}
  p(n) = \frac{1/n}{\sum_{k=1}^N 1/k}
\end{equation}

WWM做法就是将wordpiece先合并为完成的word 然后在random之后，在mask。refer to [code](https://github.com/brightmart/albert_zh/blob/master/create_pretraining_data.py).
  ```Python
  def create_masked_lm_predictions(tokens, masked_lm_prob,
                                   max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
      if token == "[CLS]" or token == "[SEP]":
        continue
      # Whole Word Masking means that if we mask all of the wordpieces
      # corresponding to an original word. When a word has been split into
      # WordPieces, the first token does not have any marker and any subsequence
      # tokens are prefixed with ##. So whenever we see the ## token, we
      # append it to the previous set of word indexes.
      #
      # Note that Whole Word Masking does *not* change the training code
      # at all -- we still predict each WordPiece independently, softmaxed
      # over the entire vocabulary.
      if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
              token.startswith("##")):
        cand_indexes[-1].append(i)
      else:
        cand_indexes.append([i])

    rng.shuffle(cand_indexes)
  ```

- 为什么不要NSP？
  NSP设计的作用是为了多句子关系的下游任务做准备的，但是在XLNET，spanBERT，RoberTa等文章中，都发现它不仅没有正向效果，还有反向效果，这是为什么呢？
  文章中发现这是因为NSP任务在对于bert模型太简单了，因为NSP的作用是预测句子A和句子B是不是来自于同一个文档document，这其实是由两种任务组成的，分别是top prediction主题预测和coherence prediction连贯性预测。

  \begin{equation}
  L_{NSP}= L_{topic\ pred} + L_{coherence\ pred}
  \end{equation}

  但是其实topic prediction对于语言模型很简单，因为我们在做MLM任务的时候同样也需要关注文章的主题，如果两个句子之间有多个相同的词，模型就更轻而易举的能够知道这两个句子是同一个主题，所以模型不会关注句子连贯性（SOP）的问题。

  下面这张图做了一个实验，训练一个albert模型，但是使用三种loss方式一个是None，一个是NSP，采用一个SOP，结果发现使用NSP的任务，虽然在下个句子的预测（NSP）能够得到90.5的高分，但是在句子连贯性上（SOP）得到52.0的分数，相当于是盲猜，效果甚至不如没有NSP的loss，毕竟没有NSP，两个句子可以合成一个大的句子，模型学到的序列长度更长，更能学到连贯性。

  上图我们可以看出来NSP任务，只关注了简单的任务，top prediction而在SOP任务上盲猜，但是SOP的任务可以一定程度上解决NSP的问题。所以这很好地提高了我们下游任务的支持。这边需要提到的是SOP的任务在Ernie2.0中也用到了。

  ![albert-loss](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/d90cba464dd7f17b346ada5d33364032.png)

- 那我们只focus在句子连贯性上吧，SOP
  那么SOP任务的具体操作是怎么样呢？SOP使用的是来自同个document的连续的两个句子，有50%的几率调换句子之间的顺序，正序的记为正样本，逆序的记为负样本。refer to [code](https://github.com/brightmart/albert_zh/blob/master/create_pretraining_data.py).
  ```Python
  if rng.random() < 0.5: # 交换一下tokens_a和tokens_b
    is_random_next=True
    temp=tokens_a
    tokens_a=tokens_b
    tokens_b=temp
  else:
    is_random_next=False

  ```

### Ablation Studies
- factorization的影响
  我们可以发现在单独（no-shared模型）进行embedding参数的factorization下，模型效果是会降低的。

  但是在参数的albert参数共享的情况下，进行embedding factorization并不会是的模型效果降低。

  ![alber-ab1](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/ca9a3e8aa85c48990ac7d7be32cc4ea6.png)


- parameters sharing 的影响
  在只使用参数共享没有做embedding factorization的模式下(E=768), 可以发现共享参数都使得模型效果降低，但是共享不同的层，降低的效果不一样

  效果如下：
  不共享 > 共享self-attention > 全共享 > 共享FFN

  所以FFN共享效果是最差的，模型中FFN 是两层： input ->intermediate -> output 这可能是因为FFN的作用是使得多头注意力(multi-head attention)的结果进行交互，不同的block之间的注意力需要交互的权重不一致，导致精度降低。

  ![albert-ab2](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/1156c9ace00abe705a4deb82ab30e370.png)

  值得注意的是，在factorization和parameter sharing共同作用下，模型的效果是提升的，而不是持续下降

- SOP的作用

  SOP的作用大大提升了模型的效果

  ![albert-sop](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/507e76c9a10f9ee9e8506634a6da8456.png)

### Factors affecting model performance
在XLNET大举超越BERT之后，RoberTa提出了其实BERT并没有被超越，因为它只是under-trained，只要好好的训它，它就可以变得很强。所以就引出了一个疑问，是不是只要一味的使用更多的数据，把模型叠得更深，hidden size采用更大。就可以得到SOTA的效果呢？

- EFFECT OF NETWORK DEPTH AND WIDTH
  - Depth
  分析指出，并不是更深的网络更好，在12层和24层效果差不多了，48层的效果，甚至不如24层的好。

  ![albert-depth](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/3adca65b485023b850dfaade85c190d6.png)
  - Width
  Hidden size的结论和Depth的结论一直，并不是一味追加hidden size就能够使得模型效果更好，4096的长度效果是最好的。值得注意的是，就算在6144的hidden size上，模型仍然没有over fitting的情况。
  ![albert-width](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/02b91c180dcc3f615996f1e35a922b67.png)

- DO VERY WIDE ALBERT MODELS NEED TO BE DEEP(ER) TOO ?
  在很宽的模型参数Hidden size=4096的情况下，我们需要更深的模型吗？作者通过实验说明，在albert的参数共享的情况下，答案是不需要，12层足矣。

  ![albert-d-w](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/215d75b992b109194b7f8f85cd789cf4.png)

- 数据更多更好吗？
  在加了更多数据后，MLM的准确率确实得到了大幅的提升，在下游任务中也得到了提升，除了SQuAD任务，但是这个任务本来就是基于WikiPedia的数据集的（本来就当作训练数据），所以没有明显的提升。

  总的来说，更多的数据确实是有好的正向效果的。
  ![albert-data](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/3d531dca8c29ed9cf92ae4ebd66160e0.png)

  ![albert-data2](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/1dca5131ea5b6a382409459437594402.png)

- Dropout的作用？
  我们通过实验在非常深以及非常宽的网络中，甚至在训练了100万次之后，模型都没有overfitting的现象，但是我们模型训练的时候，加入了dropout，这会影响模型的能力(capacity)，所以我们在配置中将dropout rate设为0，并得到更好的结果。

  ![albert-dp](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/529ee47800e52ae1aa31f7b88c5449c0.png)

# Dynamic Computation
模型预测的计算量可否根据装置的电量而改变呢？dynamic computation的提出是基于需求。它希望模型有多个不同的classifier，针对不同的场景，动态地选择自身的classifier。

有论文就提出，那我们直接在中间层添加classifer输出，然后训练的时候，要求每一层的输出都可以获得相对好的结果。但是实际上这样子是行不通的，可以在右下角的表，在前面几层强行添加classifier不仅不能获得好的结果，反而导致之后的层数的效果变差。拿CNN举例，在前面几层的CNN，往往是提取简单的特征，但是如果+classifier的话，简单的特征往往不足以使得他获得好的结果，所以CNN前面几层就会也提取复杂的特征，但是这样子的话，他的表征能力也不够，所以导致过犹不及。

![dc](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/177e3f7b885968a75b46d023ce2f129e.png)

所以提出了一个解决方案：
具体参考这个
![dc1](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/d0b32138d210aa8c9722ad460e67bdc2.png)
# ref
![ref1](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/8c82fb6c6477acbd1329e4fabe25a149.png)
[李宏毅networks compression](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2019/Lecture/Small%20(v6).pdf)
[李宏毅-Network Compression课程笔记](https://cloud.tencent.com/developer/article/1557709)
[Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT](https://medium.com/huggingface/distilbert-8cf3380435b5)
[BERT 瘦身之路：Distillation，Quantization，Pruning](https://mp.weixin.qq.com/s/ir3pLRtIaywsD94wf9npcA)
[帮BERT瘦身！我们替你总结了BERT模型压缩的所有方法](https://mp.weixin.qq.com/s/KPEBT_S1rWPwwPUkp5fmoA)
[Tiny Bert的使用说明参见ref](https://mp.weixin.qq.com/s/cqYWllVCgWwGfAL-yX7Dww)
[KD 讲解视频(youtube)](https://www.youtube.com/watch?v=ueUAtFLtukM)
[Knowledge Distillation : Simplified](https://towardsdatascience.com/knowledge-distillation-simplified-dd4973dbc764)
[Pruning BERT to accelerate inference](https://blog.rasa.com/pruning-bert-to-accelerate-inference/)
[Compressing BERT for faster prediction](https://blog.rasa.com/compressing-bert-for-faster-prediction-2/)

# appendix
## Depthwise Separable Convolution
标准的CNN的参数量是$kernal\_size * kernal\_size * input\_channel * output\_channel$
![dsc1](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/8bc243b1a18f3e13e059c316c112309a.png)
使用Depthwise Separable Convolution的方式的话，首先先用$input\ channel\ size$ 个数的filters， size： kernal size*kernal size ，每个filters对应一个input channels，单独对各自的channel 做卷积操作。

![dsc2](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/58157bc2c511af13bbcd1937b3729f63.png)

然后使用 $output\ channel\ size$ 个数的filters， 每个的卷积核的channel数和input channel一致， size： 1*1，对之前步骤产生的卷积输出做正常卷积。得到最后的输出。

![dsc3](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/04db5bb09a420a0e0bbc913310acb42b.png)

分析： 使用正常的卷积操作的话，相当于是将kernel size\* kernel size\*input channel的神经元作为输出，输出为1；
使用Depthwise Separable Convolution的话，是将kernel size\* kernel size 作为输入，输出为1，这样子的神经元有input channel 个，然后在经过另外一层的网络。（<-- 这边需要再重新改一下）
![dsc4](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/dc0c6e4911607dd4a7d609329961e25d.png)

![dsc5](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/4eabbd577c61ddf02bc0642f309e5ff2.png)

TODO：
- pruning 和quantization 和 Dynamic Computation
- Well-Read Students Learn Better: On the Importance of Pre-training Compact Models 论文研读
- appendix修改
