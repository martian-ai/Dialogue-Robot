
# Outline
![attention](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/0e0d02cfe0c0c35de437ab8273fce4ac.png)
# Intuition
吸睛这个词就很代表attention，我们在看一张图片的时候，很容易被更重要或者更突出的东西所吸引，所以我们把更多的注意放在局部的部分上，在计算机视觉（CV）领域，就可以看作是图片的局部拥有更多的权重，比如图片生成标题，标题中的词就会主要聚焦于局部。
![att_exp](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/b0c619392d29a8429e3376c8366c2da1.png)
NLP领域，可以想象我们在做阅读理解的时候，我们在看文章的时候，往往是带着问题去寻找答案，所以文章中的每个部分是需要不同的注意力的。例如我们在做评论情感分析的时候，一些特定的情感词，例如amazing等，我们往往需要特别注意，因为它们是很重要的情感词，往往决定了评论者的情感。如下图 [Yang et al. HAN](https://www.aclweb.org/anthology/N16-1174.pdf)

![txt_exp](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/9c1998ce9b028ca00d815b006ad65eee.png)
直白地说，attention就是一个权重的vector。

# Analysis
## Pros
attention的好处主要是具有很好的解释性，并且极大的提高了模型的效果，已经是很多SOTA 模型必备的模块，特别是transformer（使用了self / global/ multi-level/ multihead/ attention）的出现极大得改变了NLP的格局。
## Cons
没法捕捉位置信息，需要添加位置信息。
就transformer而言，其最大的坏处是空间消耗大，这是因为我们需要储存attention score（N*N）的维度，所以Sequence length（N）不能太长。对于过长的sequence 会采取截断的方式（如果原始sequence 长度是1000， 如果超过500 就截断的话，会产生sequence A 和 sequence B，两者直接不会进行attention计算，就没有关联了) 。（具体参照XLNET以及XLNET的解决方式）

## From Seq2Seq To Attention Model
为什么会有attention？attention其实就是为了翻译任务而生的（但最后又不局限于翻译任务），我们来看看他的具体演化。
### seq2seq
[Seq2Seq model](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) 是有encoder和decoder组成的，它主要的目的是将输入的文字翻译成目标文字。其中encoder和decoder都是RNN，（可以是RNN/LSTM/或者GRU或者是双向RNN）。模型将source的文字编码成一串固定长度的context编码，之后利用这段编码，使用decoder解码出具体的输出target。这种转化任务可以适用于：翻译，语音转化，对话生成等序列到序列的任务。

![en_de](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/57d0131f3a5bfba1e81b4cb65a6653fc.png)

但是这种模型的缺点也很明显：
- 首先所有的输入都编码成一个固定长度的context vector，这个长度多少合适呢？很难有个确切的答案，一个固定长度的vector并不能编码所有的上下文信息，导致的是我们很多的长距离依赖关系信息都消失了。
- decoder在生成输出的时候，没有一个与encoder的输入的匹配机制，对于不同的输入进行不同权重的关注。

### attention
[NMT【paper】](https://arxiv.org/pdf/1409.0473.pdf)          [【code】](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/nmt_with_attention.ipynb#scrollTo=rd0jw-eC3jEh)最早提出了在encoder以及decoder之间追加attention block，最主要就是解决encoder 以及decoder之间匹配问题。

![en_de_w_att](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/c1a14c2e6a1bf1d087df6e5f072cb693.png)

- 其中$s_0$是decoder的初始化hidden state，是随机初始化的，相比于seq2seq（他是用context vector作为decoder的hidden 初始化），$s_i$ 是decoder的hidden states。
- $h_j$ 代表的是第j个encoder位置的输出hidden states
- $a_{ij}$代表的是第i个decoder的位置对j个encoder位置的权重，计算过程如下图
    ![img](https://tva1.sinaimg.cn/large/007S8ZIlly1gfl83iitxuj30ii0dqgmm.jpg)
- $y_i$是第i个decoder的位置的输出，就是经过hidden state输出之后再经过全连接层的输出
- $c_i$代表的是第i个decoder的context vector，其实输出hidden output的加权求和
- decoder的输入是由自身的hidden state以及$[y_i;c_{ij}]$ 这两个的concat结果

### implement

详细的实现可以参照tensorflow的repo使用的是tf1.x [Neural Machine Translation (seq2seq) tutorial](https://github.com/tensorflow/nmt). 这里的代码用的是最新的2.x的代码 [code](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/nmt_with_attention.ipynb#scrollTo=umohpBN2OM94&uniqifier=2).

输入经过encoder之后得到的hidden states 的形状为 *(batch_size, max_length, hidden_size)* ， decoder的 hidden state 形状为 *(batch_size, max_length, hidden_size)*.

以下是被implement的等式：

<img src="https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200614184047-2020-6-14-18-40-50" alt="attention equation 0" width="800">

+ score 的计算方式如下
<img src="https://www.tensorflow.org/images/seq2seq/attention_equation_1.jpg" alt="attention equation 1" width="800">


# Taxonomy of attention
根据不同的分类标准，可以将attention分为多个类别，但是具体来说都是q（query）k（key）以及v（value）之间的交互，通过q以及k计算score，这个score的计算方法各有不同如下表，再经过softmax进行归一化。最后在将计算出来的score于v相乘加和（或者取argmax 参见pointer network）。

Below is a summary table of several popular attention mechanisms and corresponding alignment score functions:

| Name                   |                                                                            Alignment score function                                                                            | Citation                                                                      |
|------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|-------------------------------------------------------------------------------|
| Content-base attention |                                                                        $score(s_t,h_i)=cosine[s_t,h_i]$                                                                        | [Graves2014](https://arxiv.org/abs/1410.5401)                                 |
| Additive(*)            |                                                                    $score(s_t,h_i)=v^T_a tanh(W_a[s_t;h_i])$                                                                   | [Bahdanau2015](https://arxiv.org/pdf/1409.0473.pdf)                           |
| Location-Base          |                                  $α_{t,i}=softmax(W_as_t)$ Note: This simplifies the softmax alignment to only depend on the target position.                                  | [Luong2015](https://arxiv.org/pdf/1508.04025.pdf)                             |
| General                | $score(s_t,h_i)=s^T_t W_a h_i $ where Wa is a trainable weight matrix in the attention layer.                                                                                  | [Luong2015](https://arxiv.org/pdf/1508.04025.pdf)                             |
| Dot-Product            | $score(s_t,h_i)=s^T_t h_i $                                                                                                                                                    | [Luong2015](https://arxiv.org/pdf/1508.4025.pdf)                              |
| Scaled Dot-Product(^)  | $score(s_t,h_i)=\frac{s^T_t h_i}{\sqrt{n}} $ Note: very similar to the dot-product attention except for a scaling factor; where n is the dimension of the source hidden state. | [Vaswani2017](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) |

(\*) Referred to as “concat” in Luong, et al., 2015 and as “additive attention” in Vaswani, et al., 2017.
(^) It adds a scaling factor 1/n‾√1/n, motivated by the concern when the input is large, the softmax function may have an extremely small gradient, hard for efficient learning.

以下的分类不是互斥的，比如说[HAN模型](https://www.aclweb.org/anthology/N16-1174.pdf)，就是一个multi-level，soft，的attention model（AM）。
## number of sequence
根据我们的query以及value来自的sequence来分类。
### distinctive
attention的query和value分别来自不同两个不同的input sequence和output sequence，例如我们上文提到的NMT，我们的query来自于decoder的hidden state，我们的value来自去encoder的hidden state。
### co-attention
co-attention 模型对多个输入sequences进行联合学习权重，并且捕获这些输入的交互作用。例如[visual question answering](https://arxiv.org/pdf/1606.00061.pdf) 任务中，作者认为对于图片进行attention重要，但是对于问题文本进行attention也同样重要，所以作者采用了联合学习的方式，运用attention使得模型能够同时捕获重要的题干信息以及对应的图片信息。
### self
例如文本分类或者推荐系统，我们的输入是一个序列，输出不是序列，这种场景下，文本中的每个词，就去看与自身序列相关的词的重要程度关联。如下图
![self_att](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/518e294f3e65fdd41f4e20f6ab89ad22.png)
我们可以看看bert的self attention的实现的函数说明，其中如果from tensor= to tensor，那就是self attention

```python
def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the **same**, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector
  """
```

## number of abstraction
这是根据attention计算权重的层级来划分的。
### single-level
在最常见的case中，attention都是在输入的sequence上面进行计算的，这就是普通的single-level attention。
### multi-level
但是也有很多模型，例如[HAN](https://www.aclweb.org/anthology/N16-1174.pdf)，模型结构如下。模型是hierarchical的结构的，它的attention也是作用在多层结构上的。
我们介绍一下这个模型的作用，它主要做的是一个文档分类的问题，他提出，文档是由句子组成的，句子又是由字组成的，所以他就搭建了两级的encoder（双向GRU）表示，底下的encoder编码字，上面的encoder编码句子。在两个encoder之间，连接了attention层，这个attention层是编码字层级上的注意力。在最后输出作文本分类的时候，也使用了一个句子层级上的attention，最后输出来Dense进行句子分类。
需要注意的是，这里的两个query $u_w$ 以及 $u_s$ 都是随机初始化，然后跟着模型一起训练的，score方法用的也是Dense方法，但是这边和NMT不同的是，他是self attention。
![han](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/696e496aea6b267cff8a73780e5fae62.png)
## number of positions
根据attention 层关注的位置不同，我们可以把attention分为三类，分别是global/soft（这两个几乎一样），local以及hard attention。[Effective Approaches to Attention-based Neural Machine Translation.](https://www-nlp.stanford.edu/pubs/emnlp15_attn.pdf)  提出了local global attention，[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.](https://arxiv.org/pdf/1502.03044.pdf) 提出了hard soft attention
### soft/global
global/soft attention 指的是attention 的位置为输入序列的所有位置，好处在与平滑可微，但是坏处是计算量大。

### hard
hard attention 的context vector是从采样出来的输入序列hidden states进行计算的，相当于将hidden states进行随机选择，然后计算attention。
这样子可以减少计算量，但是带来的坏处就是计算不可微，需要采用强化学习或者其他技巧例如variational learning methods。

### local
local的方式是hard和soft的折中
- 首先从input sequence中找到一个需要attention的点或者位置
- 在选择一个窗口大小，create一个local的soft attention
这样做的好处在于，计算是可微的，并且减少了计算量

## number of representations
通常来说single-representation是最常见的情况，which means 一个输入只有一种特征表示。但是在其他场景中，一个输入可能有多种表达，我们按输入的representation方式分类。
### multi-representational
在一些场景中，一种特征表示不足以完全捕获输入的所有信息，输入特征可以进行多种特征表示，例如Show, attend and tell: Neural image caption generation with visual attention. 这篇论文就对文本输入进行了多种的word embedding表示，然后最后对这些表示进行attention的权重加和。再比如，一个文本输入分别词，语法，视觉，类别维度的embedding表示，最后对这些表示进行attention的权重加和。
### multi-dimensional
顾名思义，这种attention跟维度有关。这种attention的权重可以决定输入的embedding向量中不同维度之间的相关性。其实embedding中的维度可以看作一种隐性的特征表示（不像one_hot那种显性表示直观，虽然缺少可解释性，但是也算是特征的隐性表示），所以通过计算不同维度的相关性就能找出起作用最大的特征维度。尤其是解决一词多义时，这种方式非常有效果。所以，这种方法在句子级的embedding表示、NLU中都是很有用的。
## summary
![att_sum](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/2bd4f57c8e32ded38317bf537464ffaa.png)
# Networks with Attention
介绍了那么多的attention类别，那么attention通常是运用在什么网络上的呢，我们这边总结了两种网络，一种是encoder-decoder based的一种是memory network。
## encoder-decoder
encoder-decoder网络+attention是最常见的+attention的网络，其中NMT是第一个提出attention思想的网络。这边的encoder和decoder是可以灵活改变的，并不绝对都是RNN结构。
### CNN/RNN + RNN
对于图片转文字这种任务，可以将encoder换成CNN，文字转文字的任务可以使用RNN+RNN。
### Pointer Networks
并不是所有的序列输入和序列输出的问题都可以使用encoder-decoder模型解决，(e.g. sorting or travelling salesman problem).
例如下面这个问题：我们想要找到一堆的点，能够将图内所有的点包围起来。我们期望得到的效果是，输入所有的点$P_1...P_{10}$ 最后输出的是$P_4, P_2, P_7, P_6, P_5, P_3$
![p1](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/81e18682bdb85b93f9c3730c42ded27f.png)

如果直接下去训练的话，下图所示：input 4个data point的坐标，得到一个红色的vector，再把vector放到decoder中去，得到distribution，再做sample（比如做argmax，决定要输出token 1...），最终看看work不work，结果是不work。比如：训练的时候有50 个点，编号1-50，但是测试的时候有100个点，但是它只能选择 1-50编号的点，后面的点就选不了了。

![p2](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/433c42b99179a7addd51578a8649f833.png)
改进：attention，可以让network动态的决定输出的set有多大

x0，y0代表END这些词，每一个input都会得到一个attention的weight=output的distribution。

![p3](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/ed1616cb4dd9a802e8f68354a46b0ea5.png)

最后的模型的结束的条件就是$<end>$点的概率最高
![p4](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/bf1b11a3be16a70a281923bb17c87870.png)

### Transformer
transformer网络使用的是encoder+decoder网络，其主要是解决了RNN的计算速度慢的问题，通过并行的self attention机制，提高了计算效率。但是与此同时也带来了计算量大，空间消耗过大的问题，导致sequence length长度不能过长的问题，解决参考transformerXL。（之后会写一篇关于transformer的文章）
- multihead的作用：有点类似与CNN的kernel，主要捕获不同的特征信息
## Memory Networks
像是question answering，或者聊天机器人等应用，都需要传入query以及知识数据库。End-to-end memory networks.通过一个memroy blocks数组储存知识数据库，然后通过attention来匹配query和答案。
memory network包含四部分内容：query（输入）的向量、一系列可训练的map矩阵、attention权重和、多hop推理。这样就可以使用KB中的fact、使用history中的关键信息、使用query的关键信息等进行推理，这在QA和对话中至关重要。（这里需要补充）
# Applications
## NLG
- MT：计算机翻译
- QA：problems have made use of attention to (i) better understand questions by focusing on relevant parts of the question [Hermann et al., 2015], (ii) store large amount of information using memory networks to help ﬁnd answers [Sukhbaatar et al., 2015], and (iii) improve performance in visual QA task by modeling multi-modality in input using co-attention [Lu et al., 2016].
- Multimedia Description（MD）：is the task of generating a natural language text description of a multimedia input sequence which can be speech, image and video [Cho et al., 2015]. Similar to QA, here attention performs the function of ﬁnding relevant acoustic signals in speech input [Chorowski et al., 2015] or relevant parts of the input image [Xu et al., 2015] to predict the next word in caption. Further, Li et al. [2017] exploit the temporal and spatial structures of videos using multi-level attention for video captioning task. The lower abstraction level extracts speciﬁc regions within a frame and higher abstraction level focuses on small subset of frames selectively.


## Classification
- Document classification：HAN
- Sentiment Analysis：
- Similarly, in the sentiment analysis task, self attention helps to focus on the words that are important for determining the sentiment of input. A couple of approaches for aspect based sentiment classiﬁcation by Wang et al. [2016] and Ma et al. [2018] incorporate additional knowledge of aspect related concepts into the model and use attention to appropriately weigh the concepts apart from the content itself. Sentiment analysis application has also seen multiple architectures being used with attention such as memory networks [Tang et al., 2016] and Transformer [Ambartsoumian and Popowich, 2018; Song et al., 2019].

## Recommendation Systems
Multiple papers use self attention mechanism for ﬁnding the most relevant items in user’s history to improve item recommendations either with collaborative ﬁltering framework [He et al., 2018; Shuai Yu, 2019], or within an encoderdecoder architecture for sequential recommendations [Kang and McAuley, 2018; Zhou et al., 2018].

Recently attention has been used in novel ways which has opened new avenues for research. Some interesting directions include smoother incorporation of external knowledge bases, pre-training embeddings and multi-task learning, unsupervised representational learning, sparsity learning and prototypical learning i.e. sample selection.

# ref
- [写作风格很好，最后模型那块可以再补充到本篇文章]( https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#born-for-translation )
- 非常好的综述[An Attentive Survey of Attention Models](https://arxiv.org/pdf/1904.02874.pdf)
- http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/
- [图文详解NMT](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)（decoder那边有点错误，因为decoder的初始化的embedding 是<start> 估计是定义不通，然后初始化的用的是encoder的hidden output作为attention score的key，然后其实是concat context和embedding作为输入）
- [NMT代码](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/nmt_with_attention.ipynb#scrollTo=umohpBN2OM94&uniqifier=2)
- [pointer network](https://www.youtube.com/watch?v=VdOyqNQ9aww)
- [pointer slides](https://blog.csdn.net/jiaojiaolou/article/details/90484955)
- [All Attention You Need](https://blog.csdn.net/Datawhale/article/details/103231686)











## fairseq
+ https://github.com/pytorch/fairseq


# CV
## CNNs
+ ByteNet
+ ConvS2S
+ Entend Neural GPU


# Reference

- Survey
  - [Attention, please! A Critical Review  of Neural Attention Models in Natural Language Processing](https://arxiv.org/pdf/1902.02181.pdf)
  - [An Attentive Survey of Attention Models](https://arxiv.org/pdf/1904.02874.pdf)
    ![](https://ws3.sinaimg.cn/large/006tNc79ly1g1vd70uok6j30lk0a5q4x.jpg)
- Survey on Advanced Attention-based Models
- Recurrent Models of Visual Attention (2014.06.24)  
- Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (2015.02.10)
- DRAW: A Recurrent Neural Network For Image Generation (2015.05.20)
- Teaching Machines to Read and Comprehend (2015.06.04)
- Learning Wake-Sleep Recurrent Attention Models (2015.09.22)
- Action Recognition using Visual Attention (2015.10.12)
- Recursive Recurrent Nets with Attention Modeling for OCR in the Wild (2016.03.09)
- 模型汇总16 各类Seq2Seq模型对比及《Attention Is All You Need》中技术详解
- https://zhuanlan.zhihu.com/p/27485097
- 哈佛NLP组论文解读:基于隐变量的注意力模型|附开源代码


# Seq2Seq


# Memory

+ http://www.shuang0420.com/2017/12/04/论文笔记

## Facebook AI：

- 2015年提出MEMORY NETWORKS,使用记忆网络增强记忆。（引用数：475）
- 2015年提出End-To-End Memory Networks，针对上一篇文章中存在的无法端到端训练的问题，提出了端到端的记忆网络。（引用数：467）
- 2016年提出Key-Value Memory Networks for Directly Reading Documents，在端到端的基础上增加记忆的规模。（引用数：68）
- 2017年提出TRACKING THE WORLD STATE WITH RECURRENT ENTITY NETWORKS，论文提出了一种新的动态记忆网络，其使用固定长度的记忆单元来存储世界上的实体，每个记忆单元对应一个实体，主要存储该实体相关的属性（譬如一个人拿了什么东西，在哪里，跟谁等等信息），且该记忆会随着输入内容实时更新。（引用数：27）

## Google DeepMind:

- 2014年提出Neural Turing Machines，神经图灵机，同facebook团队的记忆网络一样，是开篇之作。（引用数：517）
- 2015年提出Neural Random Access Machines,神经网络随机存取机。（引用数：55）
- 2015年提出Learning to Transduce with Unbounded Memory,使用诸如栈或（双端）队列结构的连续版本。（引用数：99）
- 2016年提出Neural GPUs Learn Algorithms,神经网络GPU,使用了带有读写磁头的磁带。（引用数：86）