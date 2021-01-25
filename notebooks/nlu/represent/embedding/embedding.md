<!-- TOC -->

1. [Category](#category)
    1. [Static and Dynamic](#static-and-dynamic)
        1. [Static](#static)
        2. [Dynamic](#dynamic)
    2. [AR and AE](#ar-and-ae)
        1. [Auto-Regressive LM:AR](#auto-regressive-lmar)
        2. [Auto-Encoder LM:AE](#auto-encoder-lmae)
        3. [AR+AE](#arae)
2. [Metric](#metric)
    1. [wordsim353](#wordsim353)
    2. [半监督](#半监督)
    3. [Ref](#ref)
3. [Word2Vec](#word2vec)
    1. [词向量基础](#词向量基础)
    2. [CBOW](#cbow)
        1. [Naïve implement](#naïve-implement)
        2. [optimized methods](#optimized-methods)
            1. [Hierarchical Softmax](#hierarchical-softmax)
            2. [Negative Sampling](#negative-sampling)
    3. [Skip-Gram](#skip-gram)
        1. [Naïve implement](#naïve-implement-1)
        2. [optimized methods](#optimized-methods-1)
            1. [Hierarchical Softmax](#hierarchical-softmax-1)
            2. [Negative sampling](#negative-sampling)
    4. [FastText词向量与word2vec对比](#fasttext词向量与word2vec对比)
    5. [ref:](#ref)
4. [Glove](#glove)
    1. [Glove](#glove-1)
    2. [实现](#实现)
    3. [如何训练](#如何训练)
    4. [Glove和LSA以及Word2vec的比较](#glove和lsa以及word2vec的比较)
    5. [公式推导](#公式推导)
5. [Cove](#cove)
6. [ELMo](#elmo)
    1. [Tips](#tips)
    2. [Bidirectional language models（biLM）](#bidirectional-language-modelsbilm)
    3. [Framework](#framework)
    4. [Evaluation](#evaluation)
    5. [Analysis](#analysis)
    6. [Feature-based](#feature-based)
7. [ULM-Fit](#ulm-fit)
8. [GPT-2](#gpt-2)
    1. [Tips](#tips-1)
    2. [Unsupervised-Learning](#unsupervised-learning)
    3. [Supervised-Learning](#supervised-learning)
    4. [Task specific input transformation](#task-specific-input-transformation)
9. [BERT](#bert)
    1. [Tips](#tips-2)
    2. [Motivation](#motivation)
    3. [Pretrain-Task 1 : Masked LM](#pretrain-task-1--masked-lm)
    4. [Pretrain-task 2 : Next Sentence Prediction](#pretrain-task-2--next-sentence-prediction)
    5. [Fine Tune](#fine-tune)
    6. [Experiment](#experiment)
    7. [View](#view)
    8. [Abstract](#abstract)
    9. [Introduction](#introduction)
        1. [预训练language representation 的两种策略](#预训练language-representation-的两种策略)
        2. [Contributions of this paper](#contributions-of-this-paper)
    10. [Related Work](#related-work)
    11. [Train Embedding](#train-embedding)
        1. [Model Architecture](#model-architecture)
        2. [Input](#input)
        3. [Loss](#loss)
    12. [Use Bert for Downstream Task](#use-bert-for-downstream-task)
10. [BERT-WWW](#bert-www)
11. [ERNIE - 百度](#ernie---百度)
    1. [ERNIE - 清华/华为](#ernie---清华华为)
        1. [把英文字变成中文词](#把英文字变成中文词)
        2. [使用TransE 编码知识图谱](#使用transe-编码知识图谱)
12. [MASS](#mass)
    1. [Tips](#tips-3)
    2. [Framework](#framework-1)
    3. [Experiment](#experiment-1)
    4. [Advantage of MASS](#advantage-of-mass)
    5. [Reference](#reference)
13. [Uni-LM](#uni-lm)
14. [XLNet](#xlnet)
15. [Doc2Vec](#doc2vec)
16. [Tools](#tools)
    1. [gensim](#gensim)
17. [Reference](#reference-1)

<!-- /TOC -->

# Category

## Static and Dynamic

### Static

+ Word2Vec
+ Glove

### Dynamic 

+ Cove
+ ELMo
+ GPT
+ BERT

## AR and AE 

### Auto-Regressive LM:AR

+ N-Gram LM
+ NNLM
+ RNNLM
+ GPT
+ Transformer
+ ELMo

### Auto-Encoder LM:AE

+ W2V
+ BERT

### AR+AE

+ XLNet



# Metric

## wordsim353
+ 当前绝大部分工作（比如以各种方式改进word embedding）都是依赖wordsim353等词汇相似性数据集进行相关性度量，并以之作为评价word embedding质量的标准
+ 然而，这种基于similarity的评价方式对训练数据大小、领域、来源以及词表的选择非常敏感。而且数据集太小，往往并不能充分说明问题。

## 半监督
+ seed + pattern

## Ref
+ Evaluation of Word Vector Representations by Subspace Alignment (Tsvetkov et al.)
+ Evaluation methods for unsupervised word embeddings (Schnabel et al.)


# Word2Vec

+ word2vector 是将词向量进行表征，其实现的方式主要有两种，分别是CBOW（continue bag of words) 和 Skip-Gram两种模型。这两种模型在word2vector出现之前，采用的是DNN来训练词与词之间的关系，采用的方法一般是三层网络，输入层，隐藏层，和输出层。之后，这种方法在词汇字典量巨大的时候，实现方式以及计算都不现实，于是采用了hierarchical softmax 或者negative sampling模型进行优化求解。
![IMAGE](images/mindmap.jpg)

## 词向量基础
+ 用词向量来表示词并不是word2vec的首创，在很久之前就出现了。最早的词向量是很冗长的，它使用是词向量维度大小为整个词汇表的大小，对于每个具体的词汇表中的词，将对应的位置置为1。比如我们有下面的5个词组成的词汇表，词"Queen"的序号为2， 那么它的词向量就是(0,1,0,0,0)。同样的道理，词"Woman"的词向量就是(0,0,0,1,0)。这种词向量的编码方式我们一般叫做1-of-N representation或者one hot representation.
![IMAGE](https://images2015.cnblogs.com/blog/1042406/201707/1042406-20170713145606275-2100371803.png)
+ one hot representation 的优势在于简单，但是其也有致命的问题，就是在动辄上万的词汇库中，one hot表示的方法需要的向量维度很大，而且对于一个字来说只有他的index位置为1其余位置为0，表达效率不高。而且字与字之间是独立的，不存在字与字之间的关系。
+ 如何将字的维度降低到指定的维度大小，并且获取有意义的信息表示，这就是word2vec所做的事情。
+ 比如下图我们将词汇表里的词用"Royalty","Masculinity", "Femininity"和"Age"4个维度来表示，King这个词对应的词向量可能是(0.99,0.99,0.05,0.7)。当然在实际情况中，我们并不能对词向量的每个维度做一个很好的解释
![IMAGE](images/embd_vis.png)
+ 有了用Distributed Representation表示的较短的词向量，我们就可以较容易的分析词之间的关系了，比如我们将词的维度降维到2维，有一个有趣的研究表明，用下图的词向量表示我们的词时，我们可以发现：
$\vec{Queen} = \vec{King} - \vec{Man} + \vec{Woman}$
![IMAGE](https://images2015.cnblogs.com/blog/1042406/201707/1042406-20170713151608181-1336632086.png)

## CBOW
+ CBOW 模型的输入是一个字上下文的，指定窗口长度，根据上下文预测该字。
+ 比如下面这段话，我们上下文取值为4，特定词为`learning`，上下文对应的词共8个，上下各四个。这8个词作为我们的模型输入。CBOW使用的是词袋模型，这8个词都是平等的，我们不考虑关注的词之间的距离大小，只要是我们上下文之内的就行。
![IMAGE](https://images2015.cnblogs.com/blog/1042406/201707/1042406-20170713152436931-1817493891.png)
+ CBOW模型的训练输入是某一个特征词的上下文相关的词对应的词向量，而输出就是这特定的一个词的词向量。

### Naïve implement
+ 这样我们这个CBOW的例子里，我们的输入是8个词向量，输出是所有词的softmax概率（训练的目标是期望训练样本特定词对应的softmax概率最大），对应的CBOW神经网络模型**输入层有8个神经元（#TODO：check），输出层有词汇表大小V个神经元**。隐藏层的神经元个数我们可以自己指定。通过DNN的反向传播算法，我们可以求出DNN模型的参数，同时得到所有的词对应的词向量。这样当我们有新的需求，要求出某8个词对应的最可能的输出中心词时，我们可以通过一次DNN前向传播算法并通过softmax激活函数找到概率最大的词对应的神经元即可。

![IMAGE](images/cbow.jpg)

+ 我们输入的词是特定词，输出是softmax概率前8的8个词，对应的SkipGram模型有1个神经元，输出层有词汇表个神经元大小，

### optimized methods
+ word2vec为什么不用现成的DNN模型，要继续优化出新方法呢？最主要的问题是DNN模型的这个处理过程非常耗时。我们的词汇表一般在百万级别以上，这意味着我们DNN的输出层需要进行softmax计算各个词的输出概率的的计算量很大。有没有简化一点点的方法呢？
> word2vec基础之霍夫曼树
       word2vec也使用了CBOW与Skip-Gram来训练模型与得到词向量，但是并没有使用传统的DNN模型。最先优化使用的数据结构是用霍夫曼树来代替隐藏层和输出层的神经元，霍夫曼树的叶子节点起到输出层神经元的作用，叶子节点的个数即为词汇表的小大。 而内部节点则起到隐藏层神经元的作用。
　　　　具体如何用霍夫曼树来进行CBOW和Skip-Gram的训练我们在下一节讲，这里我们先复习下霍夫曼树。
　　　　霍夫曼树的建立其实并不难，过程如下：
　　　　输入：权值为$(w1,w2,...wn)$的n个节点
　　　　输出：对应的霍夫曼树
　　　　1）将$(w1,w2,...wn)$看做是有n棵树的森林，每个树仅有一个节点。
　　　　2）在森林中选择根节点权值最小的两棵树进行合并，得到一个新的树，这两颗树分布作为新树的左右子树。新树的根节点权重为左右子树的根节点权重之和。
　　　　3） 将之前的根节点权值最小的两棵树从森林删除，并把新树加入森林。
　　　　4）重复步骤2）和3）直到森林里只有一棵树为止。
　　　　下面我们用一个具体的例子来说明霍夫曼树建立的过程，我们有(a,b,c,d,e,f)共6个节点，节点的权值分布是(20,4,8,6,16,3)。
　　　　首先是最小的b和f合并，得到的新树根节点权重是7.此时森林里5棵树，根节点权重分别是20,8,6,16,7。此时根节点权重最小的6,7合并，得到新子树，依次类推，最终得到下面的霍夫曼树。
那么霍夫曼树有什么好处呢？一般得到霍夫曼树后我们会对叶子节点进行霍夫曼编码，由于权重高的叶子节点越靠近根节点，而权重低的叶子节点会远离根节点，这样我们的高权重节点编码值较短，而低权重值编码值较长。这保证的树的带权路径最短，也符合我们的信息论，即我们希望越常用的词拥有更短的编码。如何编码呢？一般对于一个霍夫曼树的节点（根节点除外），可以约定左子树编码为0，右子树编码为1.如上图，则可以得到c的编码是00。
　　　　**在word2vec中，约定编码方式和上面的例子相反，即约定左子树编码为1，右子树编码为0，同时约定左子树的权重不小于右子树的权重。**
      ![IMAGE](https://img2018.cnblogs.com/blog/1042406/201812/1042406-20181205104643781-71258001.png)

- 在传统的DNN模型中，由于词汇表很大，softmax的计算量很大，要计算每个词的softmax的概率，再找出最大的值。

![IMAGE](https://blog.aylien.com/wp-content/uploads/2016/10/cbow.png)

- word2vector对这个模型进行了改进。
  - 首先对于**输入层到隐藏层的映射**，没有采用传统的神经网络的线性加权加激活函数，而是直接采用了**简单的加和平均**。
    - 比如输入的是三个4维词向量：$(1,2,3,4),(9,6,11,8),(5,10,7,12)$,那么我们word2vec映射后的词向量就是$(5,6,7,8)$。由于这里是从多个词向量变成了一个词向量。
- 第二个改进就是**隐藏层到输出层**的**softmax的计算量**进行了改进，为了避免:计算所有词的softmax的概率
- word2vector采用了两种方式进行改进，分别为hierarchical softmax和negative sampling。降低了计算复杂度。


#### Hierarchical Softmax

![IMAGE](https://miro.medium.com/max/600/0*hUAFJJOBG3D0PgKl.)

- 在计算之前，先计算统计出一颗Huffman树
- 我们通过将softmax的值的计算转化为huffman树的树形结构计算，如下图所示，我们可以沿着霍夫曼树从根节点一直走到我们的叶子节点的词$w_2$

![IMAGE](https://images2017.cnblogs.com/blog/1042406/201707/1042406-20170727105752968-819608237.png)
 
  - 跟神经网络类似，根结点的词向量为contex的词投影（加和平均后）的词向量
  - 所有的叶子节点个数为Vocabulary size
  - 中间节点对应神经网络的中间参数，类似之前神经网络隐藏层的神经元,
  - 通过投影层映射到输出层的softmax结果，是根据huffman树一步步完成的
   - $p(w_2) = p_1(-)p_2(-)p_3(+) = (1-\sigma(x_{w_2}\theta_1)) (1-\sigma(x_{w_2}\theta_2)) \sigma(x_{w_2}\theta_3) $
- 如何通过huffman树一步步完成softmax结果，采用的是logistic regression。规定沿着左子树走，代表负类（huffman code = 1），沿着右子树走，代表正类（huffman code=0）。
  - $P(+)=\sigma(x^Tw_θ)=\frac{1}{1+e^{−x^Tw_θ}}$
  - $P(-)=1-\sigma(x^Tw_θ)=1-\frac{1}{1+e^{−x^Tw_θ}}$
  - 其中$x_w$ 是当前内部节点的词向量，$\theta$是利用训练样本求出的lr的模型参数
- Huffman树的好处
  - 计算量从V降低到了$logV$，并且使用huffman树，越高频的词出现的深度越浅，计算所需的时间越短，例如`的`作为target词，其词频高，树的深度假设为2，那么其的softmax值就只有2项
  - 被划为左子树还是右子树取决于$P(-) or P(+)$, 主要取决与$\theta$ 和$x_{w_i}$
  
- 如何训练？
   - 目标是是的所有合适的节点的词向量$x_{w_i}$以及内部的节点$\theta$ 使得训练样本达到最大似然
   - 分别对$x_{w_i}$ 以及$\theta$求导
   例子：以上面$w_2$作为例子
$\prod_{i=1}^{3}P_i=(1−\frac{1}{1+e^{−x^Tw_{θ_1}}})(1−\frac{1}{1+e^{−x^Tw_{θ_2}}})\frac{1}{e^{−{x^Tw_{θ_3}}}}$
对于所有的训练样本，我们期望最大化所有样本的似然函数乘积。
我们定义输入的词为$w$，其从输入层向量平均后输出为$x_w$，从根结点到$w$所在的叶子节点，包含的节点数为$l_w$个，而该节点对应的模型参数为$\theta_i^w$, 其中i=1,2,....$l_w-1$，没有$l_w$，因为他是模型参数仅仅针对与huffman树的内部节点
  - 定义$w$经过的霍夫曼树某一个节点j的逻辑回归概率为$P(d_j^w|x_w,\theta_{j-1}^w)$，其表达式为：

    \begin{equation}
    P(d_j^w|x_w,\theta_{j-1}^w)=\left\{
    \begin{aligned}
    \sigma(x^T_w\theta^w_{j-1}) &  & d_j^w = 0 \\
    1-\sigma(x^T_w\theta^w_{j-1}) &  & d_j^w = 1 
    \end{aligned}
    \right.
    \end{equation}
    
  - 对于某个目标词$w$， 其最大似然为：
  $\prod_{j=2}^{l_w}P(d_j^w|x_w,\theta_{j-1}^w)=\prod_{j=2}^{l_w}[(\frac{1}{1+e^{−x_w^Tw_{θ_{j-1}}}})]^{1-d_j^w}[1-\frac{1}{e^{−{x^Tw_{θ_{j-1}}}}}]^{d_j^w}$

  - 采用对数似然
  $ L=log\prod_{j=2}^{l_w}P(d^w_j|x_w,θ^w_{j−1})=\sum_{j=2}^{l_w}((1−d^w_j)log[\sigma(x^T_wθ_{w_{j-1}})]+d_{w_j}log[1−\sigma(x^T_{w}θ_{w_{j−1}})])$
要得到模型中$w$词向量和内部节点的模型参数$θ$, 我们使用梯度上升法即可。首先我们求模型参数$θ^w_{j−1}$的梯度：
![IMAGE](quiver-image-url/5F2C09CCF3FFD5C43D99C460F374FDA8.jpg =786x132)
同样的方法，可以求出$x_w$的梯度表达式如下：![IMAGE](quiver-image-url/DEF080F577A2BBDC416ECF7F1A8E20F0.jpg =510x84)
有了梯度表达式，我们就可以用梯度上升法进行迭代来一步步的求解我们需要的所有的$θ^w_{j−1}$和$x_w$。

- 基于Hierarchical Softmax的CBOW模型
  - 首先我们先定义词向量的维度大小M，以及CBOW的上下文2c，这样我们对于训练样本中的每一个词，其前面的c个词和后c个词都是CBOW模型的输入，而该词作为样本的输出，期望其softmax最大。
  - 在此之前，我们需要先将词汇表构建成一颗huffman树
  - 从输入层到投影层，需要对2c个词的词向量进行加和平均
  $x_w = \frac{1}{2c}\sum^{2c}_{i=1}x_i$
  - 通过梯度上升更新$\theta^w_{j-1}$和$x_w$， 注意这边的$x_w$是多个向量加和，所以需要对每个向量进行各自的更新，我们做梯度更新完毕后会用梯度项直接更新原始的各个$x_i(i=1,2,,,,2c)$
      $\theta_{j-1}^w = \theta_{j-1}^w+\eta (1−d^w_j−\sigma(x^T_w\theta^w_{j−1}))x_w \forall j=2 \ to \ l_w $ 
      $x_i = x_i + \eta\sum^{l_w}_{j=2}(1-d^w_j-\sigma(x^T_w\theta^w_{j-1}))\theta^w_{j-1} \forall i = \ 1 \ to 2c$
    Note: $\eta$ 是learning rate
  ```
  Input：基于CBOW的语料训练样本，词向量的维度大小M，CBOW的上下文大小2c 步长η
  Output：霍夫曼树的内部节点模型参数θ，**所有的词向量w**
  1. create huffman tree based on the vocabulary
  2. init all θ, and all embedding w
  3. updage all theta adn emebedding w based on the gradient ascend
  ```

  `for all trainning sample (context(w), w) do:`
  
   > $e = 0 , x_w = \sum_{i=1}^{2c}x_i$
          
    - `for j=2..l_w:`
  
      > $ g = 1−d^w_j−\sigma(x^T_w\theta^w_{j−1})$
    $e = e+ g*\theta^w_{j-1}$
    $\theta^w_{j-1} = = \theta_{j-1}^w+\eta_\theta*g*x_w$
   
    - `for all x_i in context(w), update x_i :`
      > $x_i = x_i + e$
      d) 如果梯度收敛，则结束梯度迭代，否则回到步骤3继续迭代。
 ------
 
#### Negative Sampling
采用hsoftmax在生僻字的情况下，仍然可能出现树的深度过深，导致softmax计算量过大的问题。如何解决这个问题，negative sampling在和h-softmax的类似，采用的是将多分类问题转化为多个2分类，至于多少个2分类，这个和negative sampling的样本个数有关。
- negative sampling放弃了Huffman树的思想，采用了负采样。比如我们有一个训练样本，中心词是w,它周围上下文共有2c个词，记为context(w)。由于这个中心词w,的确和context(w)相关存在，因此它是一个真实的正例。通过Negative Sampling采样，我们得到neg个和w不同的中心词wi,i=1,2,..neg，这样context(w)和$w_i$就组成了neg个并不真实存在的负例。利用这一个正例和neg个负例，我们进行二元逻辑回归，得到负采样对应每个词$w_i$对应的模型参数$θ_i$，和每个词的词向量。
- 如何通过一个正例和neg个负例进行logistic regression
  - 正例满足：$P(context(w_0), w_i) = \sigma(x_0^T\theta^{w_i}) \  y_i=1, i=0$
  - 负例满足：$P(context(w_0), w_i) = 1- \sigma(x_0^T\theta^{w_i}) \ y_i=0, i=1..neg$
  - 期望最大化：$\Pi^{neg}_{i=0}P(context(w_0), w_i) =\Pi^{neg}_{i=0}  [\sigma(x_0^T\theta^{w_i})]^y_i[1- \sigma(x_0^T\theta^{w_i})]^{1-y_i} $
   对数似然为：$log\Pi^{neg}_{i=0}P(context(w_0), w_i) =\sum^{neg}_{i=0}  y_i*log[\sigma(x_0^T\theta^{w_i})]+(1-y_i)*log[1- \sigma(x_0^T\theta^{w_i})] $
 - 和Hierarchical Softmax类似，我们采用随机梯度上升法，仅仅每次只用一个样本更新梯度，来进行迭代更新得到我们需要的$x_{w_i},θ^{w_i},i=0,1,..neg$, 这里我们需要求出$x_{w_0},θ^{w_i},i=0,1,..neg$的梯度。
  - $θ^{w_i}$:
  ![IMAGE](quiver-image-url/95D887BE46AC5DD376055CA49FCE5D30.jpg =582x78)
  - $x_{w_0}$:
  ![IMAGE](quiver-image-url/C83C84AB4F64858A850697D10B9E09C9.jpg =591x70)
- 如何采样负例
  负采样的原则采用的是根据词频进行采样，词频越高被采样出来的概率越高，词频越低，被采样出来的概率越低，符合文本的规律。word2vec采用的方式是将一段长度为1的线段，分为V(ocabulary size)份，每个词对应的长度为:

    $len(w)=\frac{Count(w)}{\sum_{u\in V}Count(u)}$
  在word2vec中，分子和分母都取了3/4次幂如下：
   $len(w)=\frac{Count(w)^{3/4}}{[\sum_{u\in V}Count(u)]^{3/4}}$
   在采样前，我们将这段长度为1的线段划分成M等份，这里M>>V，这样可以保证每个词对应的线段都会划分成对应的小块。而M份中的每一份都会落在某一个词对应的线段上。在采样的时候，我们只需要从M个位置中采样出neg个位置就行，此时采样到的每一个位置对应到的线段所属的词就是我们的负例词。
   ![IMAGE](https://images2017.cnblogs.com/blog/1042406/201707/1042406-20170728152731711-1136354166.png)在word2vec中，M取值默认为 $10^8$
   
----

  ```
  Input：基于CBOW的语料训练样本，词向量的维度大小M，CBOW的上下文大小2c 步长η，负采样的个数neg
  Output：词汇表中每个词对应的参数θ，**所有的词向量$w$**
  1. init all θ, and all embedding w
  2. sample neg negtive words w_i, i =1,2,..neg
  3. updage all theta adn emebedding w based on the gradient ascend
  ```

`for all trainning sample (context(w), w) do:`
  
   > $e = 0 , x_w = \frac{1}{2c}\sum_{i=0}^{2c}x_i$
          
  - `for i=0..neg:`
  
      > $ g =\eta*(y_i−\sigma(x^T_{w}\theta^{w_i}))$
    $e = e+ g*\theta^{w_i}$
    $\theta^{w_i} = = \theta^{w_i}+g*x_{w}$
   
  - `for all x_k in context(w) (2c in total), update x_k :`
      > $x_k = x_k + e$
      d) 如果梯度收敛，则结束梯度迭代，否则回到步骤3继续迭代
      

## Skip-Gram
Skip gram 跟CBOW的思路相反，根据输入的特定词，确定对应的上下文词词向量作为输出。 

![IMAGE](https://images2015.cnblogs.com/blog/1042406/201707/1042406-20170713152436931-1817493891.png)

这个例子中，`learning`作为输入，而上下文8个词是我们的输出。

### Naïve implement
我们输入是特定词，输出是softmax概率前8的8个词，对应的SkipGram神经网络模型，**输入层有1个神经元，输出层有词汇表个神经元。**[#TODO check??]，隐藏层个数由我们自己指定。通过DNN的反向传播算法，我们可以求出DNN模型的参数，同时得到所有的词对应的词向量。

![IMAGE](images/skipgram.jpg)

### optimized methods

#### Hierarchical Softmax
  - 首先我们先定义词向量的维度大小M，此时我们输入只有一个词，我们希望得到的输出context(w）2c个词概率最大。
  - 在此之前，我们需要先将词汇表构建成一颗huffman树
  - 从输入层到投影层，就直接是输入的词向量
  - 通过梯度上升更新$\theta^w_{j-1}$和$x_w$， 注意这边的$x_w$周围有2c个词向量，此时我们希望$P(x_i|x_w, i=1,2,..2c$最大，此时我们注意到上下文是相互的，我们可以认为$P(x_w|x_i), i=1,2,..2c$也是最大的，对于那么是使用$P(x_i|x_w)$
好还是$P(x_w|x_i)$好呢，word2vec使用了后者，这样做的好处就是在一个迭代窗口内，我们不是只更新xw一个词，而是xi,i=1,2...2c共2c个词。不同于CBOW对于输入进行更新，SkipGram对于输出进行了更新。
  
  - $x_i(i=1,2,,,,2c)$
      $\theta_{j-1}^w = \theta_{j-1}^w+\eta (1−d^w_j−\sigma(x^T_w\theta^w_{j−1}))x_w \forall j=2 \ to \ l_w $ 
      $x_i = x_i + \eta\sum^{l_w}_{j=2}(1-d^w_j-\sigma(x^T_w\theta^w_{j-1}))\theta^w_{j-1} \forall i = \ 1 \ to 2c$
    Note: $\eta$ 是learning rate
  ```
  Input：基于Skip-Gram的语料训练样本，词向量的维度大小M，Skip-Gram的上下文大小2c 步长η
  Output：霍夫曼树的内部节点模型参数θ，**所有的词向量w**
  1. create huffman tree based on the vocabulary
  2. init all θ, and all embedding w
  3. updage all theta and emebedding w based on the gradient ascend
  ```

`for all trainning sample (w, context(w)) do:`
- `for i = 1..2c`
    > $e = 0 $
          
   - `for j=2..l_w:`
       > $ g = 1−d^w_j−\sigma(x^T_w\theta^w_{j−1})$
    $e = e+ g*\theta^w_{j-1}$
    $\theta^w_{j-1} = = \theta_{j-1}^w+\eta_\theta*g*x_w$
  
       >$x_i = x_i + e$
b)如果梯度收敛，则结束梯度迭代，算法结束，否则回到步骤a继续迭代。

#### Negative sampling
有了上一节CBOW的基础和上一篇基于Hierarchical Softmax的Skip-Gram模型基础，我们也可以总结出基于Negative Sampling的Skip-Gram模型算法流程了。梯度迭代过程使用了随机梯度上升法：
  ```
  Input：基于Skip-Gram的语料训练样本，词向量的维度大小M，Skip-Gram的上下文大小2c 步长η, 负采样的个数为neg
  Output：词汇表中每个词对应的参数θ，**所有的词向量w**
  1. init all θ, and all embedding w
  2. for all context(w_0, w_0), sample neg negative words w_i, i =1,2..neg
  3. updage all theta and emebedding w based on the gradient ascend
  ```


`for all trainning sample (w, context(w)) do:`
- `for i = 1..2c`
    > $e = 0 $
          
   - `for j=0..neg:`
     > $ g =\eta*(y_i−\sigma(x^T_{w}\theta^{w_j}))$
    $e = e+ g*\theta^{w_j}$
    $\theta^{w_j} = = \theta^{w_j}+g*x_{w_i}$
  
    >$x_i = x_i + e$
b)如果梯度收敛，则结束梯度迭代，算法结束，否则回到步骤a继续迭代。

## FastText词向量与word2vec对比 
  - FastText= word2vec中 cbow + h-softmax的灵活使用
  - 灵活体现在两个方面： 
    1. 模型的输出层：word2vec的输出层，对应的是每一个term，计算某term的概率最大；而fasttext的输出层对应的是 分类的label。不过不管输出层对应的是什么内容，起对应的vector都不会被保留和使用； 
    2. 模型的输入层：word2vec的输出层，是 context window 内的term；而fasttext 对应的整个sentence的内容，包括term，也包括 n-gram的内容；
  - 两者本质的不同，体现在 h-softmax的使用。 
    - Wordvec的目的是得到词向量，该词向量 最终是在输入层得到，输出层对应的 h-softmax 也会生成一系列的向量，但最终都被抛弃，不会使用。 
    - fasttext则充分利用了h-softmax的分类功能，遍历分类树的所有叶节点，找到概率最大的label（一个或者N个）
  - http://nbviewer.jupyter.org/github/jayantj/gensim/blob/683720515165a332baed8a2a46b6711cefd2d739/docs/notebooks/Word2Vec_FastText_Comparison.ipynb#
  - https://www.cnblogs.com/eniac1946/p/8818892.html 

## ref:

+ Distributed Representations of Sentences and Documents
+ Efficient estimation of word representations in vector space
+ [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf)
+ https://zhuanlan.zhihu.com/p/26306795
+ https://blog.aylien.com/overview-word-embeddings-history-word2vec-cbow-glove/
+ https://www.cnblogs.com/pinard/p/7160330.html
+ https://www.cnblogs.com/pinard/p/7243513.html
+ https://www.cnblogs.com/pinard/p/7249903.html

# Glove

- Count-based模型, 本质上是对共现矩阵进行降维
- 首先，构建一个词汇的共现矩阵，每一行是一个word，每一列是context。共现矩阵就是计算每个word在每个context出现的频率。
- 由于context是多种词汇的组合，其维度非常大，我们希望像network embedding一样，在context的维度上降维，学习word的低维表示。这一过程可以视为共现矩阵的重构问题，即reconstruction loss。

- http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdfß
## Glove
- 基于**全局词频统计**的词表征向量。它可以把一个单词表达成一个由实数组成的向量，这些向量捕捉到了单词之间一些语义特性，比如相似性（similarity）、类比性（analogy）等。我们通过对向量的运算，比如欧几里得距离或者cosine相似度，可以计算出两个单词之间的语义相似性。

## 实现
- 根据语料库（corpus）构建共现矩阵（Co-occurrence Matrix）$X$，矩阵中的每个元素$X_{ij}$ 代表单词$j$在单词$X_i$的特定上下文窗口(context window)中出现的次数。一般而言这个，这个矩阵中数值最小单位（每次最少+）为1，但是GloVe提出，根据两个单词的的上下文窗口的距离$d$，提出了一个$decay = \frac{1}{d}$ 用于计算权重，这也意味着，上下文距离越远，这两个单词占总计数的权重越小
> In all cases we use a decreasing weighting function, so that word pairs that are d words apart contribute 1/d to the total count.


- 构建词向量（word vector）和共现矩阵（co-occurrence matrix）之间的近似关系：
\begin{equation}
w_i^T \tilde w+ b_i + \tilde b_j = log(X_{ij})\tag{$1$}
\end{equation}
   - $w_i^T 和\tilde w$ 是我们最终的词向量 
   - $b_i 和\tilde b_j$分别是词向量的bias term
   - 之后会解释这个公式怎么来的

- 根据公式(1)，定义loss function，使用gradient descend 方式训练，得到$w$ 词向量
\begin{equation}
L = \sum_{i,j =1}^{V} f(X_{ij})(w^T_i \tilde w_j +b_i +\tilde b_j -log(X_{ij})^2 \tag{$2$}
\end{equation}
  - MSE loss
  - $f(X_{ij})$ 权重函数: 在语料库中肯定出现很多单词他们一起出现的次数是很多的（frequent cooccurrence) 我们希望：
    - 这些单词的权重要大于那些很少出现的单词（rare-occurrence），所以这个函数是一个非递减的函数
    - 我们希望这个权重不要太大（overweighted），当到达一定程度之后应该不要在增加
    - 如果两个单词没有出现过，$X_{ij}=0$, 我们不希望他参与到loss function的计算之中，也就是$f(x)=0$
    
\begin{equation}
f(x)=\left\{
\begin{aligned}
(x/x_{max})^\alpha &  & ifx <x_{max} \\
1 &  & otherwise    \tag{$3$}
\end{aligned}
\right.
\end{equation}
![IMAGE](quiver-image-url/0850340DCFCD774D14425196F2D09F83.jpg =469x296)
这篇论文中的所有实验，$\alpha$的取值都是0.75，而x_{max}取值都是100。以上就是GloVe的实现细节，那么GloVe是如何训练的呢？


## 如何训练
- unsupervised learning
- label是公式2中的$log(X_ij)$, 需要不断学习的是$w和 \tilde w$ 
- 训练方式采用的是梯度下降
- 具体：采用AdaGrad的梯度下降算法，对矩阵$X$中的所有非零元素进行随机采样，learning rate=0.05，在vector size小于300的情况下迭代50次，其他的vector size迭代100次，直至收敛。最终学习得到的$w \tilde w$，因为$X$ 是对称的，所以理论上$w 和\tilde w$也是对称的，但是初始化参数不同，导致最终值不一致，所以采用$(w +\tilde w)$ 两者之和 作为最后输出结果，提高鲁棒性
- 在训练了400亿个token组成的语料后，得到的实验结果如下图所示：
![IMAGE](quiver-image-url/6666E056417FF47C8110D93B7884D081.jpg =1005x325)
这个图一共采用了三个指标：语义准确度，语法准确度以及总体准确度。那么我们不难发现Vector Dimension在300时能达到最佳，而context Windows size大致在6到10之间。

## Glove和LSA以及Word2vec的比较
- LSA（Latent Semantic Analysis）：基于count-based的词向量表征工具，他是基于cooccurrence matrix的，但是他是基于SVD（奇异值分解）的矩阵分解技术对大矩阵进行降维，SVD的计算复杂度很大，所以他的计算代价很大。所有的单词的统计权重是一直的。
- Word2Vec：采用的是SkipGram或者CBOW的深度网络，训练数据是窗口内的数据，最大的缺点是它没有充分利用所有的语料的统计信息
- Glove：将两者的优点都结合起来，既运用了所有的词的统计信息，也增加了统计权重，同时也结合了梯度下降的训练方法使得计算复杂度降低。从这篇论文给出的实验结果来看，GloVe的性能是远超LSA和word2vec的，但网上也有人说GloVe和word2vec实际表现其实差不多。

## 公式推导
公式（1）怎么推导出来的呢？
- $X_{ij}$ 表示单词$j$出现在单词$i$的上下文之间的次数（乘以decay）
- $X_i$：单词$i$上下文的单词次数加和， $X_i = \sum^k {X_{ik}}$
- $P_{ij} = P(j|i) = X_{ij}/X_i$: 单词$j$出现在单词$i$上下文的概率
有了这些定义之后，我们来看一个表格：
![IMAGE](quiver-image-url/BC88729C261BE9FD68B73FF6D5510757.jpg =700x148)

最后一行：
- $P(k|i)/P(k|j)$: 表示的是这两个概率的的比值，我们可以使用它来观察出两个单词i和j相对于k哪个更相关，
- 如果$P(k|i)$和$P(k|j)$ 都不相关或者都很相关，$P(k|i)/P(k|j)$趋近于1
- $P(k|i)$相关$P(k|j)$不相关，$P(k|i)/P(k|j)$ 远大于1
- $P(k|i)$不相关$P(k|j)$相关，$P(k|i)/P(k|j)$ 远小于1

以上推断可以说明通过概率的比例而不是概率本身去学习词向量可能是一个更恰当的方式。

于是为了捕捉上面提到的概率比例，我们可以构造如下函数：

\begin{equation}
F(w_i, w_j, w_k) = \frac {P_{ik}} {P_{jk}}\tag{$4$}
\end{equation}
其中，函数$F$
的参数和具体形式未定，它有三个参数$w_i, w_j, w_k$是不同的向量
\begin{equation}
F((w_i- w_j)^T w_k) = \frac {P_{ik}} {P_{jk}}\tag{$5$}
\end{equation}
这时我们发现公式5的右侧是一个数量，而左侧则是一个向量，于是我们把左侧转换成两个向量的内积形式：
\begin{equation}
F((w_i- w_j)^T w_k) = \frac {F(w_i^T \tilde w_k)} {F(w_j^T \tilde w_k)} \tag{$6$}
\end{equation}
我们知道$X$
是个对称矩阵，单词和上下文单词其实是相对的，也就是如果我们做如下交换：$w <->\tilde w _k, X <-> X^T$公式6应该保持不变，那么很显然，现在的公式是不满足的。为了满足这个条件，首先，我们要求函数$F$要满足同态特性（homomorphism）：
\begin{equation}
 \frac {F(w_i^T \tilde w_k)} {F(w_j^T \tilde w_k)} = \frac {P_{ik}} {P_{jk}}\tag{$7$}
\end{equation}
结合公式6，我们可以得到：
\begin{equation}
\begin{split}
 F(w_i^T \tilde w_k) &= P_{ik} \\
                     &= \frac {X_{ik}}{X_i}\\
                   e^{w_i^T \tilde w_k}  &= \frac {X_{ik}}{X_i} 
\end{split}
\tag{$8$}
\end{equation}

然后，我们令$F = exp$， 于是我们有

\begin{equation}
\begin{split}
w^T_i\tilde w_k &= log(\frac {X_{ik}}{X_i})\\
  & = logX_{ik} - logX_{i}
\end{split}
\tag{$9$}
\end{equation}
此时，我们发现因为等号右侧的$log(X_i)$的存在，公式9是不满足对称性（symmetry）的，而且这个$log(X_i)$其实是跟$k$独立的，它只跟$i$有关，于是我们可以针对$w_i$增加一个bias term $b_i$把它替换掉，于是我们有：
\begin{equation}
w^T_i\tilde w_k +b_i = logX_{ik} \tag{$10$}
\end{equation}
但是公式10还是不满足对称性，于是我们针对$w_k$增加一个bias term $b_k$而得到公式1的形式
\begin{equation}
w^T_i\tilde w_k +b_i + b_k = logX_{ik} \tag{$1$}
\end{equation}

# Cove

# ELMo

## Tips

- Allen Institute / Washington University / NAACL 2018
- use
  - [ELMo](https://link.zhihu.com/?target=https%3A//allennlp.org/elmo)
  - [github](https://link.zhihu.com/?target=https%3A//github.com/allenai/allennlp)
  - Pip install allennlp

- a new type of contextualized word representation that model

  - 词汇用法的复杂性，比如语法，语义

  - 不同上下文情况下词汇的多义性

## Bidirectional language models（biLM）

- 使用当前位置之前的词预测当前词(正向LSTM)
- 使用当前位置之后的词预测当前词(反向LSTM)

## Framework

- 使用 biLM的所有层(正向，反向) 表示一个词的向量

- 一个词的双向语言表示由 2L + 1 个向量表示

- 最简单的是使用最顶层 类似TagLM 和 CoVe

- 试验发现，最好的ELMo是将所有的biLM输出加上normalized的softmax学到的权重 $$s = softmax(w)$$

  $$E(Rk;w, \gamma) = \gamma \sum_{j=0}^L s_j h_k^{LM, j}$$

  - $$ \gamma$$ 是缩放因子， 假如每一个biLM 具有不同的分布， $$\gamma$$  在某种程度上在weight前对每一层biLM进行了layer normalization

  ![](https://ws2.sinaimg.cn/large/006tNc79ly1g1v384rb0wj30ej06d0sw.jpg)

## Evaluation

![](https://ws4.sinaimg.cn/large/006tNc79ly1g1v3e0wyg7j30l909ntbr.jpg)

## Analysis



## Feature-based

+ 后在进行有监督的NLP任务时，可以将ELMo直接当做特征拼接到具体任务模型的词向量输入或者是模型的最高层表示上
+ 总结一下，不像传统的词向量，每一个词只对应一个词向量，ELMo利用预训练好的双向语言模型，然后根据具体输入从该语言模型中可以得到上下文依赖的当前词表示（对于不同上下文的同一个词的表示是不一样的），再当成特征加入到具体的NLP有监督模型里

# ULM-Fit

+ http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html

# GPT-2

## Tips

+ https://www.cnblogs.com/robert-dlut/p/9824346.html

+ GPT = Transformer + UML-Fit
+ GPT-2 = GPT + Reddit + GPUs
+ OpneAI 2018 
+ Improving Language Understanding by Generative Pre-Training
+ 提出了一种基于半监督进行语言理解的方法
  - 使用无监督的方式学习一个深度语言模型
  - 使用监督的方式将这些参数调整到目标任务上

+ GPT-2 predict next word
+ https://blog.floydhub.com/gpt2/
+ ![](https://paper-attachments.dropbox.com/s_972195A84441142620E4C92312EA63C9665C3A86AFFD1D713034FA568ADFC5F9_1555424144125_openai-transformer-language-modeling.png)

## Unsupervised-Learning

![](https://img2018.cnblogs.com/blog/670089/201810/670089-20181021105844156-2101267400.png)

## Supervised-Learning

+ 再具体NLP任务有监督微调时，与**ELMo当成特征的做法不同**，OpenAI GPT不需要再重新对任务构建新的模型结构，而是直接在transformer这个语言模型上的最后一层接上softmax作为任务输出层，然后再对这整个模型进行微调。额外发现，如果使用语言模型作为辅助任务，能够提升有监督模型的泛化能力，并且能够加速收敛

  ![](https://img2018.cnblogs.com/blog/670089/201810/670089-20181021105844634-618425800.png)

## Task specific input transformation

![](https://img2018.cnblogs.com/blog/670089/201810/670089-20181021105845000-829413930.png)

# BERT

## Tips

+ BERT predict the mask words
+ https://blog.floydhub.com/gpt2/

![](https://paper-attachments.dropbox.com/s_972195A84441142620E4C92312EA63C9665C3A86AFFD1D713034FA568ADFC5F9_1555424126367_BERT-language-modeling-masked-lm.png)

## Motivation

## Pretrain-Task 1 : Masked LM

## Pretrain-task 2 : Next Sentence Prediction

## Fine Tune

## Experiment

## View

- 可视化
  - https://www.jiqizhixin.com/articles/2018-1-21
- load bert checkpoint
  - https://blog.csdn.net/wshzd/article/details/89640269

## Abstract

- 核心思想
  - 通过所有层的上下文来预训练深度双向的表示
- 应用
  - 预训练的BERT能够仅仅用一层output layer进行fine-turn, 就可以在许多下游任务上取得SOTA(start of the art) 的结果, 并不需要针对特殊任务进行特殊的调整

## Introduction

- 使用语言模型进行预训练可以提高许多NLP任务的性能
  - Dai and Le, 2015
  - Peters et al.2017, 2018
  - Radford et al., 2018
  - Howard and Ruder, 2018
- 提升的任务有
  - sentence-level tasks(predict the relationships between sentences)
    - natural language inference
      - Bowman et al., 2015
      - Williams et al., 2018
    - paraphrasing(释义)
      - Dolan and Brockett, 2005
  - token-level tasks(models are required to produce fine-grained output at token-level)
    - NER
      - Tjong Kim Sang and De Meulder, 2003
    - SQuAD question answering

### 预训练language representation 的两种策略

- feature based
  - ELMo(Peters et al., 2018) [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
    - use **task-specific** architecture that include pre-trained representations as additional features representation
    - use shallow concatenation of independently trained left-to-right and right-to-left LMs
- fine tuning
  - Generative Pre-trained Transformer(OpenAI GPT) [Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)
    - introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning the pre-trained parameters
    - left-to-right

### Contributions of this paper

- 解释了双向预训练对Language Representation的重要性
  - 使用 MLM 预训练 深度双向表示
  - 与ELMo区别
- 消除(eliminate)了 繁重的task-specific architecture 的工程量
  - BERT is the first fine-tuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many systems with task-specific architectures
  - extensive ablations
    - goo.gl/language/bert

## Related Work

- review the most popular approaches of pre-training general language represenattions
- Feature-based Appraoches
  - non-neural methods
    - pass
  - neural methods
    - pass
  - coarser granularities(更粗的粒度)
    - sentence embedding
    - paragrqph embedding
    - As with traditional word embeddings,these learned representations are also typically used as features in a downstream model.
  - ELMo
    - 使用biLM(双向语言模型) 建模
      - 单词的复杂特征
      - 单词的当前上下文中的表示
    - ELMo advances the state-of-the-art for several major NLP bench- marks (Peters et al., 2018) including question 
      - answering (Rajpurkar et al., 2016) on SQuAD
      - sentiment analysis (Socher et al., 2013)
      - and named entity recognition (Tjong Kim Sang and De Meul- der, 2003).
- Fine tuning Approaches
  - 在LM进行迁移学习有个趋势是预训练一些关于LM objective 的 model architecture, 在进行有监督的fine-tuning 之前
  - The advantage of these approaches is that few parameters need to be learned from scratch
  - OpenAI GPT (Radford et al., 2018) achieved previously state-of-the-art results on many sentencelevel tasks from the GLUE benchmark (Wang et al., 2018).
- Transfer Learning from Supervised Data 
  - 无监督训练的好处是可以使用无限制的数据
  - 有一些工作显示了transfer 对监督学习的改进
    - natural language inference (Conneau et al., 2017)
    - machine translation (McCann et al., 2017)
  - 在CV领域, transfer learning 对 预训练同样发挥了巨大作用
    - Deng et al.,2009; Yosinski et al., 2014

## Train Embedding

### Model Architecture

- [Transformer](https://github.com/Apollo2Mars/Algorithms-of-Artificial-Intelligence/blob/master/3-1-Deep-Learning/1-Transformer/README.md)

- BERT v.s. ELMo v.s. OpenGPT

  ![img](https://ws2.sinaimg.cn/large/006tKfTcly1g1ima1j4wjj30k004ydge.jpg)

### Input

- WordPiece Embedding
  - WordPiece是指将单词划分成一组有限的公共子词单元，能在单词的有效性和字符的灵活性之间取得一个折中的平衡，例如下图中‘playing’被拆分成了‘play’和‘ing’
- Position Embedding
  - 讲单词的位置信息编码成特征向量
- Segment Embedding
  - 用于区别两个句子，例如B是否是A的下文(对话场景，问答场景)，对于句子对，第一个句子的特征值是0，第二个句子的特征值是1

![img](https://ws4.sinaimg.cn/large/006tNc79ly1g2ql45wou8j30k005ydgg.jpg)

### Loss

- Multi-task Learning

## Use Bert for Downstream Task

- Sentence Pair Classification
- Single Sentence Classification Task
- Question Answering Task
- Single Sentence Tagging Task


# BERT-WWW
+ https://www.jiqizhixin.com/articles/2019-06-21-01


# ERNIE - 百度

- https://zhuanlan.zhihu.com/p/76757794
- https://cloud.tencent.com/developer/article/1495731

## ERNIE - 清华/华为

- https://zhuanlan.zhihu.com/p/69941989

### 把英文字变成中文词

![](https://pics1.baidu.com/feed/09fa513d269759ee43efeba2c2b2c4126c22dfee.png?token=dd737a03414c5fb8c6c69efaa9665ebf&s=4296A62A8D604C0110410CF403008032)

### 使用TransE 编码知识图谱

![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/VBcD02jFhglJEicBrKD32A5pErPnYJ7H2BfuD9zp8MRQPV73UTSMwJ4uo99hJsbnumWJasOVvdgfd4YexHNKwAg/640?wx_fmt=png)

# MASS

## Tips

- **BERT通常只训练一个编码器用于自然语言理解，而GPT的语言模型通常是训练一个解码器**

## Framework

![img](https://mmbiz.qpic.cn/mmbiz_png/HkPvwCuFwNOxFonDn2BP0yxvicFyHBhltUXrlicMwOLIHG93RjMYYZxuesuiaQ7IlXS83TpNFx8AEVyJYO1Uu1YGw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

- 如上图所示，编码器端的第3-6个词被屏蔽掉，然后解码器端只预测这几个连续的词，而屏蔽掉其它词，图中“_”代表被屏蔽的词

- MASS有一个重要的超参数k（屏蔽的连续片段长度），通过调整k的大小，MASS能包含BERT中的屏蔽语言模型训练方法以及GPT中标准的语言模型预训练方法，**使MASS成为一个通用的预训练框架**

  - 当k=1时，根据MASS的设定，编码器端屏蔽一个单词，解码器端预测一个单词，如下图所示。解码器端没有任何输入信息，这时MASS和BERT中的屏蔽语言模型的预训练方法等价

    ![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g6fbmmgapuj30u005tt97.jpg)

  - 当k=m（m为序列长度）时，根据MASS的设定，编码器屏蔽所有的单词，解码器预测所有单词，如下图所示，由于编码器端所有词都被屏蔽掉，解码器的注意力机制相当于没有获取到信息，在这种情况下MASS等价于GPT中的标准语言模型

    ![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g6fbmyz68xj30u005r3z7.jpg)

  - MASS在不同K下的概率形式如下表所示，其中m为序列长度，u和v为屏蔽序列的开始和结束位置，表示从位置u到v的序列片段，表示该序列从位置u到v被屏蔽掉。可以看到，当**K=1或者m时，MASS的概率形式分别和BERT中的屏蔽语言模型以及GPT中的标准语言模型一致**

    ![img](https://tva1.sinaimg.cn/large/006y8mN6ly1g6fbnaoskzj30u007tjsb.jpg)



- 当k取大约句子长度一半时（50% m），下游任务能达到最优性能。屏蔽句子中一半的词可以很好地平衡编码器和解码器的预训练，过度偏向编码器（k=1，即BERT）或者过度偏向解码器（k=m，即LM/GPT）都不能在该任务中取得最优的效果，由此可以看出MASS在序列到序列的自然语言生成任务中的优势

## Experiment

+ 无监督机器翻译
+ 低资源

## Advantage of MASS

+ 解码器端其它词（在编码器端未被屏蔽掉的词）都被屏蔽掉，以鼓励解码器从编码器端提取信息来帮助连续片段的预测，这样能**促进编码器-注意力-解码器结构的联合训练**
+ 为了给解码器提供更有用的信息，编码器被强制去抽取未被屏蔽掉词的语义，以**提升编码器理解源序列文本的能力**
+ 让解码器预测连续的序列片段，以**提升解码器的语言建模能力**(???)

## Reference

- https://mp.weixin.qq.com/s/7yCnAHk6x0ICtEwBKxXpOw



# Uni-LM

![](../../../../Desktop/PPT/Uni-LM.jpg)

# XLNet

+ https://indexfziq.github.io/2019/06/21/XLNet/
+ https://blog.csdn.net/weixin_37947156/article/details/93035607

# Doc2Vec

+ https://blog.csdn.net/lenbow/article/details/52120230

+  http://www.cnblogs.com/iloveai/p/gensim_tutorial2.html

+ Doc2vec是Mikolov在word2vec基础上提出的另一个用于计算长文本向量的工具。它的工作原理与word2vec极为相似——只是将长文本作为一个特殊的token id引入训练语料中。在Gensim中，doc2vec也是继承于word2vec的一个子类。因此，无论是API的参数接口还是调用文本向量的方式，doc2vec与word2vec都极为相似
+ 主要的区别是在对输入数据的预处理上。Doc2vec接受一个由LabeledSentence对象组成的迭代器作为其构造函数的输入参数。其中，LabeledSentence是Gensim内建的一个类，它接受两个List作为其初始化的参数：word list和label list

```
from gensim.models.doc2vec import LabeledSentence
sentence = LabeledSentence(words=[u'some', u'words', u'here'], tags=[u'SENT_1'])
```

+ 类似地，可以构造一个迭代器对象，将原始的训练数据文本转化成LabeledSentence对象：

```
class LabeledLineSentence(object):
    def init(self, filename):
        self.filename = filename

    def iter(self):
        for uid, line in enumerate(open(filename)):
            yield LabeledSentence(words=line.split(), labels=['SENT_%s' % uid])
```

准备好训练数据，模型的训练便只是一行命令：

```
from gensim.models import Doc2Vec
model = Doc2Vec(dm=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=4)
```

+ 该代码将同时训练word和sentence label的语义向量。如果我们只想训练label向量，可以传入参数train_words=False以固定词向量参数。更多参数的含义可以参见这里的API文档。

+ 注意，在目前版本的doc2vec实现中，每一个Sentence vector都是常驻内存的。因此，模型训练所需的内存大小同训练语料的大小正相关。

# Tools

## gensim

- https://blog.csdn.net/sscssz/article/details/53333225 
- 首先，默认已经装好python+gensim了，并且已经会用word2vec了。

+ 其实，只需要在vectors.txt这个文件的最开头，加上两个数，第一个数指明一共有多少个向量，第二个数指明每个向量有多少维，就能直接用word2vec的load函数加载了
+ 假设你已经加上这两个数了，那么直接
+ Demo: Loads the newly created glove_model.txt into gensim API.
+ model=gensim.models.Word2Vec.load_word2vec_format(' vectors.txt',binary=False) #GloVe Model



# Reference

+ [transformer model TF 2.0 ](https://cloud.tencent.com/developer/news/417202)
+ [albert_zh](https://github.com/brightmart/albert_zh)

- https://www.zhihu.com/question/52756127
- [xlnet](https://indexfziq.github.io/2019/06/21/XLNet/)
- [self attention](https://www.cnblogs.com/robert-dlut/p/8638283.html)
- [embedding summary blog](https://www.cnblogs.com/robert-dlut/p/9824346.html)
- [ulm-fit](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)
- [open gpt](https://blog.floydhub.com/gpt2/)
- 从Word2Vec 到 Bert paper weekly
- Jay Alammar 博客， 对每个概念进行了可视化
- 中文常见的embedding
  - https://github.com/Embedding/Chinese-Word-Vectors
  - https://www.jiqizhixin.com/articles/2018-05-15-10


