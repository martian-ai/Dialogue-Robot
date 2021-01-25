# Word2Vec

+ word2vector 是将词向量进行表征，其实现的方式主要有两种，分别是CBOW（continue bag of words) 和 Skip-Gram两种模型。这两种模型在word2vector出现之前，采用的是DNN来训练词与词之间的关系，采用的方法一般是三层网络，输入层，隐藏层，和输出层。之后，这种方法在词汇字典量巨大的时候，实现方式以及计算都不现实，于是采用了hierarchical softmax 或者negative sampling模型进行优化求解。
![word2vec_mind_map](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/4262fad7602bec52b4d68015198a2a97.png)

## 词向量基础
+ 用词向量来表示词并不是word2vec的首创，在很久之前就出现了。最早的词向量是很冗长的，它使用是词向量维度大小为整个词汇表的大小，对于每个具体的词汇表中的词，将对应的位置置为1。比如我们有下面的5个词组成的词汇表，词"Queen"的序号为2， 那么它的词向量就是(0,1,0,0,0)。同样的道理，词"Woman"的词向量就是(0,0,0,1,0)。这种词向量的编码方式我们一般叫做1-of-N representation或者one hot representation.
![onehot](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/751f0cf5d56a753d6348654375a7360e.png)
+ one hot representation 的优势在于简单，但是其也有致命的问题，就是在动辄上万的词汇库中，one hot表示的方法需要的向量维度很大，而且对于一个字来说只有他的index位置为1其余位置为0，表达效率不高。而且字与字之间是独立的，不存在字与字之间的关系。
+ 如何将字的维度降低到指定的维度大小，并且获取有意义的信息表示，这就是word2vec所做的事情。
+ 比如下图我们将词汇表里的词用"Royalty","Masculinity", "Femininity"和"Age"4个维度来表示，King这个词对应的词向量可能是(0.99,0.99,0.05,0.7)。当然在实际情况中，我们并不能对词向量的每个维度做一个很好的解释
![embd_visual](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/88e16f18535557d6ccd721e1dce26238.png)
+ 有了用Distributed Representation表示的较短的词向量，我们就可以较容易的分析词之间的关系了，比如我们将词的维度降维到2维，有一个有趣的研究表明，用下图的词向量表示我们的词时，我们可以发现：
$\vec{Queen} = \vec{King} - \vec{Man} + \vec{Woman}$
![IMAGE](https://images2015.cnblogs.com/blog/1042406/201707/1042406-20170713151608181-1336632086.png)

## CBOW
+ CBOW 模型的输入是一个字的上下文，指定窗口长度，根据上下文预测该字。
+ 比如下面这段话，我们上下文取值为4，特定词为`learning`，上下文对应的词共8个，上下各四个。这8个词作为我们的模型输入。CBOW使用的是词袋模型，这8个词都是平等的，我们不考虑关注的词之间的距离大小，只要是我们上下文之内的就行。

  ![IMAGE](https://images2015.cnblogs.com/blog/1042406/201707/1042406-20170713152436931-1817493891.png)

+ CBOW模型的训练输入是某一个特征词的上下文相关的词对应的词向量，而输出就是这特定的一个词的词向量。

### Naïve implement
+ 这样我们这个CBOW的例子里，我们的输入是8个词向量，输出是所有词的softmax概率（训练的目标是期望训练样本特定词对应的softmax概率最大），对应的CBOW神经网络模型**输入层有8个神经元（#TODO：check），输出层有词汇表大小V个神经元**。隐藏层的神经元个数我们可以自己指定。通过DNN的反向传播算法，我们可以求出DNN模型的参数，同时得到所有的词对应的词向量。这样当我们有新的需求，要求出某8个词对应的最可能的输出中心词时，我们可以通过一次DNN前向传播算法并通过softmax激活函数找到概率最大的词对应的神经元即可。
  <p align="center">
  <img width="580" height="440" src="http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/438a3747ce374fdce473da67085112dd.png">
</p>



### optimized methods
+ word2vec为什么不用现成的DNN模型，要继续优化出新方法呢？最主要的问题是DNN模型的这个处理过程非常耗时。我们的词汇表一般在百万级别以上，这意味着我们DNN的输出层需要进行softmax计算各个词的输出概率的的计算量很大。有没有简化一点点的方法呢？
- word2vec基础之霍夫曼树
  - word2vec也使用了CBOW与Skip-Gram来训练模型与得到词向量，但是并没有使用传统的DNN模型。最先优化使用的数据结构是用霍夫曼树来代替 **隐藏层** 和 **输出层的神经元**，霍夫曼树的 **叶子节点起到输出层神经元的作用**，**叶子节点的个数即为词汇表的大小**。 而内部节点则起到隐藏层神经元的作用。具体如何用霍夫曼树来进行CBOW和Skip-Gram的训练我们在下一节讲，这里我们先复习下霍夫曼树。
  - 霍夫曼树的建立其实并不难，过程如下：
    - 输入：权值为$(w1,w2,...wn)$的n个节点
    - 输出：对应的霍夫曼树
    - 1）将$(w1,w2,...wn)$看做是有n棵树的森林，每个树仅有一个节点。
    - 2）在森林中选择根节点权值最小的两棵树进行合并，得到一个新的树，这两颗树分布作为新树的左右子树。新树的根节点权重为左右子树的根节点权重之和。
    - 3） 将之前的根节点权值最小的两棵树从森林删除，并把新树加入森林。
    - 4）重复步骤2）和3）直到森林里只有一棵树为止。

    下面我们用一个具体的例子来说明霍夫曼树建立的过程，我们有$(a,b,c,d,e,f)$共6个节点，节点的权值分布是(20,4,8,6,16,3)。
　　 首先是最小的b和f合并，得到的新树根节点权重是7.此时森林里5棵树，根节点权重分别是20,8,6,16,7。此时根节点权重最小的6,7合并，得到新子树，依次类推，最终得到下面的霍夫曼树。

    ![huffman](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/740863966b769431c9553e07705c7b54.png)

    那么霍夫曼树有什么好处呢？一般得到霍夫曼树后我们会对叶子节点进行霍夫曼编码，由于权重高的叶子节点越靠近根节点，而权重低的叶子节点会远离根节点，这样我们的高权重节点编码值较短，而低权重值编码值较长。这保证的树的带权路径最短，也符合我们的信息论，即我们希望越常用的词拥有更短的编码。如何编码呢？一般对于一个霍夫曼树的节点（根节点除外），可以约定左子树编码为0，右子树编码为1.如上图，则可以得到c的编码是00。

    **在word2vec中，约定编码方式和上面的例子相反，即约定左子树编码为1，右子树编码为0，同时约定左子树的权重不小于右子树的权重。**




- word2vector对这个模型进行了改进。
    <p align="center">
    <img width="500" height="340" src="http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/895a8e965e06ce2e5e22e5352a2cc430.png">
    </p>


  - 首先对于输入层到隐藏层的映射，没有采用传统的神经网络的线性加权加激活函数，而是直接采用了**简单的加和平均**, 如上图。
    - 比如输入的是三个4维词向量：$(1,2,3,4),(9,6,11,8),(5,10,7,12)$,那么我们word2vec映射后的词向量就是$(5,6,7,8)$。由于这里是从多个词向量变成了一个词向量。
  - 第二个改进就是**隐藏层(Projection Layer)到输出层(Output Layer)的softmax的计算量**进行了改进，为了避免计算所有词的softmax的概率
  - word2vector采用了两种方式进行改进，分别为hierarchical softmax和negative sampling。降低了计算复杂度。


#### Hierarchical Softmax

<p align="center">
  <img width="440" height="270" src="https://miro.medium.com/max/600/0*hUAFJJOBG3D0PgKl.">
</p>

- 在计算之前，先根据词频统计出一颗Huffman树
- 我们通过将softmax的值的计算转化为huffman树的树形结构计算，如下图所示，我们可以沿着霍夫曼树从根节点一直走到我们的叶子节点的词$w_2$

<p align="center">
  <img width="440" height="300" src="https://images2017.cnblogs.com/blog/1042406/201707/1042406-20170727105752968-819608237.png">
</p>

  - 跟神经网络类似，根结点的词向量为contex的词投影（加和平均后）的词向量
  - 所有的叶子节点个数为Vocabulary size
  - 中间节点对应神经网络的中间参数，类似之前神经网络隐藏层的神经元
  - 通过投影层映射到输出层的softmax结果，是根据huffman树一步步完成的
  - 如何通过huffman树一步步完成softmax结果，采用的是logistic regression。规定沿着左子树走，代表负类（huffman code = 1），沿着右子树走，代表正类（huffman code=0）。
    - $P(+)=\sigma(x^Tw_θ)=\frac{1}{1+e^{−x^Tw_θ}}$ hufman编码为0
    - $P(-)=1-\sigma(x^Tw_θ)=1-\frac{1}{1+e^{−x^Tw_θ}}$huffman编码为1
    - 其中$x_w$ 是当前内部节点的输入词向量，$\theta$是利用训练样本求出的lr的模型参数
  - $p(w_2) = p_1(-)p_2(-)p_3(+) = (1-\sigma(x_{w_2}\theta_1)) (1-\sigma(x_{w_2}\theta_2)) \sigma(x_{w_2}\theta_3) $
- Huffman树的好处
  - 计算量从V降低到了$logV$，并且使用huffman树，越高频的词出现的深度越浅，计算所需的时间越短，例如`的`作为target词，其词频高，树的深度假设为2，那么计算其的softmax值就只有2项
  - 被划为左子树还是右子树即$P(-) or P(+)$, 主要取决与$\theta$ 和$x_{w_i}$
----

- 如何训练？
   - 目标是使得所有合适的节点的词向量$x_{w_i}$以及内部的节点$\theta$ 使得训练样本达到最大似然
   - 分别对$x_{w_i}$ 以及$\theta$求导

   例子：以上面$w_2$作为例子
$\prod_{i=1}^{3}P_i=(1−\frac{1}{1+e^{−x^Tw_{θ_1}}})(1−\frac{1}{1+e^{−x^Tw_{θ_2}}})\frac{1}{e^{−{x^Tw_{θ_3}}}}$

    对于所有的训练样本，我们期望 **最大化所有样本的似然函数乘积**。

    我们定义输入的词为$w$，其从输入层向量平均后输出为$x_w$，从根结点到$x_w$所在的叶子节点，包含的节点数为$l_w$个，而该节点对应的模型参数为$\theta_i^w$, 其中i=1,2,....$l_w-1$，没有$l_w$，因为它是模型参数仅仅针对与huffman树的内部节点
  - 定义$w$经过的霍夫曼树某一个节点j的逻辑回归概率为$P(d_j^w|x_w,\theta_{j-1}^w)$，其表达式为：

    ![equation1](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/fc080b3a7dbe57b3fd5ce9958af752f1.png)
    <!-- \begin{equation}
    P(d_j^w|x_w,\theta_{j-1}^w)=\left{
    \begin{aligned}
    \sigma(x^T_w\theta^w_{j-1}) &  & d_j^w = 0 \
    1-\sigma(x^T_w\theta^w_{j-1}) &  & d_j^w = 1
    \end{aligned}
    \right.
    \end{equation} -->
  - 对于某个目标词$w$， 其最大似然为：
  $\prod_{j=2}^{l_w}P(d_j^w|x_w,\theta_{j-1}^w)=\prod_{j=2}^{l_w}[(\frac{1}{1+e^{−x_w^Tw_{θ_{j-1}}}})]^{1-d_j^w}[1-\frac{1}{e^{−{x^Tw_{θ_{j-1}}}}}]^{d_j^w}$

  - 采用对数似然

    $ L=log\prod_{j=2}^{l_w}P(d^w_j|x_w,θ^w_{j−1})=\sum_{j=2}^{l_w}[(1−d^w_j)log[\sigma(x^T_wθ_{w_{j-1}})]+d_{w_j}log[1−\sigma(x^T_{w}θ_{w_{j−1}})]]$

    要得到模型中$w$词向量和内部节点的模型参数$θ$, 我们使用梯度上升法即可。首先我们求模型参数$θ^w_{j−1}$的梯度：
    ![equation2](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/f0839dd372da2214c28ed4015c80be1a.png)
    同样的方法，可以求出$x_w$的梯度表达式如下：![equation3](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/dc2527a7e0cdecef58a645ad2163448e.png)
    有了梯度表达式，我们就可以用梯度上升法进行迭代来一步步的求解我们需要的所有的$θ^w_{j−1}$和$x_w$。

- 基于Hierarchical Softmax的CBOW模型
  - 首先我们先定义词向量的维度大小M，以及CBOW的上下文2c，这样我们对于训练样本中的每一个词，其前面的c个词和后c个词都是CBOW模型的输入，而该词作为样本的输出，期望其softmax最大。
  - 在此之前，我们需要先将词汇表构建成一颗huffman树
  - 从输入层到投影层，需要对2c个词的词向量进行加和平均
  $\vec{x_w} = \frac{1}{2c}\sum^{2c}_{i=1}\vec{x_i}$
  - 通过梯度上升更新$\theta^w_{j-1}$和$x_w$， 注意这边的$x_w$是多个向量加和，所以需要对每个向量进行各自的更新，我们做梯度更新完毕后会用梯度项直接更新原始的各个$x_i(i=1,2,,,,2c)$

      $\theta_{j-1}^w = \theta_{j-1}^w+\eta (1−d^w_j−\sigma(x^T_w\theta^w_{j−1}))x_w$  $\forall j=2 \ to \ l_w $

      对于
      $x_i = x_i + \eta\sum^{l_w}_{j=2}(1-d^w_j-\sigma(x^T_w\theta^w_{j-1}))\theta^w_{j-1}\ \ \forall i = \ 1 \ to \ 2c$

    Note: $\eta$ 是learning rate

  ```
  Input：基于CBOW的语料训练样本，词向量的维度大小M，CBOW的上下文大小2c 步长η
  Output：霍夫曼树的内部节点模型参数θ，**所有的词向量w**
  1. create huffman tree based on the vocabulary
  2. init all θ, and all embedding w
  3. updage all theta and emebedding w based on the gradient ascend, for all trainning sample $(context(w), w)$ do:
  ```

  - a) $e = 0 , x_w = \sum_{i=1}^{2c}x_i$

  - b) `for j=2..l_w:`

    > $ g = 1−d^w_j−\sigma(x^T_w\theta^w_{j−1})$
    > $ e = e+ g*\theta^w_{j-1}$
    > $ \theta^w_{j-1} = \theta_{j-1}^w+\eta_\theta * g * x_w$

  - c) `for all x_i in context(w), update x_i :`
    > $x_i = x_i + e$

  - d) 如果梯度收敛，则结束梯度迭代，否则回到步骤 **3)** 继续迭代。
 ------

#### Negative Sampling
采用h-softmax在生僻字的情况下，仍然可能出现树的深度过深，导致softmax计算量过大的问题。如何解决这个问题，negative sampling在和h-softmax的类似，采用的是将多分类问题转化为多个2分类，至于多少个2分类，这个和negative sampling的样本个数有关。
- negative sampling放弃了Huffman树的思想，采用了负采样。比如我们有一个训练样本，中心词是$w$, 它周围上下文共有2c个词，记为$context(w)$。在CBOW中，由于这个中心词$w$的确是$context(w)$相关的存在，因此它是一个真实的正例。通过Negative Sampling采样，我们得到neg个和$w$不为中心的词$wi$, $i=1,2,..neg$，这样$context(w)$和$w_i$就组成了neg个并不真实存在的负例。利用这一个正例和neg个负例，我们进行二元逻辑回归，得到负采样对应 **每个词$w_i$对应的模型参数$θ_i$** ，和 **每个词的词向量**。
- 如何通过一个正例和neg个负例进行logistic regression？
  - 正例满足：$P(context(w_0), w_i) = \sigma(x_0^T\theta^{w_i}) \  y_i=1, i=0$
  - 负例满足：$P(context(w_0), w_i) = 1- \sigma(x_0^T\theta^{w_i}) \ y_i=0, i=1..neg$
  - 期望最大化：
    $\Pi^{neg}_{i=0}P(context(w_0), w_i) =\Pi^{neg}_{i=0}  [\sigma(x_0^T\theta^{w_i})]^{y_i}[1- \sigma(x_0^T\theta^{w_i})]^{1-y_i} $

   对数似然为：
    $log\Pi^{neg}_{i=0}P(context(w_0), w_i) =\sum^{neg}_{i=0}  y_i*log[\sigma(x_0^T\theta^{w_i})]+(1-y_i) * log[1- \sigma(x_0^T\theta^{w_i})] $

 - 和Hierarchical Softmax类似，我们采用随机梯度上升法，仅仅每次只用一个样本更新梯度，来进行迭代更新得到我们需要的$x_{w_i},θ^{w_i},i=0,1,..neg$, 这里我们需要求出$x_{w_0},θ^{w_i},i=0,1,..neg$的梯度。
  - $θ^{w_i}$:
  ![theta_grad](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/001b2a6414f0ccb6a0870d94c039da4b.png)


  - $x_{w_0}$:

  ![x_grad](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/19a718aa0a553766e410df7860c55bd1.png)

- 如何采样负例
  负采样的原则采用的是根据 **词频** 进行采样，词频越高被采样出来的概率越高，词频越低，被采样出来的概率越低，符合文本的规律。word2vec采用的方式是将一段长度为1的线段，分为 $V(ocabulary size)$ 份，每个词对应的长度为:

    $len(w)=\frac{Count(w)}{\sum_{u\in V}Count(u)}$

  在word2vec中，分子和分母都取了 $\frac{3}{4}$ 次幂如下：

   $len(w)=\frac{Count(w)^{3/4}}{[\sum_{u\in V}Count(u)]^{3/4}}$

   在采样前，我们将这段长度为1的线段划分成M等份，这里M>>V，这样可以保证每个词对应的线段都会划分成对应的小块。而M份中的每一份都会落在某一个词对应的线段上。在采样的时候，我们只需要从M个位置中采样出neg个位置就行，此时采样到的每一个位置对应到的线段所属的词就是我们的负例词。

   ![IMAGE](https://images2017.cnblogs.com/blog/1042406/201707/1042406-20170728152731711-1136354166.png)

   在word2vec中，M取值默认为 $10^8$

----

  ```
  Input：基于CBOW的语料训练样本，词向量的维度大小M，CBOW的上下文大小2c 步长η，负采样的个数neg
  Output：**词汇表中每个词对应的参数θ** ，**所有的词向量$w$**
  1. init all θ, and all embedding w
  2. sample neg negtive words w_i, i =1,2,..neg
  3. updage all theta and emebedding w based on the gradient ascend, for  all trainning sample (context(w), w) do:
  ```

  - a) $e = 0 , x_w = \frac{1}{2c}\sum_{i=0}^{2c}x_i$

  - b) `for i=0..neg:`

      > $ g =\eta*(y_i−\sigma(x^T_{w}\theta^{w_i}))$
      > $e = e+ g*\theta^{w_i}$
      > $\theta^{w_i} = = \theta^{w_i}+g*x_{w}$

  - c) `for all x_k in context(w) (2c in total), update x_k :`
      > $x_k = x_k + e$

  - d) 如果梯度收敛，则结束梯度迭代，否则回到步骤 **3)** 继续迭代。

## Skip-Gram
Skip gram 跟CBOW的思路相反，根据输入的特定词，确定对应的上下文词词向量作为输出。
<p align="center">
  <img width="600" height="180" src="https://images2015.cnblogs.com/blog/1042406/201707/1042406-20170713152436931-1817493891.png">
</p>

这个例子中，`learning`作为输入，而上下文8个词是我们的输出。

### Naïve implement
我们输入是特定词，输出是softmax概率前8的8个词，对应的SkipGram神经网络模型，**输入层有1个神经元，输出层有词汇表个神经元。**[#TODO check??]，隐藏层个数由我们自己指定。通过DNN的反向传播算法，我们可以求出DNN模型的参数，同时得到所有的词对应的词向量。

<p align="center">
  <img width="600" height="480" src="http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/0a8d3eb8f484f84a9d2c67e75b90e34a.png">
</p>

### optimized methods

#### Hierarchical Softmax

  - 首先我们先定义词向量的维度大小M，此时我们输入只有一个词，我们希望得到的输出$context(w)$ 2c个词概率最大。
  - 在此之前，我们需要先将词汇表构建成一颗huffman树
  - 从输入层到投影层，就直接是输入的词向量
  - 通过梯度上升更新$\theta^w_{j-1}$和$x_w$， 注意这边的$x_w$周围有2c个词向量，此时我们希望$P(x_i|x_w), \  i=1,2,..2c$最大，此时我们注意到上下文是相互的，我们可以认为$P(x_w|x_i), i=1,2,..2c$也是最大的，对于那么是使用$P(x_i|x_w)$
好还是$P(x_w|x_i)$好呢，word2vec使用了后者，这样做的好处就是在一个迭代窗口内，我们不是只更新$x_w$一个词，而是 $xi, \ i=1,2...2c$ 共2c个词。**不同于CBOW对于输入进行更新，SkipGram对于输出进行了更新**。

  - $x_i(i=1,2,,,,2c)$
      $\theta_{j-1}^w = \theta_{j-1}^w+\eta (1−d^w_j−\sigma(x^T_w\theta^w_{j−1}))x_w \forall j=2 \ to \ l_w $
      $x_i = x_i + \eta\sum^{l_w}_{j=2}(1-d^w_j-\sigma(x^T_w\theta^w_{j-1}))\theta^w_{j-1} \forall i = \ 1 \ to 2c$
    Note: $\eta$ 是learning rate


  ```
    Input：基于Skip-Gram的语料训练样本，词向量的维度大小M，Skip-Gram的上下文大小2c 步长η
    Output：霍夫曼树的内部节点模型参数θ，**所有的词向量w**
    1. create huffman tree based on the vocabulary
    2. init all θ, and all embedding w
    3. updage all theta and emebedding w based on the gradient ascend, for all trainning sample (w, context(w)) do:
  ```

  - a) `for i = 1..2c`
      >i) $e = 0 $

     - ii) `for j=2..l_w:`
         > $ g = 1−d^w_j−\sigma(x^T_w\theta^w_{j−1})$
         > $e = e+ g*\theta^w_{j-1}$
         > $\theta^w_{j-1} = = \theta_{j-1}^w+\eta_\theta*g*x_w$

     - iii) $x_i = x_i + e$

  - b) 如果梯度收敛，则结束梯度迭代，算法结束，否则回到步骤 **a)** 继续迭代。

#### Negative sampling
有了上一节CBOW的基础和上一篇基于Hierarchical Softmax的Skip-Gram模型基础，我们也可以总结出基于Negative Sampling的Skip-Gram模型算法流程了。梯度迭代过程使用了随机梯度上升法：

  ```
  Input：基于Skip-Gram的语料训练样本，词向量的维度大小M，Skip-Gram的上下文大小2c 步长η, 负采样的个数为neg
  Output：词汇表中每个词对应的参数θ，**所有的词向量w**
  1. init all θ, and all embedding w
  2. for all training data (context(w_0), w_0), sample neg negative words $w_i$, $i =1,2..neg$
  3. updage all theta and emebedding w based on the gradient ascend, for all trainning sample (w, context(w)) do:
  ```

  - a)  `for i = 1..2c`
      > i) $e = 0 $

     - ii) `for j=0..neg:`
       > $ g =\eta*(y_i−\sigma(x^T_{w}\theta^{w_j}))$
       > $e = e+ g*\theta^{w_j}$
       > $\theta^{w_j} = = \theta^{w_j}+g*x_{w_i}$

     - iii) $x_i = x_i + e$
  - b) 如果梯度收敛，则结束梯度迭代，算法结束，否则回到步骤 **a)** 继续迭代。

#### source code
- [Hierarchical Softmax](https://github.com/tmikolov/word2vec/blob/master/word2vec.c)

  在源代码中，基于Hierarchical Softmax的CBOW模型算法在435-463行，基于Hierarchical Softmax的Skip-Gram的模型算法在495-519行。大家可以对着源代码再深入研究下算法。在源代码中，neule对应我们上面的$e$
  , syn0对应我们的$x_w$, syn1对应我们的$θ^i_{j−1}$, layer1_size对应词向量的维度，window对应我们的$c$。

  另外，vocab[word].code[d]指的是，当前单词word的，第d个编码，编码不含Root结点。vocab[word].point[d]指的是，当前单词word，第d个编码下，前置的结点。

- [negative sampling code](https://github.com/tmikolov/word2vec/blob/master/word2vec.c)

  在源代码中，基于Negative Sampling的CBOW模型算法在464-494行，基于Negative Sampling的Skip-Gram的模型算法在520-542行。大家可以对着源代码再深入研究下算法。
  在源代码中，neule对应我们上面的$e$
  , syn0对应我们的$x_w$, syn1neg对应我们的$θ^{w_i}$, layer1_size对应词向量的维度，window对应我们的$c$。negative对应我们的neg, table_size对应我们负采样中的划分数$M
  $。另外，vocab[word].code[d]指的是，当前单词word的，第d个编码，编码不含Root结点。vocab[word].point[d]指的是，当前单词word，第d个编码下，前置的结点。这些和基于Hierarchical Softmax的是一样的。

## FastText词向量与word2vec对比
  - FastText= word2vec中 cbow + h-softmax的灵活使用
  - 灵活体现在两个方面：
    1. 模型的输出层：word2vec的输出层，对应的是每一个term，计算某term的概率最大；而fasttext的输出层对应的是 分类的label。不过不管输出层对应的是什么内容，起对应的vector都不会被保留和使用；
    2. 模型的输入层：word2vec的输出层，是 context window 内的term；而fasttext 对应的整个sentence的内容，包括term，也包括 n-gram的内容；
  - 两者本质的不同，体现在 h-softmax的使用。
    - Wordvec的目的是得到词向量，该词向量 最终是在输入层得到，输出层对应的 h-softmax 也会生成一系列的向量，但最终都被抛弃，不会使用。
    - fasttext则充分利用了h-softmax的分类功能，遍历分类树的所有叶节点，找到概率最大的label（一个或者N个）
  - fastText 可以用来做句子分类以及词向量，word2vec只能构造词向量
  - word2vec 把单词（字）作为最小的单位（和GloVe一样），但是FastText是word2vec的拓展，fastText把字作为是ngram的集合，所以一个单词的词向量是其所有的ngram的向量的加和，这样子做一定程度减少了OOV的问题。例如：

  > the word vector “apple” is a sum of the vectors of the n-grams:
  > “<ap”, “app”, ”appl”, ”apple”, ”apple>”, “ppl”, “pple”, ”pple>”, “ple”, ”ple>”, ”le>”
  > (assuming hyperparameters for smallest ngram[minn] is 3 and largest ngram[maxn] is 6).

  - 采用ngram对中文字有意义吗？因为中文并不是由subword组成的。

    这是有意义的，因为fastText的ngram组成是根据utf-8 encoding构成的，根具体的字的形式无关。
  > Yes, the minn and maxn parameters can be used for Chinese text classification, as long as your data is encoded in utf-8. Indeed, fastText assumes that the data uses utf-8 to split words into character ngrams. For Chinese text classification, I would recommend to use smaller values for minn and maxn, such as 2 and 3.

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
