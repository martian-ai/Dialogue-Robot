# Self-attention

## 计算方式

### 原始Attention
+ 一般的attention中

  $$ Attention(query, source) = \sum_{i=1}^{Len(x)} Similarity(query, key_i) * value_i$$

### 改进
+ 在 self-attention 中 认为 query=key=value=X， 公式可以表示为 $\operatorname{Softmax}\left(X X^{T}\right) X$
+ 对于 $X X^{T}$， 一个矩阵乘以其转置，等价于当前行与所有行分别做内积(向量的内积，表示一个向量在另一个向量上的投影，投影越大，则表示相关性越高)， 对于NLP场景都是词向量的情况，词向量是词在高维空间的数值映射，**词向量之间相关度高表示什么？是不是在一定程度上（不是完全）表示，在关注词A的时候，应当给予词B更多的关注**
+ 对于 softmax 是对结果进行归一化
+ 以下两图表示一个self attention 的过程

![20220115220056](https://s2.loli.net/2022/01/15/ERJ8nPr2A59vGag.png)

![20220115220716](https://s2.loli.net/2022/01/15/CjfXQKx6w12crsz.png)

+ 优点
  + 内部做attention，寻找内部联系
  + 可并行化处理，不依赖其他结果
  + 计算复杂度低，self-attention 的计算复杂度是 $n*n*d$ ,  而RNN 是 $n*d*d$ ,  这里n 是指序列长度， d指词向量的维度，一般d>n
  + self-Attention可以很好的捕获全局信息，无论词的位置在哪，词之间的距离都是1，因为计算词之间的关系时是不依赖于其他词的。在大量的文献中表明，self-Attention的长距离信息捕捉能力和RNN相当，远远超过CNN（CNN主要是捕捉局部信息，当然可以通过增加深度来增大感受野，但实验表明即使感受野能涵盖整个句子，也无法较好的捕捉长距离的信息）

## Q K V 的作用
+ 许多文章中所谓的Q K V矩阵、查询向量之类的字眼，其来源是X与矩阵的乘积，本质上都是X的线性变换。为什么不直接使用X而要对其进行线性变换？当然是为了提升模型的拟合能力，矩阵$W_Q, W_K, W_V$都是可以训练的。

![20220115220845](https://s2.loli.net/2022/01/15/HTdeFDYpOu15njq.png)

## $\sqrt{d_k}$ 的作用
+ 假设Q, K里的元素的均值为0，方差为1，那么$A^{T}=Q^{T} K$中元素的均值为0，方差为d. 当d变得很大时，  中的元素的方差也会变得很大，如果A中的元素方差很大，那么softmax(A)的分布会趋于陡峭(分布的方差大，分布集中在绝对值大的区域)。总结一下就是softmax(A)的分布会和d有关。因此A中每一个元素除以$\sqrt{d_k}$后，方差又变为1。这使得softmax(A)的分布“陡峭”程度与d解耦，从而使得训练过程中梯度值保持稳定。

## 位置编码
+ 对self-attention来说，它跟每一个input vector都做attention，所以没有考虑到input sequence的顺序。
+ 前文的计算每一个词向量都与其他词向量计算内积，得到的结果丢失了原来文本的顺序信息。对比来说，LSTM是对于文本顺序信息的解释是输出词向量的先后顺序，而上文的计算对sequence的顺序这一部分则完全没有提及，打乱词向量的顺序，得到的结果仍然是相同的。

## Multi-Head Attention

+ 原理
  + 把Q，K，V 通过参数矩阵映射一下，然后再做Attention，把整个过程重复h次，结果再拼接起来

+ 公式
  $$ MultiHead(Q,K,V) = Concat(head_1, head_2, ..., head_h) * W^O$$
  $$where\ head_i = Attention(QW_i^Q, KW_i^K, VW_i^V) $$


![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/07bc8576738c21b1e6d3d524d8419667.png)



首先我们先了解一下self-attention的作用，其实self attention大家并不陌生，比如我们有一句话，the animal didnot cross the street, because it was too tired. 这里面的it，指代的是the animal。我们在翻译it的时候会将更多的注意力放在the animal身上，self-attention起的作用跟这个类似，就是关注句子中的每个字，和其它字的关联关系。[参考实现](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)


![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/2ce284d02c97d378160ec08e22e5bc7b.png)

我们来看看这些词是怎么经过multi-head attention，得到转换的。

首先我们每个字的输入vector $\vec a^i$会经过变换得到三个vector，分别是query $\vec q$， key $\vec k$ 以及value $\vec v$, 这些向量是通过输入$\vec a^i$ 分别和query矩阵$W^Q$，key矩阵$W^K$，value矩阵$W^V$相乘得来的。query矩阵$W^Q$，key矩阵$W^K$，value矩阵$W^V$ 都是训练时学习而来的。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/a3d3757d6ec027b80c2aa5360e083585.png)

将 x1 和 WQ weight matrix 做矩阵乘法得到 q1, 即这个字对应的query向量. 类似地，我们最终得到这个字对应query向量，value向量，key向量。
- query向量：query顾名思义，是负责寻找这个字的于其他字的相关度（通过其它字的key）
- key向量：key向量就是用来于query向量作匹配，得到相关度评分的
- value向量：Value vectors 是实际上的字的表示, 一旦我们得到了字的相关度评分，这些表示是用来加权求和的


![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/b85a3a679a4b234dfdf6115c84537f4e.png)

得到每个字的$\vec q, \vec k, \vec v$ 之后，我们要得到每个字和句子中其他字的相关关系，我们只需要把这个字的query去和其他字的key作匹配，然后得到分数，最后在通过其它字的value的加权求和（权重就是哪个分数）得到这个字的最终输出。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/3c51bac530724498f844c9c4a95ccd1e.png)
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/7cb74cd7ff014659db5cb83259029c36.png)
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/51a0c1ebb599b781325c3a505c5bb83e.png)

我们来具体看看这个分数是怎么计算得到的。我们之前看到的都是单个字作self-attention，但是在GPU中，其实整个过程是并行的，一个序列$w_1, w_2...w_n$是同时得到每个$w_i$对应的Q，K，V的，这是通过矩阵乘法。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/cd2439aa09409252d47e666ae36b8b41.png)

然后每个字与其他字对应的score的算法采用的是Scaled Dot-product Attention
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/9687da5eb67b898e1e7d927b41878895.png)
具体就是以下公式
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/5337e49c01feec7399f139a9e1f38a15.png)
- 其中$softmax(x)_i = \frac{exp(x_i)}{\sum_{j}^{ }exp(x_j))}$。
- 其中，scale因子是输入的vector size $d_k$开根号。

总结来说：
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/6ee708453d8ed0549528ef8a083beb1b.png)

等等，那么什么是multi-head呢？
首先我们先了解一下什么是multi-head，其实很简单，就是我们刚才这个sub-encoder里面，我们的self-attention，只做了一次， 如果我们引入多个不同的$W^Q_i, W^K_i, W^V_i$, 然后重复刚才的步骤，我们就可以得到multi-head了。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/a7d29be7518042c772811d2418c259eb.png)
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/aa0a3058072341f5347ff4c039d29140.png)

在得到多个$Z_i$向量之后，我们把这些向量concat起来，然后再经过线性变换，得到最终的输出。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/953b28a709c52d965604bac2d03b396f.png)

那么我们为什么需要multi-head呢？这是因为，他可以提高模型的能力
- 这使得模型能够关注不同的位置，比如句子`经济。。。，教育。。。，这使得这座城市发展起来了`，句子中的`这`在不同的head中，可以着重关注不同的地方例如`经济`，`教育`。亦或者如下面的栗子。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/3876cf30bbf1ff6985df4324087ceebe.png)

- 就像是CNN采用不同的不同的kernel的效果，不同的kernel能过获取的信息不同，类似的，不同的head，能够扩展模型的不同表示空间(different representation subspaces)，因为我们有不同的QKV，这些都是随机初始化，然后通过训练得到最总结果，并且结果往往不同。关于different representation subspaces，举一个*不一定妥帖*的例子：当你浏览网页的时候，你可能在**颜色**方面更加关注深色的文字，而在**字体**方面会去注意大的、粗体的文字。这里的颜色和字体就是两个不同的表示子空间。同时关注颜色和字体，可以有效定位到网页中强调的内容。**使用多头注意力，也就是综合利用各方面的信息/特征**。
- 我觉得也可以把多头注意力看作是一种ensemble，模型内部的集成。


# Reference
+ 超详细图解Self-Attention - 伟大是熬出来的的文章 - 知乎
  + https://zhuanlan.zhihu.com/p/410776234  
