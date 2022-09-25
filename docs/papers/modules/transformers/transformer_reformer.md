
# Reformer
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Reformer](#reformer)
	- [Locality Sensitive Hashing Attention](#locality-sensitive-hashing-attention)
		- [locality Sensitive Hashing](#locality-sensitive-hashing)
		- [LSH attention](#lsh-attention)
	- [Reverible Transformer](#reverible-transformer)
		- [RevNet](#revnet)
		- [RevTransformer](#revtransformer)
	- [Chunking](#chunking)
	- [Ref](#ref)

<!-- /TOC -->
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/1d3a4411abc621d5223c3c6fcea121c6.png)

[REFORMER : THE EFFICIENT TRANSFORMER](https://openreview.net/pdf?id=rkgNKkHtvB)是google 2020 的一篇重量级的文章，文章中主要针对transfermer（[需要先复习](https://zhuanlan.zhihu.com/p/102591791)）做了如下的改进，是的模型的复杂度从$O(L^2)$变为了$O(LlogL)$。文章思路还是很清晰的，但是不好理解，需要多读几遍。

主要解决的痛点是
- transformer模型的空间复杂度高，所以sequence length必须不能很长，batch size也不能很大。在attention机制中，由于 attention 机制在计算时需要计算两两的attention score，这需要的时间和空间复杂度是 $O(L^2)$，所以即使序列的长度很短，也会使用大量的资源。
- 以BERT为例，transfer的encoder的层数越多，需要储存的参数量越大，因为我们需要储存层与层之间的连接参数（activations），用于反向传播时的计算。
- 因为我们知道，我们encoder中，分为self-attention以及feed forward neural network（FFN），其中FFN是两层的神经网络，其中的中间层的hidden size ($d_{ff}$)比self attention的hidden size ($d_{model})$更大，所以
占据了更多的内存空间。
例如：bert—base的中文模型的($d_{model}$) `"hidden_size": 768`, 而
 FFN ($d_{ff}$)的 `"intermediate_size": 3072`


采用的方式
- **Locality Sensitive Hashing Attention**
  使用了LSH的方式，将attention score 相近（即Key相似的）的分到同一个bucket中。因为我们经过softmax之后，一个 query 和其他的所有的token的计算 attention score主要是取决于高相似度的几个tokens，所以采用这种方式将近似算得最终的attention score。

- **Reversible layers**
  [RevNet](<https://arxiv.org/abs/1707.04585>) 的提出是为了解决ResNet层数加深后，我们需要储存每一层的activations（即每一层的输入），导致memory 消耗过大的问题。同样我们在transformer中也遇到了同种问题，我们采用这种方式的话，不需要我们记录中间层的activations，而只需要我们储存最后一层的输出，从而通过模型的特定结构，反推出中间层的结果。

- **Chunking FFN layers**
  将FFN分段处理，因为FFN中的输入之间互相独立，进行分段的处理可以降低空间消耗。

取得的成果

该改进版的reformer能够是的sequence length 长度达到64k，相比于之前的常见的512 长了不少。得到的效果和之前的transformer 差不多，但是速度却快了不少。

论文的主要难点在于采用的解决方案可能读者比较不熟悉，通过论文的阅读，需要的 prerequisites比较不熟悉，主要分为两个部分
-  Locality Sensitive Hash
-  RevNets

本文将会从这两个方向详细介绍。

## Locality Sensitive Hashing Attention
我们之前的Transformer 采用的是 **Dot-product attention**，具体如下：
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/f5fedf7a540ccec1d90514bb6a6fdd42.png)
其中Q：query向量，K：key向量，V：value向量，$d_k$：模型的输入的hidden size。具体参考transformer的介绍。

为了节省参数量，我们令Q=K，得到一个shared-QK transformer，文章通过实验证明，这样子的模型参数优化并没有降低模型的效果。

注意到我们需要考虑的是 $softmax(\frac {QK^T} {\sqrt d_k} )$， 对于每一个token对应的$q_i$（i.e. Q的其中一行）我们并不需要所有的token对应的keys（K中所有的行），这是因为经过softmax之后，值小的会转化为0，而值大的才起作用，这意味着，$q_iK^T$ 中，与$q_i$ 相近的 $k_i$ 才需要被考虑，而不相近的则可以被忽略。也就是说，我们只需要考虑最相近的top 32或者64 个keys。

我们可以看以下的例子：
```Python
import tensorflow as tf
res = tf.nn.softmax([10.0, 10.9, 10.8, 7, 6, 5, 4, 3, 2, 1])
with tf.Session() as sess:
    print(sess.run(res))
...
[1.7349562e-01 4.2673022e-01 3.8612169e-01 8.6378390e-03 3.1776831e-03
 1.1690042e-03 4.3005263e-04 1.5820753e-04 5.8201298e-05 2.1411061e-05]
```

我们可以看到，对于这10个scores， 我们只需要考虑前三个scores（最相关），因为它们经过softmax出来都是$10^{-1}$ 量级的，而其他的值出来都是 $10^{-3}$量级。

那么我们要怎么为每一个query找到最近邻呢？我们使用的就是Locality Sensitive Hashing（LSH）。


### locality Sensitive Hashing

首先我们先回忆一下什么是Hash 函数，一个Hash 函数就是使用一个hash 表，将特定的值映射到不同的桶（bucket）中，使得通常情况下我们可以在$O(1)$的时间复杂度下获得这个值。我们说通常情况下，意味着也有不同的情况，那么是什么情况呢。在采用linked list实现的哈希表中，如果我们存入的值得到的hash值，一样，那么我们的查找时间复杂度$O(k)$，k代表了冲突的个数，也就是说，如果我们有三个值，他们的hash值相同，那么我们对于其中一个值的查找的时间复杂度最大为$O(3)$。

通常情况下，我们希望我们的存入hash table的每个值获得不同的hash value，但是在我们现在的最近邻问题中，我们刚好可以利用这个性质，希望相近的keys的hashvalue相同。但是这还不够，因为这样子很容易满足，只要让所有的key的hash value一样，即进入相同的bucket，那么就解决了。所以还需要保证不相近的keys 拥有不同的hash value。

我们就引出了我们的LSH，这个局部敏感哈希就是设计使得：
- 两个相近的输入比相远的输入更容易获得相同的hash 值。
>  Locality-sensitive hash functions are specifically designed so that hash value collisions are more likely for two input values that are close together than for inputs that are far apart.

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/12da96221edd3e88c9f607b4706f5929.png)

我们举一个生活中的例子，使得我们更容易理解这背后的思想：

假设我们在一个大房间，里面有伦敦各个地方的人，我们需要让住的近的人站一块。

我们不会让一个人去询问每个人住在哪里，然后在划分。而是首先先在区域中，写上区域的邮政编号（伦敦的每户房子都有自己单独的邮编，前三位代表了自己所在的地区）。然后让里面的人自己走到各自的邮编区域。

首先这个方法的好处：
- 并行性：可以并行处理，每个人可以自己走都特定的区域

但是也有可以带来的坏处：
- 近似性：如果两个人住得很近，但是刚好在不同的区域，即临界的地区，那么这两个可能会被分的很远。

文章使用的是一种叫做random projections的方式，将key映射到不同的bucket中。具体的操作如下：
假设我们的key的vector size 是$d_k$，我们的想要获得$b$ 个hash bukcet桶。我们定义映射到特定hash bucket的函数 $h$， 以及随机的矩阵R，size：$[d_k, \frac b 2]$, argmax 函数指的是获得最大值对应的index:

\begin{equation}
h(x) = argmax[xR; -xR]
\end{equation}

如下图，b=4，对于Random Rotation 0，h(x) = 0, h(y) = 3。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/60282a3557f91ec115874044991b8e51.png)
### LSH attention

对于transformer的decoder的attention来说，每个token的query $q_i$， 只能attend到它自己本身以及之前的key $k_j$。所以它的输出如下 (为了方便除了第一步之后，都省略了$\sqrt d_k$

\begin{equation}
\begin{split}
o_i &= \sum _{0 \le j \le i} softmax(\frac {q_i · k_j} {\sqrt d_k}) v_j \\
&= \sum_{j \in P_i} \frac {e^{q_i · k_j}} {\sum_{l \in P_i} e ^{q_i·k_l}} v_j \\
&= \sum_{j \in P_i}  exp(q_i · k_j - z(i, P_i))v_j  \ \ \ \ \ \ where \ P_i = \{j: i \ge j\}
\end{split}
\end{equation}

注：其中 $z(i, P_i)$ 是归一化项， $P_i$指的是position i可以attend to 的所有位置。

为了实现方便，我们一版是采用look-ahead mask的方式进行，
即对于不能attend to的位置，其的score=0，我们采用的是在$q_i·k_j$ 的值之间减去正无穷，然后经过softmax函数之后其 score = 0，这样就不需要对于每个位置i 都有单独的P_i。我们令$\tilde P_i = \{ 0,1, ..., l\} \supseteq P_i$

\begin{equation}
\begin{split}
o_i &= \sum_{j \in \tilde P_i}  exp(q_i · k_j -m(j, P_i) - z(i, P_i))v_j  \ \ \ \ \ \ where \ m(j, P_i) =\left\{
\begin{aligned}
\infty &  & ifj \notin P_i \\
0 &  & j \in P_i
\end{aligned}
\right.
\end{split}
\end{equation}

当我们使用LSH的时候，我们将不会考虑全部的i之前的位置，我们将只考虑与position i在同个hash bucket的keys。即$P_i = \{j : h(q_i) =  h(k_j)\}$。

我们将根据下图来一步一步推导我们LSH attention的具体实现。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/b4f9df9daa418d291871857448709d64.png)

右半边图中
- （a）：我们可以看到在q和k不同的情况下，即普通的attention机制中，黑点代表的是需要softmax中占主导的位置，注意这边的attention使用的是encoder的attention， 否则$q_3$ 是无法attend to $k_6$ 的。 我们可以清楚看到，对于需要attend 的位置是稀疏的，我们可以利用这个降低我们的时间空间复杂度。
- （b）：我们不改变q和k，但是我们这次使用了LSH就只attend to 相同bucket的位置的keys。我们按照bucket进行排序，然后对于同一个bucket又按照原本的位置进行排序得到图b。我们可以看到，对于同一个bucket，可以出现一个bucket中有多个query但是很少keys的情况，例如图中蓝色的bucket。
- （c）：为了减小bucket中q和k不均衡的问题，文章提出了保证通过令 $k_j = \frac {q_j} {|q_j|}$ 从而使得 $h(k_j) = h(q_j)$, 即使用了share-QK attention。然后在按照bucket 排序，每个bucket中，仍按照原本的position 位置大小排序。得到图c。这时候就能保证对角线都是attend to的而且q和k在bucket中的个数一样（因为Q=K）。我们注意到对角线的点为空心，这是因为我们虽然在正常实现上，我们的q会attend to本身位置的value，但是在share-QK的实现下，如果attend to本身，会导致其值特别大，其他的值特别小，经过softmax之后，其他都是0，就自己本身是1。所以为了避免这种情况，我们q不会去attend 自身位置的值，除非只有自己本身可以attend to（例如图3/4的 $q_1$）。
- （d）：即使Q=K了，但是还是会出现一个问题就是，有的bucket中个数多，有的bucket中个数少，出一个极端的情况，对于2个bucket，我们其中一个bucket占据了所有的keys，另一个bucket为空，那么我们的LSH attention就没有起到作用。于是在c的基础上，增减了chunk的操作。具体的操作就是我们在对我们的输入进行排序之后（先bucket排序，同个bucket内按照position排序）得到新的序列顺序$s_i$ 即 $i →  s_i$。例如图d中的序列由$[q_1, q_2, q_3, q_3, q_5, q_6]$ 到了$[q_1, q_2, q_4, q_3,q_6,q_5]$。我们将设每个bucket的个数为 $m = \frac {2l}{n_{bucket}}$, (l 为输入query的个数) 然后对于bucket中的每个query，都可以attend to**自己以及前一个**bucket 中**相同**hash 值的key。
即其后选集 $\tilde P_i$为，（注意候选集之后仍需要保证hash值相同）：
\begin{equation}
\tilde P_i = \lfloor{\frac{s_i} {m}}\rfloor -1 \le \lfloor{\frac{s_j} {m}}\rfloor \le \lfloor{\frac{s_i} {m}}\rfloor
\end{equation}

总结来说，整个过程就如左半边图：
- 首先我们令输入序列的queries = keys
- 然后我们对其做LSH bucketing，得到每个query和key都在各自的bucket中（不同颜色表示）
- 我们跟根据bucket对query进行排序，同个bucket中，按照query原本的position进行排序。
- 在之后我们对于每个排序后的新序列，进行chunk 拆分
- 最后我们对于每个query只管制自己以及自己之前的chunk，对于这些候选集中相同bucket的key进行attend。


我们在分析最近邻的例子中，我们提到了LSH 有近似性，即我们不能保证相似的输入能在同一个bucket中。为了减轻这个问题，文章中采用了**multi-round LSH attention**。即我们query通过多轮的LSH，然后将这些轮中相同bucket的query取并集。在$n_{rounds}$ 中对于每一轮，我们都有各自的不同的hash 函数$\{h^{1}, h^{2}, ...  \}$:
\begin{equation}
P_i =  \cup _ {r=1} ^ {n_{rounds}} P_i ^ {(r)} \ \ \ where\ P_i ^{(r)}  = \{j: h^{(r)} (q_i) = h^{(r)}(q_j) \}
\end{equation}

> **Causal masking for shared-QK attention**. 这个之后补充
In a Transformer decoder, masking (denoted by m(j, P i ) in Equation 3) is used to prevent positions from attending into the future. To implement masking in LSH attention, we associate every query/key vector with a position index, re-order the position indices using the same permutations used to sort the query/key vectors, and then use a comparison operation to compute the mask.

## Reverible Transformer
对于我们的transformer中的sub-encoder我们的attention和ffn之间的相连，都需要记忆其中的activations，对于多层以及多个sub-encoder，这将会导致大量的内存消耗。我们将借鉴RevNet的思想，我们无需保存中间层的activations，只需要知道最后一层的activations就可以得出中间层的activations，注意这边的activations不是指激活函数，而是指激活函数的输入。保存这些输入的意义在于用于back propagation时的参数更新。

### RevNet
[The Reversible Residual Network: Backpropagation Without Storing Activations](<https://arxiv.org/abs/1707.04585>) 提出了RevNet的思想，即每一层的activations可以根据下一层的activations 推导获得，从而我们不需要在内存中储存activations。
在原本的residual layer中，我们的输出activations 是由 $y = x + F(x)$ 得到。其中F是residual 函数。

而在RevNet中，首先将输入 $x$ 分为两个部分 $x_1$ 和 $x_2$ 然后通过不同residual functions： $F(·)$ 和 $G(·)$ 得到输出 $y_1$ 和 $y_2$ 。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/5d1eed3e3caba352faa1c32d2c889b8c.png)其中我们根据以下结构，可以从输出获得输入。

\begin{equation}
\begin{split}
y_1 &= x_1 + F(x_2) \\
y_2 &= x_2 + G(y_1)
\end{split}
\end{equation}
由此可以推导：
\begin{equation}
\begin{split}
x_2 &= y_2 - G(y_1) \\
x_1 &= y_1 - F(x_2)
\end{split}
\end{equation}

### RevTransformer
在transformer的sub-encoder block之中，我们的attention layer和 FFN layer是通过ResNET 相连的，所以我们就可以将这个转化为RevNet，从而减少内存的消耗。

我们令F 函数作为我们的attention 层，G 函数作为FFN 层。（注意我们的layer normalization是包含在residual blocks中的）。

\begin{equation}
\begin{split}
y_1 &= x_1 + Attention(x_2) \\
y_2 &= x_2 + FFN(y_1)
\end{split}
\end{equation}

## Chunking
在FFN中，我们例如两层的FFN，通常中间隐藏层的纬度会非常大，例如 $d_{ff} = 4k$ 或者更大。 我们通常是一次性计算完全部，但是我们知道FFN的输入是独立的，所以我们为了降低memory的使用，可以进行chunk拆分计算, 每次计算一个chunk，通过时间消耗获取空间。

\begin{equation}
\begin{split}
y_2 &= x_2 + FFN(y_1) \\
&= [y_2^{(1)}; y_2^{(2)};...;y_2^{(c)}] \\
&= [x_2 ^{(1)} + FFN(y_1 ^{(1)}); x_2 ^{(2)} + FFN(y_1 ^{(2)});...; x_2 ^{(c)} + FFN(y_1 ^{(c)})]
\end{split}
\end{equation}

