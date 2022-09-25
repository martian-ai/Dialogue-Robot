
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

<!-- \begin{equation}
f(x)=\left\{
\begin{aligned}
(x/x_{max})^\alpha &  & ifx <x_{max} \\
1 &  & otherwise    \tag{$3$}
\end{aligned}
\right.
\end{equation} -->
<p align="center">
  <img width="560" height="100" src="http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/b4b50ad2f6cd5e10613f0a8b9a5e36b3.png">
</p>

<p align="center">
  <img width="400" height="180" src="http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/54409df2a8cc506b7d0ef9a912db2f62.png">
</p>

这篇论文中的所有实验，$\alpha$的取值都是0.75，而x_{max}取值都是100。以上就是GloVe的实现细节，那么GloVe是如何训练的呢？


## 如何训练
- unsupervised learning
- label是公式2中的$log(X_ij)$, 需要不断学习的是$w和 \tilde w$
- 训练方式采用的是梯度下降
- 具体：采用AdaGrad的梯度下降算法，对矩阵$X$中的所有非零元素进行随机采样，learning rate=0.05，在vector size小于300的情况下迭代50次，其他的vector size迭代100次，直至收敛。最终学习得到的$w \tilde w$，因为$X$ 是对称的，所以理论上$w 和\tilde w$也是对称的，但是初始化参数不同，导致最终值不一致，所以采用$(w +\tilde w)$ 两者之和 作为最后输出结果，提高鲁棒性
- 在训练了400亿个token组成的语料后，得到的实验结果如下图所示：

<p align="center">
  <img width="400" height="180" src="http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/4f9f168447cbcd4d5ba3879aeee20875.png">
</p>

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

![table_glove](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/9ba582cc1f59d952e71c7216d8f94d7f.png)

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