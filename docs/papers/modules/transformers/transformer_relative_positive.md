# Self-Attention with Relative Position Representations
## Self-Attention with Relative Position Representations
[Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155.pdf)
论文中还发现，+relative position encoding 在transformer的translation的task上得到了提升，但是结合了absolute以及relative的话，效果没提升。

论文很短，很容易理解。

首先我们先了解在self-attention中，我们$e_{ij}, a_{ij}, z_i$ 的计算：
\begin{aligned}
e _ { i j } &= \frac { \left( x _ { i } W ^ { Q } \right) \left( x _ { j } W ^ { K } \right) ^ { T } } { \sqrt { d _ { z } } } \\
\alpha _ { i j } &= \frac { \exp e _ { i j } } { \sum _ { k = 1 } ^ { n } \exp e _ { i k } } \\
z _ { i } &= \sum _ { j = 1 } ^ { n } \alpha _ { i j } \left( x _ { j } W ^ { V } \right)\\ \tag{$1$}
\end{aligned}

### Relation-aware Self-Attention
文章中引入了两个位置相关的向量，量：$a _ { i j } ^ { V } , a _ { i j } ^ { K } \in \mathbb { R } ^ { d _ { z } }$，之所以采用$d_a$维向量的表示形式，主要是为了套用原来self-attention的计算公式， 因为$xW$的维度是这个。$a _ { i j } ^ { V } , a _ { i j } ^ { K }$ 是在所有的attention layer中共享的。

在引入了这两个相对位置信息向量之后上式（1）将改编为：
\begin{aligned}
e _ { i j } &= \frac { \left( x _ { i } W ^ { Q } \right) \left( x _ { j } W ^ { K } + a ^ { K } _ {i j } \right) ^ { T } } { \sqrt { d _ { z } } } \\
\alpha _ { i j } &= \frac { \exp e _ { i j } } { \sum _ { k = 1 } ^ { n } \exp e _ { i k } } \\
z _ { i } &= \sum _ { j = 1 } ^ { n } \alpha _ { i j } \left( x _ { j } W ^ { V } + a _ {i j} ^ { V } \right)\\ \tag{$2$}
\end{aligned}


### Relative Position Representations
Relative Position Representations的目标是给出$a _ { i j } ^ { V } , a _ { i j } ^ { K }$的计算方式。作者假设如果序列中两个元素的距离超过k，则这两元素之间的位置信息就没有意义了。同时，$a _ { i j } ^ { V } , a _ { i j } ^ { K }$应该只跟相对位置有关，而与$x_i, x_j$没有关系。作者直接将$a _ { i j } ^ { V } , a _ { i j } ^ { K }$定义为了可训练的向量，本质上是训练$w ^ { K } = \left( w _ { - k } ^ { K } , \ldots , w _ { k } ^ { K } \right)$和$w ^ { V } = \left( w _ { - k } ^ { V } , \ldots , w _ { k } ^ { V } \right)$：
\begin{aligned}
a _ { i j } ^ { K } & = w _ { \operatorname { clip }( j - i , k ) } ^ { K } \\
a _ { i j } ^ { V } & = w _ { \operatorname { clip } ( j - i , k ) } ^ { V } \\
\operatorname { clip } ( x , k ) & = \max ( - k , \min ( k , x ) )
\end{aligned}
其中`clip`函数的作用就是截断$j-i$的长度，使得其落在$[-k,k]$之间

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/1f4e0048485c7d55553fcc9e33c2fd1c.png)
A矩阵的示意图，k代表了考虑的距离，箭头表示的一对相对位置表示。


注意：这边的主要给出了$a_{i,j}$的表示方式，这是论文中最难的部分，但是理解了就不难了，其实就是一个一个可训练的矩阵

### Implement

\begin{aligned}
e _ { i j } &= \frac { \left( x _ { i } W ^ { Q } \right) \left( x _ { j } W ^ { K } + a ^ { K } _ {i j } \right) ^ { T } } { \sqrt { d _ { z } } } \\
&=\frac {x_i W^Q (x_j W^k)^T}{\sqrt {d_z}} + \frac {x_i W^Q  (a^K_{ij})^T}{\sqrt {d_z}}
 \tag{$3$}
\end{aligned}

> tensor reshaping can be used to compute n parallel multiplications of bh×d zand d z×n matrices. Each matrix multiplication computes contributions to eij for all heads and batches, corresponding to a particular sequence position. Further reshaping allows adding the two terms. The same approach can be used to efﬁciently compute z_i

## ref
[Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155.pdf)
[Self-Attention with Relative Position Representations 解读](https://blog.csdn.net/weixin_41089007/article/details/91477253)