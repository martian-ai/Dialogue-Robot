
# NEZHA
[NEZHA](https://arxiv.org/pdf/1909.00204.pdf) 就是一个针对中文预训练模型的一个整合报告。它首次将函数型的相对位置编码加入了模型中。其实文章中的实作经验就是多篇论文的整合，模型得到了不错的效果，目前应该是仅次于[ZEN](https://arxiv.org/abs/1911.00720)和[ERNIE 2.0: A Continual Pre-Training Framework for Language Understanding](https://arxiv.org/pdf/1907.12412.pdf) 百度ERNIE2.0。

文章的意义主要在于，他提供了多个训练的技巧。

## Functional Relative Position Encoding
- [Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155.pdf)提出了相对位置排序，但是这个是基于parametric的方式。（PRPE）
  >文章中引入了两个位置相关的向量，量：$a _ { i j } ^ { V } , a _ { i j } ^ { K } \in \mathbb { R } ^ { d _ { z } }$，之所以采用$d_a$维向量的表示形式，主要是为了套用原来self-attention的计算公式， 因为$xW$的维度是这个。$a _ { i j } ^ { V } , a _ { i j } ^ { K }$ 是在所有的attention layer中共享的。

  > 在引入了这两个相对位置信息向量之后：
  \begin{aligned}
  e _ { i j } &= \frac { \left( x _ { i } W ^ { Q } \right) \left( x _ { j } W ^ { K } + a ^ { K } _ {i j } \right) ^ { T } } { \sqrt { d _ { z } } } \\
  \alpha _ { i j } &= \frac { \exp e _ { i j } } { \sum _ { k = 1 } ^ { n } \exp e _ { i k } } \\
  z _ { i } &= \sum _ { j = 1 } ^ { n } \alpha _ { i j } \left( x _ { j } W ^ { V } + a _ {i j} ^ { V } \right)\\ \tag{$1$}
  \end{aligned}

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) 提出的方式是Functional Absolute Position Encoding（FAPE）。文章没对比，但是这个在BERT中证明和PAPE效果差不多。
  > 这边的函数使用的是正弦位置编码：
  \begin{equation}
  PE(pos, k)=\left\{
  \begin{aligned}
  sin(\frac {pos}{10000^{\frac {k}{d_{model}}}}) &  & if \ k 是偶数 \ k>=0 \\
  cos(\frac {pos}{10000^{\frac {k-1}{d_{model}}}}) &  & if \ k 是奇数 \ k>=1
  \end{aligned}
  \right.
  \end{equation}

  > - $d_{model}$指的是模型输出的embedding size, 每个attention的hidden size（ie. total hidden size/ number of head)
  > - pos 代表是字在序列中的位置
  > - $k$代表的是position embedding 之后的第$k$维，即$[pe_0,...,pe_k,.. pe_n]$
    这个公式比较具有迷惑性，特别是论文中的写法，结合例子就比较好理解了，如pos=3,d(model)=128,那么3对应的位置向量如下：
    $[sin(\frac 3{10000^{\frac {0}{128}}}), cos(\frac 3{10000^{\frac {0}{128}}}), sin(\frac 3{10000^{\frac {2}{128}}}), cos(\frac 3{10000^{\frac {2}{128}}}), sin(\frac 3{10000^{\frac {4}{128}}}), cos(\frac 3{10000^{\frac {4}{128}}})...]$
    ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/1aae8e9924d01206a740726bd88b1b01.png)

- [BERT（**B**idirectional **E**ncoder **R**epresentations from **T**ransformers）](https://arxiv.org/pdf/1810.04805.pdf) 使用的是Parametric Absolute Position Encoding的方式。（PAPE）
即直接将[1,2,3...Max_leng] 经过position embedding lookup得到embedding。

- 本文采用的是Funcional Relative Position Encoding（FRPE）
  \begin{equation}
  a_{ij}(j-i, k)=\left\{
  \begin{aligned}
  sin(\frac {j-i}{10000^{\frac {i}{d_{model}}}}) &  & if \ i 是偶数 \ i>=0 \\
  cos(\frac {j-i}{10000^{\frac {i-1}{d_{model}}}}) &  & if \ i 是奇数 \ i>=1
  \end{aligned}
  \right.
  \end{equation}

  - $d_{model}$指的是模型输出的embedding size, 每个attention的hidden size（ie. total hidden size/ number of head)
  - j-i 代表是需要attend的字在序列中的相对位置
  - $k$代表的是position embedding 之后的第$k$维结合例子就比较好理解了，如果一句话是`我喜欢吃苹果`，其中如果是`我`（i=0）对于`吃`(j=3)做attention，则相对位置的表示为j-i=3,d(model)=128,那么3对应的位置向量如下：
    $[sin(\frac 3{10000^{\frac {0}{128}}}), cos(\frac 3{10000^{\frac {0}{128}}}), sin(\frac 3{10000^{\frac {2}{128}}}), cos(\frac 3{10000^{\frac {2}{128}}}), sin(\frac 3{10000^{\frac {4}{128}}}), cos(\frac 3{10000^{\frac {4}{128}}})...]$

  在引入了这两个相对位置信息向量之后：$a_{ij}^V$ 和 $a_{ij}^K$ 都是从上面式子中来的，（这不就是一样吗？具体细节看看code，j-i有没有clip？）
  \begin{aligned}
  e _ { i j } &= \frac { \left( x _ { i } W ^ { Q } \right) \left( x _ { j } W ^ { K } + a ^ { K } _ {i j } \right) ^ { T } } { \sqrt { d _ { z } } } \\
  \alpha _ { i j } &= \frac { \exp e _ { i j } } { \sum _ { k = 1 } ^ { n } \exp e _ { i k } } \\
  z _ { i } &= \sum _ { j = 1 } ^ { n } \alpha _ { i j } \left( x _ { j } W ^ { V } + a _ {i j} ^ { V } \right)\\ \tag{$1$}
  \end{aligned}
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/fb301cf2f1bfe658a67f6eb9e84a59ec.png)

## Implemenntation
- More Data
  - Chinese Wikipedia
  - Baidu Baike
  - Chinese News. 爬虫
- LAMB optimizer
- Whole word masking
- Mixed Precision Training

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/f0a0aaba35814efd9d1c3ae09471d48a.png)