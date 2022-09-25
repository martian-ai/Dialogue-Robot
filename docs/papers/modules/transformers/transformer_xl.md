# Transformer-XL

## The motivation for Transformer-XL.
首先，为什么会提出transformerXL呢，它的提出主要是为了解决transformer的问题。我们首先先分析一下RNN以及Transformer的优缺点。
- RNN
  - 优点：
    - 支持可变长
    - 支持记忆
    - 有序列顺序关系
  - 缺点：
    - gradient vanish
    - 耗时，无法并行
- Transformer
  - 优点：
    - 并行
    - 考虑到sequence的long term dependency信息（相对于RNN）
    - 可解释性

  - 缺点：
    - 句子与句子之间的关系
    - 空间占用大（因为我每个encoder的score matrix（sequenceLen*sequecenLen是$N^2$的空间复杂度(BOOOOM!💥)
    如下图
    ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/e6cd1de1a51866eed26229f0d0a7ba59.png)
    ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/445c0be69eb60aa340c770da4e97e8e6.png)
    - batch size也不能很大
  - 解决的方案，将文章documnet切成segments，喂给transformer
  ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/724c1f23966930d26a817b9e63214aa2.png)
  但是segment之间没有信息传递，This problem is called context fragmentation.！

  > The daughter had a nice umbrella that her mother gave her.
  `daughter` and `her` are in different segment
  ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/922db1d2707e08b212ecedd7569242dc.png)
  前后句就不能够了解这个雨伞是他妈妈给的

那么如果解决这个问题呢？我们其实只需要使用RNN的 hidden state来解决信息的传递，我们在不同的segment之间传入传递memory信息。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/90212426d6b30a7b6980086078c7490c.png)
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/6b9c243803fb86923010ae973f28e450.png)

## Transformer-XL: the proposed solution: Basic idea.
所以transformer：（1+2: positional embedding， 3: stacks of encoders）

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/a23007c68ea1202b673b4a0e763cd6af.png)

升级变成下图（注意是embedding/hidden output的concat，不是score的concat）

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/d4c685ff332774313c38f0378a589c09.png)

可以简单的理解 transformerXL = transformer + RNN => segment-wise的RNN
[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860)
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/8852e25c0b80d8ce3ad586c218c23315.png)

```
对于所有的encoder i 除了最后一个encoder
  Set h_{-1,i } 为全0矩阵，矩阵形状和之后的segment的output矩阵形状一致
当我们计算 segment 0时:
  对于所有的encoder i 除了最后一个encoder:
    Combine the saved hidden states: h_{-1,i-1} and h_{0,i-1}.
  对于所有的encoder i 除了最后一个encoder:
    Make a copy of its output h_{0,i }(hidden state).
当我们计算segment 1时:
  对于所有的encoder i 除了最后一个encoder:
    Combine the saved hidden states: h_{0,i-1} and h_{1,i-1}.
  对于所有的encoder i 除了最后一个encoder:
    Make a copy of its output h_{1,i }(hidden state).
…
当我们计算 segment t:
  对于所有的encoder i 除了最后一个encoder:
    Combine the saved hidden states: h_{t-1,i-1} and h_{t,i-1}.
  对于所有的encoder i 除了最后一个encoder:
    Make a copy of its output h_{t,i }(hidden state).
* This shape will be (batch_size, segment_len, emb_dim).
```

### combine hidden states
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/48e92a26d7a3f557d86b844a4b22e8e1.png)
我们来看看如何`Combine the saved hidden states: h_{t-1,i-1} and h_{t,i-1}.`，其实很简单，就是直接直接在 segment 这个维度上面concat起来。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/2d6e82257da07322a8ad2583d01656b0.png)

原本的输出shape(batch\_size, segment\_len, emb\_dim), 现在的combinate之后，输出变成了(batch\_size, 2*segment\_len, emb\_dim)

值得注意的是，在训练的时候，我们是不用反向传播更新我们的memery的，我们的memory是之前的sequence的结果，我们可以在pytorch中设置`.requires_grad=False`。

### how to compute self-attention
在做self-attention的时候，输入的$h_{t,i}$作为from\_tensor 和to\_tensor自己attend to 自己，$h_{t,i}$用来产生Q，K，V矩阵，但是在transformer-XL中，我们的query Q用的仍然是我们的输入$h_{t,i}$产生，但是K，V，都是用的是$conc\_h_{t,i}$, 其中$conc\_h_{t,i}= [h_{t-1},i;h_{t,i}]$
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/bafada54b454382239691525f950e9d8.png)

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/f5fedf7a540ccec1d90514bb6a6fdd42.png)

softmax 出来的结果：
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/3dafddcdc3cdba028382d7a20e49c4d0.png)

对于decoder来说我们需要加上一个look-ahead mask，就和trasnformer

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/1c8b926d22b357fd9611abdfc4230347.png)

我们每次都只concat前一次的$h_{t-1,i}$，这是因为我们认为我们前一次的输出已经包括了之前所有的信息了。
## Absolute Positional Encoding & Memory:
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/94add13483e2a6facbd66470be37804d.png)


如果我们继续使用之前的absolute positing encoding的话，对于所有的sequence的序列，只要这个字在序列中的位置一样的话，它的position encoding也会一样，这样的话，对于我们concat之后的输出，我们无法区别每个字的位置。

如下图：`The`和`that`的position encoding完全一样，模型无法区分两者位置区别。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/1bd6566e9cdef43d251fd61d14e1dd67.png)

所以Transformer-XL 首先分析了position encoding在计算中的作用，然后根据这个结果将交互项转化为relative position encoding。

- 分析了每个position encoding在计算中的作用
  - $E+P$: embeddimng+position encoding
  - $(E+P)_{i, .}W^Q$: Q
  - $(W^K)^T(E+P)^T_{.,j}$: $K^T$
  ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/fc8448461cc9ba4b63ba34032262e691.png)
  The notation $(i, •)$ refers to the entire row $i$ and $(•, j)$ to the entire column $j$ .
  经过计算，这个式子可以分为4项。

  - a) 这一项中没有包含$P$ 位置信息，代表的是在第 $i$ 行的字应该对第 $j$ 列的字提供多大的注意力。这是不管他们两个字的位置信息的。

  - b) 这一项捕获的是模型的global attention，指的是一个字在position i 应该要对 position j 付出多大的注意力。例如两个字的位置越远，期望它们之间的注意力越小。

  - c) 这一项捕获的是在row i的字对其他位置的关注信息，例如在position i是一个字"狗"， 应该要对j=i-1 这个位置特别注意，否则可能出现j=i-1是“热”， 出现是“热狗”的情况。

  - d) 这个是c) 的逆向表示，指的是j的字要pay attention to 位置i的字。


- 根据这个观测，转化relative position
  通过了解了每一项的意义，我们了解了两个字的相对位置对这个score的作用。我们将
  b), c) and d) 替换为如下式子。

  ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/0903e5acf67612ca816264c85458ddad.png)

  我们可以看到主要的变化
  - 我们将使用的是相对的position encoding i.e. 取消 $P_{•, j}$ 而采用 $P_{•, i — j}$ 相对位置。
  - 每次使用 $P_{•, i — j}$ 我们都将 $W^k$ 替换为 $WˆR$ (两者的形状相同)。这是为了区别$W^k$（仍使用） 和 $WˆR$，使得两者可以各自捕获有意义的位置信息而不会相互干预，因为$W^R$和$P_{•, i — j}$相匹配，而$W^K$和$E^T_{•,j}$ 像对于。
  - $P_{i,•}W^Q$这一项被替代为 $u$ 和 $v$ ，这两个向量的维度为 (1, d_k)。因为我们使用的是相对位置编码，所以我们并不需要提供绝对位置$i$，所以我们可以直接把整项替换掉。这边使用两个向量的原因是因为一项是更换了相对位置(b)，一项没有(d)，所以这样能够focus on the general position and the position given the word we attend to as its the case of u and v respectively.（这边没有非常理解）


  所以$(QK^T)_{i,j}$的公式被替换为：
  \begin{equation}
  (QK^T)_{i,j} = E_{i,•}W^Q(W^K)^TE^T_{•,j}+u(W^R)^TP^T_{•,i-j}+E_{i,•}W^Q(W^R)^TP^T_{•,i-j}+v(W^K)^TE^T_{•,j}
  \end{equation}


## summary
- Memory between segments
- Change from Absolute to Relative Positional Encoding.

## 应用和不足
最主要的应用是他用在XLNET上
不足的话，memory的公式的设计不好，直接concat。
以及relative position encoding的设计也不是很合理。

## ref
[Dissecting Transformer-XL](https://medium.com/@mromerocalvo/dissecting-transformer-xl-90963e274bd7)

[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860)

[XLNET详解](https://www.bilibili.com/video/av73657563?from=search&seid=11939921467334417999)