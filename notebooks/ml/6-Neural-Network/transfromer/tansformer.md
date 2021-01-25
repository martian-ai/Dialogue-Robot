# Transformers

## Overview

+ Transformer 来自 (Attention is all you need)

+ 抛弃了之前 Encoder-Decoder 模型中 的CNN/RNN，只用Attention 来实现

+ 引入self-attention

+ Transformer 的整个架构就是层叠的self-attention 和 全连接层

+ 左侧 Encoder， 右侧Decoder

+ 文本分类中一般只使用Encoder

+ Decoder 主要用于 NLG

  ![](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/img/encoder.png)

## Self-attention

+ 一般的attention中

  $$ Attention(query, source) = \sum_{i=1}^{Len(x)} Similarity(query, key_i) * value_i$$

+ 在 self-attention 中 认为 query=key=value， 内部做attention，寻找内部联系

+ 优点

  + 可并行化处理，不依赖其他结果
  + 计算复杂度低，self-attention 的计算复杂度是 $n*n*d$ ,  而RNN 是 $n*d*d$ ,  这里n 是指序列长度， d指词向量的维度，一般d>n
  + self-Attention可以很好的捕获全局信息，无论词的位置在哪，词之间的距离都是1，因为计算词之间的关系时是不依赖于其他词的。在大量的文献中表明，self-Attention的长距离信息捕捉能力和RNN相当，远远超过CNN（CNN主要是捕捉局部信息，当然可以通过增加深度来增大感受野，但实验表明即使感受野能涵盖整个句子，也无法较好的捕捉长距离的信息）

## Scaled dot-product attention

+ 公式

  $$ Attention(Q,K,V) = softmax(QK^T/\sqrt d_k) * V$$

+ Q  和 K 的向量维度都是$$d_k$$, V 的向量维度是$$d_v$$

+ 使用点积计算相似度

+ 然而点积的方法面临一个问题，当 $$\sqrt{d_k}$$太大时，点积计算得到的内积会太大，这样会导致softmax的结果非0即1，因此引入了$$\sqrt{d_k}$$来对内积进行缩放

![](http://ww4.sinaimg.cn/large/006tNc79ly1g5d1ohokd8j30ba0dkt8y.jpg)

## Multi-Head Attention

+ 原理

+ 把Q，K，V 通过参数矩阵映射一下，然后再做Attention，把整个过程重复h次，结果再拼接起来

+ 公式

  $$ MultiHead(Q,K,V) = Concat(head_1, head_2, ..., head_h) * W^O$$

  $$where\ head_i = Attention(QW_i^Q, KW_i^K, VW_i^V) $$

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/07bc8576738c21b1e6d3d524d8419667.png)

# Transformer
## 什么是transformer
首先我们先说结论：[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)提出的transformer 其实就是 seq2seq + self attention。 [代码实现, 非常清晰](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/transformer.ipynb#scrollTo=yxKGuXxaBeeE)

seq2seq 任务指的是输入和输出都是序列的任务。例如说法语翻译成英文。

通常来说，Seq2Seq任务最常见的是使用encoder+decoder的模式，先将一个序列编码成一个上下文矩阵，在使用decoder来解码。当然，我们仅仅把context vector作为编码器到解码器的输入。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/96cbeed40a708a4d2347752b3d26f995.png)

这样子往往得不到好的效果，因为我们的编码器的很多信息都无法完全编码在这个向量中，并且我们在解码的时候，对于输入的每个单词的权重是不一致的，所以在NMT任务上，还添加了attention的机制。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/c1a14c2e6a1bf1d087df6e5f072cb693.png)

所以目前来说，我们可以直接先把transformer当成一个黑盒，就是transformer可以当成是一个序列转码的模型，只是它其中用了特殊的self-attention的机制。如下图所示：

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/f5fe62f3d52aabdbb6308890b9d1eac6.png)
## 为什么需要用transformer

在提到为什么需要用transformer的时候，我们需要了解，在没有transformer的时候，我们都是用什么来完成这系列的任务的呢？

其实在之前我们使用的是RNN（或者是其的单向或者双向变种LSTM/GRU等） 来作为编解码器。


[RNN](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)模块每次只能够吃进一个输入token和前一次的隐藏状态，然后得到输出。它的时序结构使得这个模型能够得到长距离的依赖关系，但是这也使得它不能够并行计算，模型效率十分低。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/d925f649b90e036c95b7ac497b722ab3.png)


![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/310192d780c6ffad8b420236023feb00.png)
当然这边的的RNN可以通过CNN替换，从而达到并行的效果，可以看到下图，总共是两层的卷积层，第一层画出了两个filter，每个1D filter的size是2，到了第二层的卷积层的filter的size是3。

第一层的filter考虑的是两个字之间的关联，但是到了第二层，考虑了三个前一层输出的交互，从而考虑到了较长序列之间的关系。比如说这边序列是 $a1,a2,a3,a4$ , 第一层只考虑了 $a1,a2$, ..  $a3,a4$ 的交互，第二层考虑了 $b1,b2,b3$，而 $b1,b2,b3$ 是前一层两两交互关系的结果，所以第二层考虑了 $a1,a2,a3,a4$ 这个序列的结果了。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/ea62a336742765ab475311cc045da03e.png)

但是对于CNN每次一般我们的卷积核设的长度为3/5这种较小的值，对于序列长度较长的，比如512，就需要堆叠多层的卷积层，导致模型过于冗杂。

那么，我们有没有办法提出一个新的模型，能够并行，并且能够考虑到输入序列不同token的权重？聪明的科学家们提出了一种新的模型叫做transformer。

其实他就encoder+decoder模式，只是其中的编解码器采用了self-attention的机制。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/40e63faae5fddbc703ca0bd6351408e8.png)

当然transformer真的就比RNN好吗？有人提出，凡事用RNN做的模型，都可以直接用self-attention替代。这个我们会在transformer的缺点中讨论。# tranformer的内部结构

transformer其实是由encoder以及decoder不是单一模块，而是由小的多个sub-encoder block和sub-decoder block组成。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/a250e6aaab6bffea77f51b24ed18ec5b.png)

我们来看看transformer的具体结构图。由下图所示，它主要由左边的encoder+input以及右边的decoder+input+output组成。我们将会一一介绍。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/8f5f57355d9f80b36541e2560dbd0a3b.png)


### encoder
这边的encoder由input以及多个sub-encoder blocks组成。我们将会先讲sub-encoder，再讲输入，因为输入的设计是为了弥补self-attention的缺陷的。


#### sub-encoder block

首先每个sub-encoder都由两个主要的部分组成（略过部分细节，之后会写），分别是self-attention layer以及ffn layer。


具体的实现机制就是：我们的输入每个词经过embedding 之后，然后经过self-attention ，根据自己的路径，经过转换得到新的输出vector，最后再经过ffn layer，得到新的输出，作为下一层sub-encoder的输入。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/f1152369b656b6ceec2508a3793bbc7d.png)

##### multi-head self-attention

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

##### FFN
在self-attention层之后模型会经过FFN层。
\begin{equation}
  FFN(x) = max(0, xW_1 + b_1 )W_2 + b_2
\end{equation}
这边的实现就是两层的Dense layer，第一层的激活函数是RELU。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/dc7f511bcfb5222efab8d70d7dc6c931.png)
两个sub-layer的连接并不是直接相连，而是先通过ADD&Normalize层，所谓的ADD&Normalize层，由以下两个组成
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/48fb5ef3a7cf3ba82954e044ee5ea9b4.png)
- ADD：将输入+self-attention的输出
- Normalize：在经过[layer-normalization](https://arxiv.org/abs/1607.06450)以及dropout操作。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/ab49f4845e45080ca70151fc111981ae.png)
layer normalization：其实很简单就是每一条样本都经过(x-mean) / std, 其mean和std 都是按照单条样本进行计算的。

#### input
对于encoder的输入，由于self-attention的机制讲没有考虑输入序列的顺序，但是一个句子的输入顺序其实很重要，例如`你喜欢苹果不`,`你不喜欢苹果`，两个句子的含义不同，所以我们需要为输入embedding添加position encoding。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/14a610899f63562e50e455dadb488db5.png)

这边的position encoding，主要可以分为通过序列的关系可以分为
- 绝对位置：例如每个sequence$w_0, w_1, w_2, ...w_n$, 位置都是从0，1..n开始
- 相对位置：位置的表示是由字与字之间的差表示的。相对位置表达[Relative Position Representations (RPR)是Shaw et al., 2018](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1803.02155)，这个论文指出，同一个sequence中使用相对位置更好。

它根据encoding的方式也可以分为，
- functional encoding: 这个是指的是通过特定函数的方式，将输入的位置idx变换为embedding。
- parametric encoding：指的是通过embedding loopup的方式，让模型自己学习位置的embedding
这两种方式的效果都差不多，但是functional的可以减少模型的参数。

BERT使用的是 parametric absolute positional encoding (PAPE)
而transformer使用的是functional absolute positional encoding (FAPE)。

这边的函数使用的是正弦位置编码：
\begin{equation}
PE(pos, i)=\left\{
\begin{aligned}
sin(\frac {pos}{10000^{\frac {i}{d_{model}}}}) &  & if \ i 是偶数 \ i>=0 \\
cos(\frac {pos}{10000^{\frac {i-1}{d_{model}}}}) &  & if \ i 是奇数 \ i>=1
\end{aligned}
\right.
\end{equation}

- $d_{model}$指的是模型输出的embedding size
- pos 代表是字在序列中的位置
- $i$代表的是position embedding 之后的第$i$维，即$[pe_0,...,pe_i,.. pe_n]$
这个公式比较具有迷惑性，特别是论文中的写法，结合例子就比较好理解了，如pos=3,d(model)=128,那么3对应的位置向量如下：
$[sin(\frac 3{10000^{\frac {0}{128}}}), cos(\frac 3{10000^{\frac {0}{128}}}), sin(\frac 3{10000^{\frac {2}{128}}}), cos(\frac 3{10000^{\frac {2}{128}}}), sin(\frac 3{10000^{\frac {4}{128}}}), cos(\frac 3{10000^{\frac {4}{128}}})...]$
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/1aae8e9924d01206a740726bd88b1b01.png)

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/c9dfccc609d01f4179658ed94ce58743.png)

这个编码函数的可视化结果：
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/dc5b5bc84a5f9d9b571530a05e438a1d.png)

### decoder
编码器完成之后我们需要解码器进行工作，最后一层的输出会被转化为一组 attention vectors K and V. 作为encoder-decoder attention层的K，V矩阵使用，这些能够帮助decoder关注输入的合适位置。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/75640465b049103d1fa84368216589da.png)
每一个timestamp的输出都会被喂给decoder，我们将这个输出做embedding 输出在添加position encoding。decoder的解码工作的停止条件就是知道特殊字符\<end of sentence\> 得到了。

#### input with look-ahead mask
decoder的输入和encoder的输入不太一样，引文decoder的self-attention layer只能够关注输出序列当前位置以及之前的字，不能够关注之后的字。所以这边需要将这之后的字都添加上mask，即q*k之后加上负无穷(-inf)，使得其再经过softmax之后的权重变为0。

> The look-ahead mask is used to mask the future tokens in a sequence. In other words, the mask indicates which entries should not be used.

look-ahead mask 是用来mask序列的future tokens。具体的做法如下：

```Python
def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

x = tf.random.uniform((1, 3))
temp = create_look_ahead_mask(x.shape[1])
>><tf.Tensor: shape=(3, 3), dtype=float32, numpy=
>>array([[0., 1., 1.],
>>       [0., 0., 1.],
>>       [0., 0., 0.]], dtype=float32)>
```

刚看到这边的时候，我有个问题，就是decoder的每次timestamp的输入不都是之前的前一次的输出吗，如何并行？这不是跟RNN一样？
但是其实在训练的时候，我们是把所有的target 的序列直接作为decoder的输入的！然后通过look-ahead mask来模拟不同timestamp。
```Python
sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                         dff=2048, target_vocab_size=8000,
                         maximum_position_encoding=5000)
target_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

output, attn = sample_decoder(target_input,
                              enc_output=sample_encoder_output,
                              training=False,
                              look_ahead_mask=None,
                              padding_mask=None)
```
在预测的时候，才是真正将decoder的输出作为下一次的输入。但这时候模型已经是一个黑盒了。
```Python
def evaluate(inp_sentence):
  start_token = [tokenizer_pt.vocab_size]

  end_token = [tokenizer_pt.vocab_size + 1]

  # inp sentence is portuguese, hence adding the start and end token
  inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)

  # as the target is english, the first word to the transformer should be the
  # english start token.
  decoder_input = [tokenizer_en.vocab_size] # <start_of_sentence>
  output = tf.expand_dims(decoder_input, 0)

  for i in range(MAX_LENGTH):
    print(output)
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)
    predictions, attention_weights = transformer(encoder_input,
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)

    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # return the result if the predicted_id is equal to the end token
    if predicted_id == tokenizer_en.vocab_size+1: # <end_of_sentence>
      return tf.squeeze(output, axis=0), attention_weights

    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights
translate("este é um problema que temos que resolver.")
print ("Real translation: this is a problem we have to solve .")
>> tf.Tensor([[8087]], shape=(1, 1), dtype=int32)
>> tf.Tensor([[8087   16]], shape=(1, 2), dtype=int32)
>> tf.Tensor([[8087   16   13]], shape=(1, 3), dtype=int32)
>> tf.Tensor([[8087   16   13    7]], shape=(1, 4), dtype=int32)
>> tf.Tensor([[8087   16   13    7  328]], shape=(1, 5), dtype=int32)
>> tf.Tensor([[8087   16   13    7  328   10]], shape=(1, 6), dtype=int32)
>> tf.Tensor([[8087   16   13    7  328   10   14]], shape=(1, 7), dtype=int32)
>> tf.Tensor([[8087   16   13    7  328   10   14   24]], shape=(1, 8), dtype=int32)
>> tf.Tensor([[8087   16   13    7  328   10   14   24    5]], shape=(1, 9), dtype=int32)
>> tf.Tensor([[8087   16   13    7  328   10   14   24    5  966]], shape=(1, 10), dtype=int32)
>> tf.Tensor([[8087   16   13    7  328   10   14   24    5  966   19]], shape=(1, 11), dtype=int32)
>> tf.Tensor([[8087   16   13    7  328   10   14   24    5  966   19    2]], shape=(1, 12), dtype=int32)
Input: este é um problema que temos que resolver.
Predicted translation: this is a problem that we have to solve it .
Real translation: this is a problem we have to solve .
```
#### sub-decoder block
sub-decoder block 跟encoder几乎一样，只是它比普通的encoder多了一个Encoder-Decoder Attention，The “Encoder-Decoder Attention” layer和multiheaded self-attention的工作机制一样，除了它使用的是 Keys 和 Values matrix 是encoder的输出, 这就意味着，我们decoder的query考虑到了encoder的所有的字了。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/15298b4a3f34793e085fdfc685585751.png)

### output layer

decoder的output是一个vector，这时候再经过一个dense层得到vocabulary size的logits，再经过softmax在取argmax得到输出的字。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/5098921cad68582885fa786255f564c8.png)

### summary
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/b2bc1b6248c46fefa880dbb69adaa501.png)

```Python
class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                           input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                           target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inp, tar, training, enc_padding_mask,
           look_ahead_mask, dec_padding_mask):

    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)

    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output, attention_weights
```
## transformer的缺点
- tranformer 的空间以及时间复杂度非常大，sequence length $L$, 达到了$O(L^2)$，这是因为每一层的self attention 都要储存$L^2$的score用于之后的更新，所以L的长度不能很大，否则会遇到OOM的问题。在这种情况下，如果一个句子特别长, 那么他就不得不被分为两个sequence作为输入，但是这个时候前后句子之间的关系就没了，但是RNN可以不管多长的输入都能handle。
- 运行时间太慢，模型太大
- position encoding 使用absolute encoding，而[Self-Attention with Relative Position Representations](https://link.zhihu.com/?target=https%253A//arxiv.org/abs/1803.02155)指出了相对位置更好

## transformer的应用
翻译等， summary

## ref

[李宏毅transformer](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2019/Lecture/Transformer%20(v5).pdf)

[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

[the-illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)

[The Evolved Transformer – Enhancing Transformer with Neural Architecture Search](https://www.lyrn.ai/2019/03/12/the-evolved-transformer/)

[Transformer-XL – Combining Transformers and RNNs Into a State-of-the-art Language Model7](https://www.lyrn.ai/2019/01/16/transformer-xl-sota-language-model/)

[code](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/transformer.ipynb#scrollTo=a1jXoAMRZyvu&uniqifier=2)


----
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
---
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

# Reference

- http://tylerneylon.com/a/lsh1/
- https://www.shangyexinzhi.com/article/details/id-460384/
- https://zhuanzhi.ai/vip/2aa277566082fc7bb38a862be6f8ed40
- https://wemp.app/posts/70fb630c-f780-4408-9929-ecb910a86c5a
- https://ai.googleblog.com/2020/01/reformer-efficient-transformer.html
- https://arxiv.org/pdf/1509.02897.pdf
- https://www.youtube.com/watch?v=EulWJgvNWfM

- Attention is All you need
  - https://arxiv.org/pdf/1706.03762.pdf
- Universal Transformer
  - https://arxiv.org/pdf/1807.03819.pdf
  - https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer.py
- https://www.cnblogs.com/jiangxinyang/p/10210813.html
- https://www.cnblogs.com/jiangxinyang/p/10069330.html
- [](https://kexue.fm/archives/4765)
- https://blog.csdn.net/malefactor/article/details/78767781
- The Illustrated Transformer
  - https://jalammar.github.io/illustrated-transformer/
- The annotated Transformer
  - http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3
- https://zhuanlan.zhihu.com/p/42213742
- https://link.zhihu.com/?target=https%3A//jalammar.github.io/illustrated-transformer/
- https://jalammar.github.io/illustrated-transformer/