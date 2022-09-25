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

## Self-attention()
+ 见 attention.md

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
+ 见 attention.md

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



