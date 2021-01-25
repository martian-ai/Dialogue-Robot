# ELMo

[Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)提出了ELMO(Embedding from Language Models), 提出了一个使用**无监督**的**双向**语言模型进行预训练，得到了一个context depenedent的词向量预训练模型，并且这个模型可以很容易地plug in 到现有的模型当中。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/974eaa7c3e0f2cd5c2c4bc40ae2cee72.png)


## model structure
模型主要有两个对称的模型组成，一个是前向的网络，一个是反向的网络。
每一个网络又由两部分组成，一个是embedding layer，一个是LSTM layer；模型结是参考[这个模型结构](http://proceedings.mlr.press/v37/jozefowicz15.pdf)
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/a59ea76090c1d5cbfb2ec32f8a1f44c4.png)
- embedding layers
  - 2048 X n-gram CNN filters
  - 2 X highway layers
  - 1 X projection

- LSTM layers (x2)

TODO：参考<https://github.com/allenai/bilm-tf>以及[这个模型结构](http://proceedings.mlr.press/v37/jozefowicz15.pdf)
> we halved all embedding and hidden dimensions from the single best model CNN-BIG-LSTM in J´ozefowicz et al. (2016). The ﬁnal model uses L = 2 biLSTM layers with 4096 units and 512 dimension projections and a residual connection from the ﬁrst to second layer. The context insensitive type representation uses 2048 character n-gram convolutional ﬁlters followed by two highway layers (Srivastava et al., 2015) and a linear projection down to a 512 representation. As a result, the biLM provides three layers of representations for each input token, including those outside the training set due to the purely character input.

## model training
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/0fa0a28782d7b4ee4c1f821206472e17.png)
给定一个长度为N的序列 $t_1,t_2,...,t_N$, 对于每个token $t_k$，以及它的历史token序列$t_1, t_2, ..., t_{k-1}$ ：
- 对于前向的LSTM LM
\begin{equation}
p_{FWD}(t_1, t_2, ..., t_N) = \prod ^N _{k=1} p(t_k | t_1, t_2,..., t_{k-1})
\end{equation}
- 对于后向的LSTM LM
\begin{equation}
p_{BWD}(t_1, t_2, ..., t_N) = \prod ^N _{k=1} p(t_k | t_{k+1}, t_{k+2},..., t_N)
\end{equation}

所以最后的loss可以写成：
\begin{equation}
loss = 0.5*\sum _{k=1} ^N - (log p_{FWD}(t_1, t_2, ..., t_N; \overrightarrow \theta _x ; \theta _{LSTM} ; \theta _s) + log p_{BWD}(t_1, t_2, ..., t_N; \theta_x; \overleftarrow \theta _{LSTM}; \theta _s))
\end{equation}
其中
- $\theta _x$ 指的是token的表示embedding
- $\theta _s$ 指的是softmax
- $\overrightarrow \theta _x$ 或者$\overleftarrow \theta _x$指的是LSTM的模型

code：https://github.com/allenai/bilm-tf/blob/master/bilm/training.py

## model usage
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/a5455aec0f0b71735172f191d8657672.png)
模型的使用的时候
对于token $k$, 模型的输出embedding为
\begin{equation}
\begin{split}
ELMo_k ^{task} &= \gamma ^{task} \sum _{j=0} ^{L} s_j ^{task} h_{k,j}^{LM} \\
h_{k,j}^{LM} &= [\overrightarrow h_{k,j}^{LM};\overleftarrow h_{k,j}^{LM}], \ \  k=0,1,..L
\end{split}
\end{equation}
其中k=0， 表示的embedding的输出。
即表示的是每一层的输出concat起来之后再做加权平均，其中$s_j ^{task}$ 是softmax-normalized的weight。$\gamma ^{task}$ 指的是特定任务的scaler。

## 分析
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/ca8e05433b1c350e7d39b04e2ea44315.png)
对于
- LSTM第一层：主要embed的是词的语法结构信息，主要是context-independent的信息，主要更适合做POS的任务。
- LSTM第二层：主要embed的是句子的语义信息，主要是context-dependent的信息，可以用来做消歧义的任务。

## Use Demo
```Python
import tensorflow as tf
import tensorflow_hub as hub
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
embeddings = elmo(
    ["the cat ate apple"],
    signature="default",
    as_dict=True)["elmo"]
embeddings_single_word = elmo(
    ["apple"],
    signature="default",
    as_dict=True)["elmo"]
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(embeddings)
  print(sess.run(embeddings))
  print(embeddings_single_word)
  print(sess.run(embeddings_single_word))
```
```Python
INFO:tensorflow:Saver not created because there are no variables in the graph to restore
INFO:tensorflow:Saver not created because there are no variables in the graph to restore
INFO:tensorflow:Saver not created because there are no variables in the graph to restore
INFO:tensorflow:Saver not created because there are no variables in the graph to restore
Tensor("module_11_apply_default/aggregation/mul_3:0", shape=(1, 4, 1024), dtype=float32)
[[[ 0.30815446  0.266304    0.23561305 ... -0.5105163   0.32457852
   -0.16020967]
  [ 0.5142876  -0.13532336  0.11090391 ... -0.1335834   0.06437437
    0.9357102 ]
  [-0.24695906  0.34006292  0.22726282 ...  0.38001215  0.4503531
    0.6617443 ]
  [ 0.8029585  -0.22430336  0.28576007 ...  0.14826387  0.46911317
    0.6117439 ]]]
Tensor("module_11_apply_default_1/aggregation/mul_3:0", shape=(1, ?, 1024), dtype=float32)
[[[ 0.75570786 -0.2999097   0.7435455  ...  0.14826366  0.46911308
    0.61174375]]]
```

## ref
<https://arxiv.org/pdf/1804.07461.pdf>！！！ elmo使用
<https://allennlp.org/tutorials>
<https://github.com/yuanxiaosc/ELMo>
<https://github.com/allenai/bilm-tf>
https://tfhub.dev/google/elmo/3 !!!!
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