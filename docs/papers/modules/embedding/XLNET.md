
# XLNET
[XLNET](https://arxiv.org/pdf/1906.08237.pdf) 采用的是transformer-XL的encoder，采用了的是auto regressive的语言模型，而为了加上双向的信息，采用了输入序列的permutation，但是如果再输入的时候就做permutation，那占用的空间非常大，所以采用了特殊的two-stream self-attention来模拟permutation的作用。

论文感觉不简单，需要多读几遍才能彻底理解。
## 改进思路
- 首先 auto regressive是指的是模型预测下一个词的语言模型。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/bbaf62600286dc8617201d2a09e718bc.png)
  其主要的优点在于：
    - 可以用来生成句子做NLG任务
    - 考虑到了邻近词间的相关性
    - 无监督

  但是其的缺点也很明显：
    - 单向
    - 离得近未必有关系


- BERT采用的是Auto Encoding的方式，主要采用的预测Masked words，用的是没有被masked的hidden output来预测masked 词
  ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/115ed223ae33585fa5cab43338dff19f.png)
  其主要的优点在于：
    - 双向，pretraining+finetuning

  缺点：
    - 预训练和fine-tuning之间mask具有discrepancy，fine-tune 阶段没有[MASK]
    - BERT假设的是MASK词之间是独立的，比如[NEW YOR IS A CITY], mask 之后是[MASK MASK IS A CITY], 两个MASK词之间没有关系。（但是这个不能算是缺点，因为证明这样效果也不错）

- 改进思路：
  - bert 基础上+改进使之能够生成（UniLM）
  - LM基础+改进(XLNET)
    - [NADE](http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf)(双向能力)
    - TRANSFORMER XL

最后模型具备了具备生成能力，拥有了双向的信息，同时又可以进行pretraining+fintuning，且两阶段之间没有mismatch。

## LM怎么考虑到上下文（左右）

联合概率
- bayese network(有依赖关系）
$P(w_1 w_2 w_3 w_4 w_5) =  P(w_1)P(w_2|w_1)P(w_3|w_1)P(w_4|w_2 w_3)P(w_5|w_4)$
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/e47a255bde5e108bcd04e5939598b2f8.png)
- HMM（我不知道依赖关系，我靠的是假设）
$P(w_1 w_2 w_3 w_4 w_5) =  P(w_1)P(w_2|w_1)P(w_3|w_1 w_2)P(w_4|w_1 w_2 w_3)P(w_5|w_1 w_2 w_3 w_4)$
$P(w_1 w_2 w_3 w_4 w_5) = P(w_2 w_1 w_3 w_4 w_5) = P(w_2)P(w_1|w_2)P(w_3|w_2 w_1)P(w_4|w_2 w_1 w_3)P(w_5|w_2 w_1 w_3 w_4)$
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/27f25099b9aab55e3c787b7f724f84ce.png)
这五个变量，可以有permutation的表示，这五个变量是没有关系的
思路是来自于 [NADE](http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf)的文章


当permutation 考虑完，相当于考虑到了单词的上下文。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/fa704e8d083bc17bccd33f98f0558143.png)
（4!=24个不同的独立语言模型）
最大期望 Expectation， 每个语言模型的权重如果是uniform的分布的话是 $\frac 1{n!}$
\begin{equation}
max\ _{\theta} \ E_{z∼Z_T}[ \sum _{t=1} ^{T} log p_θ(x_{z_t}| x_{z<t})]
\end{equation}
对于两个token的序列，有两种情况，分别是$w_1 w_2$（1），$w_2 w_1$（2）
对于同一个训练的example，后面的是不会看到前面的，即（1）中$w_2$ 看不到$w_1$, 即（2）中$w_1$ 看不到$w_2$
但是在这个模型中，不同的训练example间，是可能看到之前的token的，例如在整个训练过程中$w_1$互相看到了$w_2$。这个是无所谓的，这就好像是BERT中，同一条训练数据，mask不同的值。

最大的目标是考虑了所有n!种可能$Z_T$，找到一组模型的参数使得所有的序列的auto regression期望最大。
即：
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/a543969047d3fd5dbb79fa7e63dcb245.png)
但是这种方式的计算量非常大，特别是对于N很大的序列，我们可以采取的是采样以及对于每一种permutation（i.e. LM） 都是对于有的位置信息量很足有的信息量进行预测：
- 抽样 permutation （对于permutation的组合）
- predict last few tokens instead of all （对于一个组合中的序列）
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/5569e673f9c0a813f9856233d37610b0.png)

可以改进：
1. 采样的优化
  如何选择最好的permutation？
2. expecation的优化，目前采用的是union的期望。

## TransformerXL
- ADD Memory
- Relative Position Encoding
具体参照TransformerXL的笔记。

## 2-stream self-attention

### Naive implementation
在实现的时候，直接使用softmax进行next token prediction$p_\theta(X_{z_t} |x_{z<t})$ 其中
$z_t$: 代表的是要预测的位置
$x_{z<t}$: 代表的是当前要预测的位置之前的信息
\begin{equation}
p_\theta(X_{z_t} |x_{z<t}) = \frac {exp(e(x)^T h_{\theta} (x_{z<t}) } { \sum _{x'} exp(e(x')^T h_{\theta} (x_{z<t}) }
\end{equation}
在这种naive的实现中，并没有考虑到要预测的token的位置信息$z_t$, 所以对于导致不同位置的预测结果一样。
例如：
序列的2个permutation：
1，2，4，3
1，2，3，4
我们要根据1，2预测下一个值
分别得到：

\begin{equation}
\begin{split}
（1） = p_\theta(X_{z_4} |x_{z<4}) = \frac {exp(e(x)^T h_{\theta} (x_{z<4}) } { \sum _{x^{\prime} } exp(e(x^{\prime} )^T h_{\theta} (x_{z<4}) }\\
（2） = p_\theta(X_{z_3} |x_{z<3}) = \frac {exp(e(x)^T h_{\theta} (x_{z<3}) } { \sum _{x^{\prime} } exp(e(x^{\prime} )^T h_{\theta} (x_{z<3}) }
\end{split}
\end{equation}

但是（1）= (2)，但是这两个token的位置并不相同。所以这种会导致问题构造的失败。


### 2-stream include position info

XLNET为了解决这个问题，将 $h_{\theta} (x_{z<t}) $ 引申为 $g_{\theta} (x_{z<t}, z_t) $，加入了位置信息。为此改变了模型结构。
\begin{equation}
p_\theta(X_{z_t} |x_{z<t}) = \frac {exp(e(x)^T g_{\theta} (x_{z<t}, z_t) } { \sum _{x^{\prime} } exp(e(x^{\prime} )^T g_{\theta} (x_{z<t}, z_t) }
\end{equation}

其中$g_{\theta} (x_{z<t}, z_t) $的作用是，利用$z_t$的位置信息，结合内容信息$x_{z<t}$ 预测下个词。

我们并不是直接将输入进行permutation之后再传入模型的，而是将正常顺序的输入传入模型。但是我们希望进行模型的并行计算，同时模拟出permutation的效果，即能够产生不同的mask，并且我们需要模型的预测是根据当前位置以及之前的输入得到的。

所以我们需要的是self-attention进行并行计算，对于permutation的模拟，采用的是attention的mask矩阵，但是我们如何模拟attend to position但是不attend to 它的context呢，XLNET采用的是2-stream self-attention，即将输入分为$h_i^l$ 和$g_i^l$， $h_i^l$关注的是context的值，$g_i^l$关注的是query的值

- context stream 就和普通的self-attention一样编码的是内容信息，但是是基于lookahead的mask 策略，即只能看到自己以及之前位置的内容信息。
- query stream 编码的是位置信息，可以看到自己的**位置**信息，还有**之前的内容信息**但是不能看到自己的内容信息。

对于顺序为3 —> 2 —>  4 —> 1来说，它的attention masks是以下的情况，这边mask对应的都是content的embedding。就拿位置为2的token来说，它的content stream只能看到3以及2的content embedding，（即第二行的2，3为填充），而对于query stream来说，它只能根据之前的content embedding即3位置上的content作出判断（G_2只有第三个位置有填充）。当然它还可以看到自己当前的位置作为依据，但是这个matrix是指的是content embedding的mask，即为涂上的$h_k$。
  ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/e16b8a17c884c9809ba08bf74393bfe3.png)


![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/fb01e156520c561ae8a9df19546cb4bc.png)
\begin{equation}
g_{z_t} ^m = Attention(Q=g_{z_t} ^{m-1}, KV = h_{z<t}^{m-1};\theta) \ (query\ stream: 使用了z_t但是不能看到x_{z_t}\\
h_{z_t} ^m = Attention(Q=h_{z_t} ^{m-1}, KV = h_{z \leq t}^{m-1};\theta) \ (content\ stream: 使用了z_t和x_{z_t})
\end{equation}

但是$g_{z_t} ^m$ 可以获得$g_{z_t} ^{m-1}$ 的信息，并且最后的输出是根据$g_{z_t} ^m$预测的。

注意，这边的采用的是partial prediction：
即不预测所有的token，而是只预测这些permutation sequence的最后几个tokens，因为这样子能保证这些tokens获得的前面的信息足够多。

## 实现细节
- 双向输入
> Since the recurrence mechanism is introduced, we use a bidirectional data input pipeline where each of the forward and backward directions takes half of the batch size.

- span prediction
- 去掉NSP
- 更多数据
http://fancyerii.github.io/2019/06/30/xlnet-theory/#two-stream-self-attention-for-target-aware-representations


http://fancyerii.github.io/2019/06/30/xlnet-theory/
这个code好难啊我现在还看不太懂
目前只看到了生成Pretraining的数据这一部分（第一部分）

## Ablation studies
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/d17f342dbf2abf6ba3af5fd84a738ae8.png)

但是最后RoBERTa发现其实BERT还是比XLNET强，但是XLNET的里面的很多思想都是被其他使用的例如no NSP，例如Span prediction，总的来说这是一个超棒的paper。

# Ref
+ https://www.bilibili.com/video/av73657563?p=2
+ https://arxiv.org/pdf/1906.08237.pdf
+ https://blog.csdn.net/u012526436/article/details/93196139
+ http://www.emventures.cn/blog/recurrent-ai-cmu-xlnet-18-nlp
+ https://indexfziq.github.io/2019/06/21/XLNet/
+ https://blog.csdn.net/weixin_37947156/article/details/93035607