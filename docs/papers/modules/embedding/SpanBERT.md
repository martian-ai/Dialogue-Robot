
# SpanBERT
[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/pdf/1907.10529.pdf)
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/349ef38270b757cf7c6ba5d42cb58630.png)
- 没有segment embedding，只有一个长的句子，类似RoBERTa
- Span Masking
- MLM+SBO

意义：提出了为什么没用NSP更好的假设，因为序列更长了。以及提出了使用更好的task能带来明显的优化效果
## Span Masking

文章中使用的随机采样，采用的span的长度采用的是集合概率分布，长度的期望计算使用到了（等比数列求和，以及等比数列和的差等，具体可以参考知乎）

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/27f8c30006751679044747ee75bf469b.png)

mask的采用的是span为单位的。

不同的MASK方式
- Word piece
- Whole Word Masking（BERT-WWM， ERNIE1.0）
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/9cd5ce46ba99939e3d6b43729ab2d2ba.png)
- Named Entity Masking（ERNIE1.0）
- Phrase Masking（ERNIE1.0）
- random Span
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/cf780e2456617cef11bcab47df7a2d2d.png)

实验证明 random Span的效果要好于其他不同的span的策略。但是单独的验证并不能够证明好于几种策略的组合（ERNIE1.0 style）。而且ERNIE1.0只有中文模型。但是这个确实是一个非常厉害的结论。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/d8008383ea6c5a14649ed8dd71dc5323.png)

## W/O NSP W/ Span Boundary Objective(SBO)
### SBO
Span Boundary Objective 使用的是被masked 的span 的左右边界的字（未masked的）以及被mask的字的position，来预测当前被mask的词。

- $x_i$: 在span中的每一个 token 表示
- $y_i$: 在span中的每一个 token 用来预测 $x_i$的输出
- $x_{s-1}$: 代表了span的开始的前一个token的表示
- $x_{e+1}$: 代表了span的结束的后一个token的表示
- $ p_i$: 代表了$x_i$的位置

  $y_i = f(x_{s-1}, x_{e+1}, p_i)$
其中$f(·)$是一个两层的feed-foreward的神经网络 with Gelu 和layer normalization
\begin{equation}
h = LayerNorm(GeLU(W_1[x_{s-1};x_{e+1};p_i]))\\
f(·) = LayerNorm(GeLU(W_2h)
\end{equation}


Loss：
\begin{equation}
Loss = L_{MLM} + L_{SBO}
\end{equation}

SBO：对于span的问题很有用例如 extractive question answering

### Single-Sentence training
为什么选择Single sentence而不是原本的两个sentence？
- 训练的长度更大
- 随机选择其他文本的输入进来作为训练增加了噪声
- Albert给出了原因，是因为NSP太简单了，只学会了top 的信息，没学会句子之间顺序SOP的信息。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/37d51aa8fc56b92d80065002d00fbcd6.png)

## ref
[SpanBERT zhihu](https://zhuanlan.zhihu.com/p/75893972)
[SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/pdf/1907.10529.pdf)