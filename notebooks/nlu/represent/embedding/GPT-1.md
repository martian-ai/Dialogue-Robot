
# GPT-1
[GPT-1](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)又称为openAI transformer，使用的是transformer的decoder的结构（不包含encoder和decoder的attention部分），用的是auto-regressive的LM objective。

GPT最大的共享是，提出了pretraining和finetuning的下游统一框架，将预训练和finetune的结构进行了统一，解决了之前两者分离的使用的不确定性，例如elmo。使用transformer结构解决了LSTM的不能捕获远距离信息的缺点。但是其的主要缺点是不能使用双向的信息。

## Unsupervised pretraining
模型使用的是auto-regressive LM obejctive
\begin{equation}
\begin{split}
h_0 &= UW_e + W_p  \\
h_l &= transformer(h_{l-1}) \ \ \forall i\in [1,n]\\
P(u) &= softmax(h_n W_e^T) \\
L_1(U) &= -\sum_i log P (u_i | u{i-k},...,u_{i-1};\Theta)
\end{split}
\tag{$1$}
\end{equation}
- k 是contex的窗口size
- n 是transformer layer的个数
- $h_n$ 是context下的hidden 输出
- $W_e$ 是embedding matrix
- $W_p$ 是position matrix
- $U = {u_1, u_2, u_3, u_4, ..., u_m}$ 是输入的sequence

# supervised finetuning

对于输入的序列$x_1, x_2, ..., x_m$, 以及label $y$, 输入先喂到预训练的模型中得到最后一层的输出$h_n ^m$，在连接全连接层with parameters $W_y$， 去预测y：
> The inputs are passed through our pre-trained model to obtain the ﬁnal transformer block’s activation $h_l^m$, which is then fed into an added linear output layer with parameters W_yto predict y:

\begin{equation}
\begin{split}
P(y|x_1,...,x_m) &= softmax(h_l^m W_y) \\
L_2(C) &= \sum_{(x,y)} log P(y|x_1,...,x_m)
\end{split}
\tag{$2$}
\end{equation}

 $h_l^m$ 是最后一个token作为clf_token, see [code](https://github.com/huggingface/pytorch-openai-transformer-lm/blob/bfd8e0989c684b79b800a49f8d9b74e559298ec2/train.py)
```Python
encoder['_start_'] = len(encoder)
encoder['_delimiter_'] = len(encoder)
encoder['_classify_'] = len(encoder)
clf_token = encoder['_classify_'] <----最后一个token
```

在finetuning的时候，在特定任务的loss的基础上，还加入了LM的loss作为auxiliary loss，使得模型得到更好的结果
```Python
clf_logits, clf_losses, lm_losses = model(*xs, train=True, reuse=do_reuse)
          if lm_coef > 0:
              train_loss = tf.reduce_mean(clf_losses) + lm_coef*tf.reduce_mean(lm_losses)
          else:
              train_loss = tf.reduce_mean(clf_losses)
```
对于不同任务有不同的任务构造方式：
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/2ea51119f69c10822e320e2e85f5f5bf.png)

所有输入都增加`(<s> <e>)`tokens
- classification
- entailment
- similarity：因为是Text1和Text2的顺序无关，所以两个方向的，文本之间通过$分割，最后的dense层通过的是两个transform 儿输出的和作为输入。
- multiple choice：bert 没有这种（[ref to](https://github.com/huggingface/transformers/pull/96)，但是构造和这个一样。Context=document+query； Text2=answer</s>
具体的输入形式：`[z;q$a_k]`,其中$为分隔符， 三个输出再经过soft Max。[RACE]data set

## Ablation Studies
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/98d3495c7d994dcc07c14d420c088310.png)
- transformer 比LSTM 好
- aux LM loss对NLI以及QQP效果有帮助，（2sentences）