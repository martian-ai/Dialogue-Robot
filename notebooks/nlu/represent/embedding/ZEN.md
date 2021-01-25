
# ZEN
[ZEN](https://arxiv.org/abs/1911.00720) 提出一个基于中文BERT模型的强化，它的创新点在于不改变原本bert模型的做法之下，增加了ngram的强化，从而得到了几乎媲美ERNIE2.0的效果，可以说这个是一个非常大的强化了。具[repo](https://github.com/sinovation/ZEN)体见

## Why ZEN
- 像ERNIE的mask的方式只能依赖于已有的词，句的信息。
- 以前的mask的方式，在pre-training 和 fine-tuning阶段存在mismatch，即fine-tuning阶段没有保存词，句的信息，但是ZEN的词句的信息在finetuning阶段也保存了
- 错误的NER或者mask会影响encoder的效果。

此外，发现在利用N-gram enhancement，可以在数据量很少的情况下，得到更好的结果，从此证明了这个策略的有效性。

## What is ZEN
ZEN 主要分为两个步骤，一个是N-gram Extraction，一个是N-gram Integration（包括 N-gram encoding和N-gram representation）

虽然模型使用了N-gram的信息，但是模型的输出还是跟原本BERT一样是character level的encoding。

### N-gram Extraction
- Lexicon：在训练之前，需要先进行Ngram extraction，这就是把语料库里所有的N-gram提取出来，所谓的N-gram就是词组。然后设置阈值，按照频次倒序排序，去掉频次低于阈值的N-gram。注意这边的N-gram，可以是包含关系，例如里面同时存在，`港澳`和`粤港澳`。对于这个Lexicon，不考虑单个词的N-gram。

- N-gram Matching Matrix：然后对于每一条输入的训练数据，长度为$k_c$，共匹配到了$k_n$个N-gram，我们创一个N-gram Matching Matrix 形状为$k_c*k_n$的$M$矩阵，矩阵中的元素为：

  \begin{equation}
  m_{i,j}=\left\{
  \begin{aligned}
  1 &  & if\ c_i \in n_j\\
  0 &  & otherwise    
  \end{aligned}
  \right.
  \end{equation}

  其中： $c_i$: 第$i$个 character， $n_j$：第$j$个N-gram
  例如输入是： 粤港澳大湾区城市竞争力强....和交通一体化
  我们提取出来的N-gram有{一体化...，港澳，粤港澳大湾区}
  M：
  ![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/4c8cd28373fb58328de265da64e2c48d.png)

### Encoding N-gram
对于N-gram的encoding，模型使用的是transformer的encoding，只是因为我们的N-gram的顺序并不需要考虑，所以模型没有使用position encoding。
TODO：这边论文并没有讲清楚它的输入，
- 是不是需要segment encoding？
- 怎么做embedding，是将ngram matching matrix做映射吗（映射到的embedding的dim肯定是hidden size的长度）还是是提取出了ngram但是还是照样用embedding lookup，只是这边词的顺序不一致
- 这个layer的size是和character encoder一样吗？
都没有讲清楚，需要看以下源码。

这个 N-gram encoder的作用是，判断目前的Ngram和其他的Ngram的关联性。所以其Q，K，V分别为
- Q：当前要query的N-gram的embedding，formally，j-th n-gram in layer l by $µ_j^{(l)}$
- K：所有的N-gram的embedding 的stack结果, formally, $U^{(l)} = [µ_0^{(l)};µ_1^{(l)};...;µ_{k_n}^{(l)}]$ 其中$k_n$是ngram的个数
- V：所有的N-gram的embedding 的stack结果即$U^{(l)}$
$\mu_j^{(l)} = MhA(Q=\mu _j^{(l)}, K=V=U^{(l)}) $

 MhA 是 multi-head attention

### Representing N-grams in Pre-training
模型在对ngram进行encoding之后，如何进行结合的？这个要就是直接进行vector的相加。
\begin{equation}
v_i^{(l)*} = v_i^{(l)} + \sum _k u_{i,k}^{(l)}
\end{equation}

其中
- $v_i^{(l)}$ 代表的是character encoder的第l层输出的第i个character的hidden output。
- $u_{i,k}^{(l)}$ 代表的是N-gram。 encoder，在包含第i个character的第k个N-gram在第l层输出的hidden output。例如上面图中的`澳`，与之相关的Ngram有两个分别是`港澳`和`粤港澳大湾区`
如果一个字没有存在在任何一个N-gram中，则$v_i^{(l)*} = v_i^{(l)} $保持不变

  \begin{equation}
  v^{(l)*} = V^{(l)} + M * U^{(l)}
  \end{equation}
  这边的$M$是我们的N-gram Matching Matrix。
需要注意的是，如果这个词在character encoder 被mask之后，那么就不将N-gram encoding进行相加。

### Performance
- 效果
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/20eecec0afd5c0619c24fd2477714003.png)
效果达到了和ERNIE的媲美，而且主要是它使用character encoder是最原始的，所以大有可为。
  - P：用BERT预训练初始化，R：随机初始化。可以看出初始化很重要
  - ZEN在token level的任务上：NER，POS和Chinese word segmentation (CWS), 任务上有更大的提升空间。
  - B：base， L：large

- 收敛速度
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/42dff6109726fcfbb58d052a5a5fc823.png)
Albert也很快收敛
- 可视化
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/ddf12ba58a15d71b4e267101725f54e7.png)
通过热度图，还通过实验分析了两个案例，将n-gram encoder的注意力机制可视化出来。

通过热度图可以清晰地看到，注意力会更多的关注在有效的n-gram。比如“波士顿”的权重明显高于“士顿”。对于有划分歧义的句子，n-gram encoder可以正确的关注到“速度”而不是“高速”。

更加有趣的是，在不同层次的encoder关注的n-gram也不同。更高层的encoder对于“提高速度”和“波士顿咨询”这样更长的有效n-gram分配了更多的权重。

这表明，结合n-gram的方法的预训练，不仅仅提供给文本编码器更强大的文本表征能力，甚至还间接产生了一种文本分析的有效方法。这个案例分析暗示我们，或许将来可以用类似地方法提供无指导的文本抽取和挖掘

- 小数据集

除了以上实验，该研究还探究了模型在小数据集上的潜力。

考虑到目前的预训练模型使用了大型的训练语料，但是对于很多特殊的领域，大型数据集很难获取。

因此本文抽出1/10的中文维基百科语料，来模拟了一种语料有限的场景，目的是探究ZEN在小数据集上的潜力。

实验结果如下图所示，在全部七个任务上，ZEN都明显优于BERT。这表明ZEN在数据有限的场景下，具有更大的潜力。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/fad4848b4c888e352f5d09932cd21c6d.png)