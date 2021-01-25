
# UniLM
[Uniﬁed Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/pdf/1905.03197.pdf)

本文提出了采用BERT的模型，使用三种特殊的Mask的预训练目标，从而使得模型可以用于NLG，同时在NLU任务获得和BERT一样的效果。
模型使用了三种语言模型的任务：
- unidirectional prediction
- bidirectional prediction
- seuqnece-to-sequence prediction


![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/2a9759231d687d84837323bb10a42d32.png)
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/9656e64fac234093e645304953c95cdf.png)

## Unidirectional LM
$x_1x_2\ [MASK]\ x_4$ 对于MASK的预测，只能使用token1和token2以及自己位置能够被使用，使用的就是一个对角矩阵的。同理从右到左的LM也类似。

## Bidirectional LM
对于双向的LM，只对padding进行mask。

## Seq2Seq LM
在训练的时候，一个序列由`[SOS]S_1[EOS]S_2[EOS]`组成，其中S1是source segments，S2是target segments。随机mask两个segment其中的词，其中如果masked是source segment的词的话，则它可以attend to 所有的source segment的tokens，如果masked的是target segment，则模型只能attend to 所有的source tokens 以及target segment 中当前词（包含）和该词左边的所有tokens。这样的话，模型可以隐形地学习到一个双向的encoder和单向decoder。（类似transformer）

也可以理解为: 输入两句。第一句采用BiLM的编码方式，第二句采用单向LM的方式。同时训练encoder(BiLM)和decoder(Uni-LM)。处理输入时同样也是随机mask掉一些token。

## 实现细节
- Span mask
- 总的loss 是三种LM的loss之和
- 我们在一个训练的batch中，1/3的时间训练bidirection LM，1/3的时间训练sequence-to-sequence LM objective， 1/6的时间训练left-to-right 和 1/6的时间训练 right-to-left LM

## Finetune
- 对于NLU的任务，就和BERT一样进行finetune。
- 对于NLG的任务，S1:source segment， S2: target segment， 则输入为“[SOS] S1 [EOS] S2 [EOS]”. 我们和预训练的时候一样也是随机mask一些span，目标是在给定的context下最大化我们的mask的token的概率。值得注意的是[EOS], which marks the end of the target sequence,也是可以被masked，因为这样可以让模型学习到什么时候生成[EOS]这样可以标志文本生成的结束。
  - abstractive summarization
  - question generation
  - generative question answering
  - dialog response generation)
  使用了label smooth和 beam search


很不错的论文，但是没有ablation studies， 也有很多需要改进的方向，但是已经很不错了，我对NLG认识不多，但是学到了不少，里面很多引用我都需要学习一下（TODO）
代码： [repo](<https://github.com/microsoft/unilm>)

![](../../../../Desktop/PPT/Uni-LM.jpg)