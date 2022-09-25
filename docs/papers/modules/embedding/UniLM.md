
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

## Pretrain
- Span mask
- 总的loss 是三种LM的loss之和 ( mask-word, nsp for _ in [ Unidirectional LM， Bidirectional LM， Seq2Seq LM ])
- 我们在一个训练的batch中，1/3的时间训练bidirection LM，1/3的时间训练sequence-to-sequence LM objective， 1/6的时间训练left-to-right 和 1/6的时间训练 right-to-left LM

## Finetune
- 对于NLU的任务，就和BERT一样进行finetune。
- 对于NLG的任务，S1:source segment， S2: target segment， 则输入为“[SOS] S1 [EOS] S2 [EOS]”. 我们和预训练的时候一样也是随机mask一些span，目标是在给定的context下最大化我们的mask的token的概率。
  - 值得注意的是[EOS], which marks the end of the target sequence,也是可以被masked，因为这样可以让模型学习到什么时候生成[EOS]这样可以标志文本生成的结束。
  - 相关NLG 任务
    - abstractive summarization
    - question generation
    - generative question answering
    - dialog response generation)
    - 使用了label smooth和 beam search

代码： [repo](<https://github.com/microsoft/unilm>)


本文UniLM既可以做Encoder和Decoder，也可以做Seq2seq。我们只要调整mask的位置 UniLM用的是self-attention，其很重要的是控制 self-attention 的范围。 BERT主要用于 NLU 任务，而 UniLM 可用于文本理解 NLU 和生成 NLG 这两项任务，因为可使用不同的自注意掩码进行配置，从而聚合用于不同类型的语言模型的上下文。
https://zhuanlan.zhihu.com/p/338593231

中文开源版 https://zhuanlan.zhihu.com/p/163483660




Training Data
简体中文维基百科数据，处理成一行一行句子对的形式。

Input Mask And Attention Mask
在一条数据中随机mask15%的token，被mask的token中80%用[MASK]表示，10%从vocab中随机选择一个token，10%不变。e.g. 一条可能的数据如下：[CLS] A1 A2 [MASK] A4 [SEP] B1 B2 B3 [MASK] B5 [SEP]，其中A1-A4是句子1的token，B1-B5是句子2的token，A3和B4被mask。
根据1中masked input的结果，生成不同的attention mask，unilm原文中说有1/3的数据采用seq2seq式的attention mask策略，1/3的数据采用bert式的attention mask策略，1/6数据采用left2right的language model式的attention mask，1/6数据采用right2left的language model式的attention mask。seq2seq其实就是对应的casual with prefix attention mask(下图，其他token在这里不可以看到被mask位置的符号):

提醒一下，序列到序列语言模型，在pretrain过程中，mask token的选取，是可以出现在第一个文本序列中，也可以出现在第二个文本序列中；但是到了fine-tuning阶段时，mask token仅出现在第二个文本序列中，因为我们需要通过第一个文本生成第二个文本。
https://zhuanlan.zhihu.com/p/113380840

作者对于不同的语言模型使用了不同句子段落向量（code中显示，双向对应0和1；单向left-right对应2；单向right-left对应3；序列对应4和5）。

Experiment
我们做了四个下游任务，分别是论文标题生成(csl)，webqa，微博新闻摘要和相似问句生成，其中前三个任务参考CLUEbencmark/CLGE 前三个任务中，我们对比了载入google原版bert权重和我们预训练的unilm权重，结果如下

csl(bleu,rouge-L)	webqa(bleu,rouge-L)	微博新闻标题生成(bleu,rouge-L)	相似问句生成(bleu)
Unilm-base	0.476, 0.648	0.358, 0.708	0.108, 0.265	
Bert-base	0.452, 0.640	0.342, 0.676	0.110, 0.267	


https://github.com/zhongerqiandan/pretrained-unilm-Chinese


微调
我们主要在做摘要任务，微调阶段只采用S2S LM进行微调即可。 此时第一句话是 article， 第二句话是 summary，并且只mask summary中的词，mask比例是75%。
https://zhuanlan.zhihu.com/p/112391971

