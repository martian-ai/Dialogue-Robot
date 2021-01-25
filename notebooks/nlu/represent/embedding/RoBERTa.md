
# RoBERTa
论文原文：[Roberta](https://arxiv.org/pdf/1907.11692.pdf)

[项目主页中文](https://github.com/brightmart/roberta_zh), 作者表示，在本项目中，没有实现 dynamic mask。
[英文项目主页](https://github.com/pytorch/fairseq)

从模型上来说，RoBERTa基本没有什么太大创新，主要是在BERT基础上做了几点调整：
1）训练时间更长，batch size更大，训练数据更多；
2）移除了next predict loss；
3）训练序列更长；
4）动态调整Masking机制。
5) Byte level BPE
RoBERTa is trained with dynamic masking (Section 4.1), FULL - SENTENCES without NSP loss (Section 4.2), large mini-batches (Section 4.3) and a larger byte-level BPE (Section 4.4).


## 更多训练数据/更大的batch size/训练更长时间
- 原本bert：BOOKCORPUS (Zhu et al., 2015) plus English W IKIPEDIA.(16G original)
  + add CC-NEWS(76G)
  + add OPEN WEB TEXT(38G)
  + add STORIES(31G)

- 更大的batch size
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/26b1fe81820b38e2633bfc96200188c0.png)
- 更长的训练时间
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/fe481510d79be5632e8ce742ca4dec9a.png)

### Dynamic Masking
- static masking: 原本的BERT采用的是static mask的方式，就是在`create pretraining data`中，先对数据进行提前的mask，为了充分利用数据，定义了`dupe_factor`，这样可以将训练数据复制`dupe_factor`份，然后同一条数据可以有不同的mask。注意这些数据不是全部都喂给同一个epoch，是不同的epoch，例如`dupe_factor=10`， `epoch=40`， 则每种mask的方式在训练中会被使用4次。
  > The original BERT implementation performed masking once during data preprocessing, resulting in a single static mask. To avoid using the same mask for each training instance in every epoch, training data was duplicated 10 times so that each sequence is masked in 10 different ways over the 40 epochs of training. Thus, each training sequence was seen with the same mask four times during training.
- dynamic masking: 每一次将训练example喂给模型的时候，才进行随机mask。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/dbf433fe8fbca51b79cb500dafd20b23.png)


## No NSP and Input Format
NSP: 0.5:从同一篇文章中连续的两个segment。0.5:不同的文章中的segment
- Segment+NSP：bert style
- Sentence pair+NSP：使用两个连续的句子+NSP。用更大的batch size
- Full-sentences：如果输入的最大长度为512，那么就是尽量选择512长度的连续句子。如果跨document了，就在中间加上一个特殊分隔符。无NSP。实验使用了这个，因为能够固定batch size的大小。
- Doc-sentences：和full-sentences一样，但是不跨document。无NSP。最优。
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/763c1ccf113e15665b1cdee0fbd643b9.png)

## Text Encoding
BERT原型使用的是 character-level BPE vocabulary of size 30K, RoBERTa使用了GPT2的 BPE 实现，使用的是byte而不是unicode characters作为subword的单位。
> learn a subword vocabulary of a modest size (50K units) that can still encode any input text without introducing any “unknown” tokens.

zh 实现没有dynamic masking
```Python
    instances = []
    raw_text_list_list=get_raw_instance(document, max_seq_length) # document即一整段话，包含多个句子。每个句子叫做segment.
    for j, raw_text_list in enumerate(raw_text_list_list): # 得到适合长度的segment
        ####################################################################################################################
        raw_text_list = get_new_segment(raw_text_list) # 结合分词的中文的whole mask设置即在需要的地方加上“##”
        # 1、设置token, segment_ids
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in raw_text_list:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        ################################################################################################################
        # 2、调用原有的方法
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
```

## ref
[RoBERTa 模型调用](https://mp.weixin.qq.com/s/K2zLEbWzDGtyOj7yceRdFQ)
[模型调用](https://mp.weixin.qq.com/s/v5wijUi9WgcQlr6Xwc-Pvw)
[知乎解释](https://zhuanlan.zhihu.com/p/75987226)

# structBERT
[StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding](https://arxiv.org/pdf/1908.04577)
StructBERT是阿里在BERT改进上面的一个实践，模型取得了很好的效果，仅次于ERNIE 2.0, 因为ERNIE2.0 采用的改进思路基本相同，都是在pretraining的时候，增加预训练的obejctives。

首先我们先看看一个下面英文和中文的两句话：
`i tinhk yuo undresatnd this sentneces.`

`研表究明，汉字序顺并不定一影阅响读。比如当你看完这句话后，才发这现里的字全是都乱的`

注意：上面的两个句子都是乱序的！

这个就是structBERT的改进思路的来源。对于一个人来说，字的或者character的顺序不应该是影响模型效果的因素，一个好的LM 模型，需要懂得自己纠错！

此外模型还在NSP的基础上，结合ALBERT的SOP，采用了三分类的方式，预测句子间的顺序。

这两个目标的加入，将句子内的结构word-level ordering，以及句子间sentence-level ordering的结构加入了BERT模型中，使得模型效果得到提升。

![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/df581b944aa1909d3134e3ff7dbdbdc6.png)

## Word Structural Objective
输入的句子首先先按照BERT一样mask15%，（80%MASK，10%UNMASK，10%random replacement）。

\begin{equation}
argmax_\theta \sum logP(pos_1 = t1, pos_2 = t_2, ..., pos_K= t_K | t_1, t_2, ..., t_K)
\end{equation}

给定subsequence of length K， 希望的结果是把sequence 恢复成正确顺序的likelihood最大。
- Larger K，模型必须要学会reconstruct 更多的干扰数据，任务比较难，但是噪声多
- Smaller K，模型必须要学会reconstruct 较少的干扰数据，可能这个任务就比较简单

论文中使用的是K=3，这个任务对单个句子的任务效果比较好。
## Sentence Structural Objective
一个句子pair $(S_1, S_2)$，执行三种任务
- 1/3的时候：$(S_1, S_2)$是上下句，分类为1
- 1/3的时候：$(S_2, S_1)$是上下句反序，分类为2
- 1/3的时候：$(S_1, S_{random})$是不同文档的句子，分类为3
这个任务对句子对的任务效果好。

## Ablation Studies
![](http://blog-picture-bed.oss-cn-beijing.aliyuncs.com/eb2271cd115c90cfcbd071404176410b.png)
前三个任务 single-sentence： 主要需要 word structure objective
后三个任务 sentence-pair ：主要需要 sentence structure objective