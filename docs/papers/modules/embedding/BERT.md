# BERT

## Abstract

- 核心思想
  - 通过所有层的上下文来预训练深度双向的表示
- 应用
  - 预训练的BERT能够仅仅用一层output layer进行fine-turn, 就可以在许多下游任务上取得SOTA(start of the art) 的结果, 并不需要针对特殊任务进行特殊的调整

## Introduction

- 使用语言模型进行预训练可以提高许多NLP任务的性能
  - Dai and Le, 2015
  - Peters et al.2017, 2018
  - Radford et al., 2018
  - Howard and Ruder, 2018
- 提升的任务有
  - sentence-level tasks(predict the relationships between sentences)
    - natural language inference
      - Bowman et al., 2015
      - Williams et al., 2018
    - paraphrasing(释义)
      - Dolan and Brockett, 2005
  - token-level tasks(models are required to produce fine-grained output at token-level)
    - NER
      - Tjong Kim Sang and De Meulder, 2003
    - SQuAD question answering

### 预训练language representation 的两种策略

- feature based
  - ELMo(Peters et al., 2018) [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
    - use **task-specific** architecture that include pre-trained representations as additional features representation
    - use shallow concatenation of independently trained left-to-right and right-to-left LMs
- fine tuning
  - Generative Pre-trained Transformer(OpenAI GPT) [Improving Language Understanding by Generative Pre-Training](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf)
    - introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning the pre-trained parameters
    - left-to-right

### Contributions of this paper

- 解释了双向预训练对Language Representation的重要性
  - 使用 MLM 预训练 深度双向表示
  - 与ELMo区别
- 消除(eliminate)了 繁重的task-specific architecture 的工程量
  - BERT is the first fine-tuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many systems with task-specific architectures
  - extensive ablations
    - goo.gl/language/bert

## Related Work

- review the most popular approaches of pre-training general language represenattions
- Feature-based Appraoches
  - non-neural methods
    - pass
  - neural methods
    - pass
  - coarser granularities(更粗的粒度)
    - sentence embedding
    - paragrqph embedding
    - As with traditional word embeddings,these learned representations are also typically used as features in a downstream model.
  - ELMo
    - 使用biLM(双向语言模型) 建模
      - 单词的复杂特征
      - 单词的当前上下文中的表示
    - ELMo advances the state-of-the-art for several major NLP bench- marks (Peters et al., 2018) including question
      - answering (Rajpurkar et al., 2016) on SQuAD
      - sentiment analysis (Socher et al., 2013)
      - and named entity recognition (Tjong Kim Sang and De Meul- der, 2003).
- Fine tuning Approaches
  - 在LM进行迁移学习有个趋势是预训练一些关于LM objective 的 model architecture, 在进行有监督的fine-tuning 之前
  - The advantage of these approaches is that few parameters need to be learned from scratch
  - OpenAI GPT (Radford et al., 2018) achieved previously state-of-the-art results on many sentencelevel tasks from the GLUE benchmark (Wang et al., 2018).
- Transfer Learning from Supervised Data
  - 无监督训练的好处是可以使用无限制的数据
  - 有一些工作显示了transfer 对监督学习的改进
    - natural language inference (Conneau et al., 2017)
    - machine translation (McCann et al., 2017)
  - 在CV领域, transfer learning 对 预训练同样发挥了巨大作用
    - Deng et al.,2009; Yosinski et al., 2014

## Train Embedding

### Model Architecture

- [Transformer](https://github.com/Apollo2Mars/Algorithms-of-Artificial-Intelligence/blob/master/3-1-Deep-Learning/1-Transformer/README.md)
- BERT v.s. ELMo v.s. OpenGPT

  ![img](https://ws2.sinaimg.cn/large/006tKfTcly1g1ima1j4wjj30k004ydge.jpg)

### Input

- WordPiece Embedding

  - WordPiece是指将单词划分成一组有限的公共子词单元，能在单词的有效性和字符的灵活性之间取得一个折中的平衡，例如下图中‘playing’被拆分成了‘play’和‘ing’
  - 中文的的情况就是字的
- Position Embedding

  - **讲单词的位置信息编码成特征向量，是learned position embedding， 不是transformer中的正弦position embedding**
- Segment Embedding

  - **用于区别两个句子，例如B是否是A的下文(对话场景，问答场景)，对于句子对，第一个句子的特征值是0，第二个句子的特征值是1**
- 三个embedding 相加

  - 相加的理解 https://www.zhihu.com/question/374835153

![img](https://ws4.sinaimg.cn/large/006tNc79ly1g2ql45wou8j30k005ydgg.jpg)

### Loss

- Multi-task Learning

## Use Bert for Downstream Task

- Sentence Pair Classification
- Single Sentence Classification Task
- Question Answering Task
- Single Sentence Tagging Task
