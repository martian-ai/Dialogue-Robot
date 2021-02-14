# Summary of Text Generation

# 文本生成方法
+ GPT
+ GAN
+ VAE
    + https://github.com/matthewvowels1/Awesome-VAEs

# 对话生成方法
+ 模版
+ 树
+ Plan-base NLG
+ Class-base LM
+ Phrase-based LM
+ Corpus-based
+ RNN-based LM
+ Semantic Conditioned LSTM
+ Structural NLG (句法树 + NN)
+ Contextual NLG 
+ Controlled Text Generation 
+ seq2seq
+ transformer
+ memory network
+ 用户建模
+ 强化学习
+ 检索和生成的结合


# 解决方案

# 基于 Seq2Seq 的方法
+ 针对闲聊数据的生成效果一般，主要存在安全回复，语义流畅性等问题
+ https://github.com/lc222/seq2seq_chatbot

# A Neural Conversational Model
+ Orio Vinyals， Google
+ 基于seq2seq 的单轮对话，使用两层lstm 进行建模
+ 预测时
    + 贪心
    + beam search
# Neural Responding Machine for Short-Text Conversation
+ ACL 2015 李航
+ 数据 ： 微博评论数据
+ 带有attention 的seq2seq 模型
![20200315210701-2020-3-15-21-7-1](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200315210701-2020-3-15-21-7-1)

# Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models(HRED)
+ 分层attention 机制
+ Encoder和Decoder RNN每个句子是一个RNN，而Context RNN只有一个RNN；
+ 在编码和解码阶段，处理每个句子的RNN结构完全相同而且共享参数（“The same encoder RNN and decoder RNN parameters are used for every utterance in a dialogue”），目的是为了生成一般化的语言模型；
+ 在解码阶段，每个时刻都会把Context RNN的输出与该时刻的输入并联一块作为输入，目的是是的每个解码时刻都可以引入上下文信息
![20200315210716-2020-3-15-21-7-17](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200315210716-2020-3-15-21-7-17)

# A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues(VHRED)
+ 加入变分编码的思想，在 context rnn 环节加入一个高斯随机变量，以便增加响应的多样性
![20200315210744-2020-3-15-21-7-45](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200315210744-2020-3-15-21-7-45)

# Attention with Intention for a Neural Network Conversation Model(AWI)
+ 讲attention 机制加入到分层seq2seq 中， 包括encoder RNN， intention RNN 和 Decoder RNN 三部分
+ 每个句子的Encoder RNN的初始化状态都是前面一个Decoder RNN的最后一个隐层状态，这样做是为了在encoder阶段把前面时刻的对话状态引入进去；
+ Encoder是输出可以使用最后一个时刻的隐层状态，也可以使用Attention机制对每个时刻隐层状态进行加权求和，然后作为Intention RNN的输入；
+ Intention RNN在每个时刻除了会考虑Encoder的输出还会结合前一时刻Decoder的输出；
+ Decoder RNN的初始化状态是该时刻Intention RNN的输出
![20200315210824-2020-3-15-21-8-24](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200315210824-2020-3-15-21-8-24)

# A Neural Network Approach to Context Sensitive Generation of Conversational Responses
+ 旨在训练一个data driven open-domain的多轮对话系统，在生成response的时候不仅仅考虑当前user message（query），而且考虑past history作为context
+ 后续有待补充

# Diversity-Promoting Objective Function for Neural Conversation Models
+ 李纪为 
+ 使用最大互信息MMI 训练seq2seq
+ MMI-antiLM
+ MMI-bidi

# Content-Introducing Approach to Generative Short-Text Conversation(seq2BF)
+ 生成短文本response，数据集是百度贴吧
+ 过程
    + 使用PMI 确定一个与query 最相关的名词
    + 使用seq2seq 依次生成名词之前的部分
    + 使用seq2seq 依次生成名词之后的部分
    + 拼接 前的部分 和 后的部分
+ 评测
    + 人工评测、生成回复长度以及熵作为评测指标
    + 其中人工评测采用了pointwise和pairwise两种方法
+ 效果
    + 使用百度贴吧进行评测，效果好与传统的seq2seq

# An Auto-Encoder Matching Model for Learning Utterance-Level Semantic Dependency in Dialogue Generation(AMM)
+ 为了解决在对话生成过程中的语义连贯性问题，提出了一种自动编码器匹配模型，其中包含两个自动编码器和一个映射模块
+ https://github.com/lancopku/AMM
​
# An Auto-Encoder Matching Model for Learning Utterance-Level Semantic Dependency in Dialogue Generation
​
# MEMORY NETWORKS
# Learning End-to-End Goal-Oriented Dialog
# Key-Value Memory Networks for Directly Reading Documents
# TRACKING THE WORLD STATE WITH RECURRENT ENTITY NETWORKS
# Evaluating Prerequisite Qualities for Learning End-to-End

# 用户模型相关
# Conversational Contextual Cues: The Case of Personalization and History for Response Ranking

# 强化学习相关
# Deep Reinforcement Learning for Dialogue Generation
# Sequence Generative Adversarial Nets with Policy Gradient
+ https://github.com/LantaoYu/SeqGAN

# GAN 相关
# Adversarial learning for neural dialogue generation
+ 李纪为 EMNLP2017
# Toward Controlled Generation of Text
+ https://arxiv.org/abs/1703.00955

# Neural Response Generation via GAN with an Approximate Embedding Layer
+ https://www.aclweb.org/anthology/D17-1065.pdf
+ https://github.com/lan2720/GAN-AEL
​

# Knowledge base 相关
# Incorporating Unstructured Textual Knowledge Sources into Neural Dialogue Systems
# Learning to Select Knowledge for Response Generation in Dialog Systems
​

# Towards a Human-like Open-Domain Chatbot(Meena)
+ 端到端模型，可以学习如何根据上下文做成响应，训练目标是减少困惑度以及下一个标记（在这种情况下为对话的下一个单词）的不确定性
+ 架构 ： Evolved Transformer seq2seq， 即通过神经架构发现的一种Transformer结构，可以改善困惑度
    + 有1个编码器和13个解码器构成
    + 编码器用于处理对话语境，理解已经说过的话的内容，解码器利用这些信息生成实际的回复
    + 经过超参数调整后，发现更强的解码器是实现高质量对话的关键
+ 语料 : Meena 根据七轮对话生成语境回复
    + 将每轮对话抽取作为训练样本，而该轮之前的 7 轮对话作为语境信息，构成一组数据
    + 据博客介绍，Meena 在 341GB 的文本上进行了训练，这些文本是从公共领域社交媒体对话上过滤得到的，和 GPT-2 相比，数据量是后者的 8.5 倍
+ 指标
    + 人类评价指标SSA
        + 基于人力的众包测试，从[Hi]开始，人类评价者评估 ‘对话是否讲的通' 和 '对话是否详细‘，结果如下， meena效果 接近人类

    + 困惑度
        实验发现，与SSA强相关
+ 不足
    + 只专注于敏感性和独特性，而其他属性如个性和真实性等依旧值得在后续的工作中加以考虑
    
https://arxiv.org/abs/2001.09977v2
​https://zhuanlan.zhihu.com/p/104383357

# Learn to ask
+ Learning to Ask Questions in Open-domain Conversational Systems with Typed Decoders
+ https://github.com/victorywys/Learning2Ask_TypedDecoder
+ https://arxiv.org/pdf/1805.04843.pdf
​

# 特定格式生成（Sentence Function）
+ 生成特定功能的句子（疑问句、陈述句、祈使句）

# Generating Informative Responses with Controlled Sentence Function
+ 引入条件变分自编码器，利用隐变量来建模和控制生成回复的功能特征
+ 同时，我们在模型中设计了类别控制器，解码回复中的每个词之前会先根据隐变量和当前解码状态预测待生成词所属的类别（即功能控制词、话题词或普通词）
+ 再根据类别信息解码出相应的词，使得功能特征和内容信息能够在生成的回复中有机结合。自动评测
​+ https://github.com/kepei1106/SentenceFunction

# 检索生成的方法
# Response Generation by Context-aware Prototype Editing
+ 吴俣 AAAI 2019
+ https://github.com/MarkWuNLP/ResponseEdit
+ https://arxiv.org/pdf/1806.07042v1.pdf
​

# Reference
+ cstghitpku：总结|对话系统中的自然语言生成技术（NLG）
+ 黄海兵：对话系统(Chatbot)论文串烧
+ super涵：对话生成模型总结（解读+开源代码）
+ 张俊：PaperWeekly第四期------基于强化学习的文本生成技术