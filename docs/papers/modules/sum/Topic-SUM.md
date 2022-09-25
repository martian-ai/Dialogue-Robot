# Topic-Oriented Spoken Dialogue Summarization for Customer Service with Saliency-Aware Topic Modeling

## Abstract
+ 在客户服务系统中，对话摘要可以通过自动为长对话创建摘要来提高服务效率，在对话中，客户和代理尝试解决特定主题的问题。在这项工作中，我们关注面向主题的对话摘要，它生成高度抽象的摘要，从对话中保留主要思想。在口语对话中，大量的对话噪音和常见的语义会模糊潜在的信息内容，使得一般的主题建模方法难以应用。此外，对于客户服务，特定角色的信息很重要，是总结中不可或缺的一部分。为了有效地对对话进行主题建模并捕获多角色信息，本文提出了一种新的主题增强两阶段对话摘要器（TDS）和显著性感知神经主题模型（SATM），用于面向主题的客户服务对话总结。通过对真实中国客户服务数据集的综合研究，证明了我们的方法相对于几个强基线的优越性。

## Introduction
+ 在主动客户服务系统中，客户和代理之间实时生成大量的对话，传递重要信息。在这样的背景下，如何有效地利用对话信息成为一个不容忽视的问题。对话摘要是一项任务，其目的是在保留显著信息的同时浓缩对话（Rambow等人2004；Pan等人2018；Shang等人2018；Liu等人2019a），通过自动创建简明摘要以避免耗时的对话阅读和理解，从而提高服务效率。
+ 大多数现有的对话总结工作主要集中在长而复杂的口语对话上，如会议和法庭辩论，这些对话通常通过串接所有对话点来总结，以保持完整的对话流程（Gillick et al.2009；Shang et al.2018；Duan et al.2019b）。然而，在客户服务场景中，对话演讲者通常有强烈而明确的动机，旨在解决与特定主题相关的问题（Wang等人，2020年）。为了更好地理解客户和代理人的意图，在这项工作中，我们重点关注面向主题的对话摘要，其目的是提取语义一致的主题并生成高度抽象的摘要，以维护对话中的主要思想。
+ 最近，引入了数十种主题感知模型，以协助完成文档摘要任务（Wang等人，2018年；Narayan等人，2018年；Fu等人，2020年）。然而，口语对话通常是由话语组成的，而不是传统文献中结构良好的句子。显著信息在这些话语中被稀释，并伴随着共同的语义。此外，噪音以不相关的聊天和转录错误的形式大量存在（Tixier等人，2017年）。这类常见或嘈杂的词，如请、谢谢和哼唱，通常频率较高，并与其他信息性词语同时出现。因此，一般的基于主题的方法很难从统计上区分有用和无用内容的混合，从而导致对主题分布的不准确估计（Li等人，2018年，2019b）。此外，在客户服务对话中，参与角色是稳定的：客户倾向于提出问题，代理人需要提供解决方案。图1显示了一个真实的客户服务对话以及一个摘要，其中包括两位发言人提供的关键信息。因此，该模型还希望捕获角色信息，以协助显著性估计。

![20220129000116](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220129000116.png)

+ 在这项工作中，我们提出了一种新的两阶段神经模型和一种增强的主题建模方法，用于对话摘要。首先，为了更好地区分潜在的信息内容与丰富的常见语义和对话噪音，我们引入了显著性感知主题模型（SATM），其中主题分为两组：有效信息主题和其他主题。在主题建模的生成过程中，我们将与标准摘要相对应的每个显著性单词限制为从有效信息主题生成，而对话中的其他单词（包括嘈杂和常见单词）仅从其他主题生成。通过此训练过程，SATM可以在对话中将每个单词与显著性（有效信息主题）或非显著性（其他主题）关联起来。其次，为了获取角色信息并从对话中提取语义主题，我们使用SATM分别对客户话语、代理话语和整体对话执行多角色主题建模。然后，设计了一个主题增强的两阶段对话摘要器（TDS），它由一个话语提取器和一个抽象提炼器组成。它可以通过话题信息注意机制在话语层面和词汇层面上提取与话题相关的显著信息。
+ 此外，由于缺乏合适的公共基准，我们收集了一个具有高度抽象摘要的真实客户服务对话数据集。在所提出的数据集上的实验结果表明，我们的模型在各种指标下的性能优于一系列强基线。
+ 代码、数据集和补充数据可在Github上找到。 https://github.com/RowitZou/topic-dialog-summ
+ 主要贡献
    + 1） 我们提出了一种新的主题模型，该模型通过直接学习单词显著性对应关系来感知对话中潜在的信息内容。
    + 2） 在多角色主题模型的基础上，我们提出了一个主题增强的两阶段模型，该模型采用主题通知注意机制来进行显著性评估和总结客户服务对话。
    + 3） 在收集的数据集上的实验结果证明了我们的方法在不同方面的有效性。

## Model
+ 在本节中，我们将详细介绍显著性感知主题模型（SATM）和主题增强的两阶段对话总结器（TDS）。SATM根据信息主题和其他主题推断多角色主题表示。然后，通过主题通知注意机制将主题信息合并到TDS的提取器和细化器中。我们模型的总体架构如图2所示。

#### Basic NTM with Variational Inference
+ 符号定义
    + $V$ 词典大小
    + $H$ 隐向量维度
    + $K$ 主题个数
    + $d$ 对话的词袋表示(去停用词)
    + $\theta \in \mathbb{R}^{K}$ 对话d的主题分布
    + $p(\theta|d)$ 后验概率 
    + $q(\theta|d)$ 推理网络
    + $\beta \in \mathbb{R}^{K \times|V|} $ 主题-词 分布
    + $\phi \in \mathbb{R}^{K \times H} $ 主题向量， 随机初始化
    + $e \in \mathbb{R}^{|V| \times H }$  词向量， 预训练的结果
    + $t \in \mathbb{H} $ 当前对话的主题表示

+ 主题推理
    + 输入:给出了一个对话的词袋表示(去除停用词) $d \in R^{|V|}$
    + 处理过程
        + NTM 图

        ![20220129000148](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220129000148.png)

        + 构造隐变量$z$
            + 隐变量$z$采样自对角高斯分布$z \sim \mathcal{N}\left(\mu(d), \sigma^{2}(d)\right)$，其中$\mu(d)$和$\sigma(d)$是神经网络
        + 在实践中，可以通过 $\hat{z}=\mu(d)+\epsilon \cdot \sigma(d)$ 使用重新参数化技巧（Kingma和Welling 2014）对$\hat z$进行采样，其中ε被采样自$N(0，I^2）$，对于神经网络的训练，ε 为常量
        + $q(θ|d)$由一个函数 $θ=f(z)$组成，一个采样$\hat θ \in R^K $的推导公式为：$\hat{\theta}=f(\hat{z})=\operatorname{softmax}\left(W_{\theta} \hat{z}+b_{\theta}\right)$, $W_{\theta}$ 和 $b_{\theta}$ 是训练参数
            + $q(θ|d) = \hat{\theta}=f(\hat{z})=\operatorname{softmax}\left(W_{\theta} \hat{z}+b_{\theta}\right)$
        + 这样就构建一个推理网络$q(θ|d)$ 来近似后验$p(θ|d)$
    + 输出:当前对话的主题分布情况$q(θ|d)$
+ 文档生成
    + $\beta$ 通过 $\beta_{k}=\operatorname{softmax}\left(e \cdot \phi_{k}^{\top}\right)$ 得到
    + 已知当前对话的主题分布 $q(θ|d)$ 和主题-词分布 $\beta$ 就可以重建对话 $d$

+ 损失函数
    + $\mathcal{L}_{T}=D_{K L}[q(\theta \mid d) \| p(\theta)]-\mathbb{E}_{q(\theta \mid d)}[\log p(d \mid \beta, \theta)] \approx D_{K L}[q(z \mid d) \| p(z)]-\sum_{n} \log p\left(w_{n} \mid \beta, \hat{\theta}\right)$
    + $w_n$ 表示在对话d中第n个可以观察到的词 
+ 当前对话的主题表示
    + $t=\phi^{\top} \cdot \hat{\theta} $

### Saliency-Aware Neural Topic Model
+ 我们提出的SATM基于具有变分推理的神经主题模型（NTM）（Miao et al.2017），该模型通过神经网络从每个对话d推断主题分布θ。我们用一种新的生成策略扩展了NTM来学习单词显著性对应。与NTM相比，SATM的结构如下图所示。

![20220129000251](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220129000251.png)

+ 符号定义
    + $s$ $d$ 的子集，表示有效信息的词
    + $o$ $d$ 的子集，表示无效信息的词
    + $K_{s}$ 含有有用信息的主题数量
    + $K_{o}$ 含有无效信息的主题数量
    + $\beta_{s} \in \mathbb{R}^{K \times|V|} $ 有效主题-词 分布
    + $\phi_{s} \in \mathbb{R}^{K \times H} $ 有效主题向量， 随机初始化
    + $\beta_{o} \in \mathbb{R}^{K \times|V|} $ 无效主题-词 分布
    + $\phi_{o} \in \mathbb{R}^{K \times H} $ 无效主题向量， 随机初始化
+ 主题推理
    + 输入:给出了一个对话的词袋表示(去除停用词) $d \in R^{|V|}$
    + 处理过程
        + 将K个主题分为两组 $K_s, K_o$
        + $\hat{\theta}_{s}=f_{s}(\hat{z})=\operatorname{softmax}\left(W_{\theta_{s}} \hat{z}+b_{\theta_{s}}\right)$
        + $\hat{\theta}_{o}=f_{o}(\hat{z})=\operatorname{softmax}\left(W_{\theta_{o}} \hat{z}+b_{\theta_{o}}\right)$
        + 概率函数表示
            + $p(d \mid \beta, \theta)=p\left(s \mid \beta_{s}, \theta_{s}\right) p\left(d-s \mid \beta_{o}, \theta_{o}\right)$
+ 损失函数
    + $\begin{aligned} \mathcal{L}_{T} \approx &-\sum_{n} \log p\left(w_{n}^{s} \mid \beta_{s}, \hat{\theta}_{s}\right)-\sum_{n} \log p\left(w_{n}^{d-s} \mid \beta_{o}, \hat{\theta}_{o}\right) + D_{K L}[q(z \mid d) \| p(z)] . \end{aligned}$

+ 对话主题表示
    + $t_s = \phi_s^{\top} \cdot \hat{\theta_s} $
    + $t_o = \phi_o^{\top} \cdot \hat{\theta_o} $

### Topic Information Augmentation
+ 使用Transformer 作为基础的编码和解码器，为了捕获角色信息并突出全局主题，通过一种主题通知注意机制将多角色主题表示纳入Transformer解码器，该机制用于显著性估计，是多头注意的扩展
+ 符号定义
    + $q_i$ 在解码阶段第i个query 词
    + $x_j$ 在memory 中 第j 个元素
    + $\tau_{s}^A$ 客服的有效主题表示
    + $\tau_{o}^A$ 客服的无效主题表示
    + $\tau_{s}^C$ 客户的有效主题表示
    + $\tau_{o}^C$ 客户的无效主题表示
    + $W_{Q}$ 映射函数
    + $W_{K}$ 映射函数
    + $W_{V}$ 映射函数
    + $W_{T}$ 映射函数
    + $W_{P}$ 映射函数
+ 原始的transformer 定义为
    + $\alpha_{i j}^{q}=\operatorname{softmax}\left(\left(q_{i} W_{Q}\right)\left(x_{j} W_{K}^{q}\right)^{\top} / \sqrt{d_{h}}\right)$
    + $\mu_{i}^{q}=\sum_{j} \alpha_{i j}^{q}\left(x_{j} W_{V}\right)$
+ 改造后的transformer 定义为
    + 如果 $x_j$ 与客户话术相关
        + $\tau_{s}=\left[t_{s} ; t_{s}^{C} ; \mathbf{0}\right], \tau_{o}=\left[t_{o} ; t_{o}^{C} ; \mathbf{0}\right]$
    + 如果 $x_j$ 与客服话术相关
        + $\tau_{s}=\left[t_{s} ; \mathbf{0} ; t_{s}^{A}\right], \tau_{o}=\left[t_{o} ; \mathbf{0} ; t_{o}^{A}\right]$
    + 受对比注意力机制的启发，设计了使得$\tau_{s}$ 和 $\tau_{o}$ 产生相反作用的 $a_{j}^{t}$
        + $\alpha_{j}^{t}=\operatorname{softmax}\left(\left(\tau_{s} W_{T}-\tau_{o} W_{T}\right)\left(x_{j} W_{K}^{t}\right)^{\top} / \sqrt{d_{h}}\right)$
    + $\mu_t$ 可以看作是一个主题感知向量，基本上丢弃了内存中嘈杂和无信息的元素
        + $\mu^{t}=\sum \alpha_{j}^{t}\left(x_{j} W_{V}\right)$

    + $p_{i}^{s e l} \in(0,1)$表示使用的选择概率作为一种软切换，可以在原始的注意力机制或者基于主题增强的注意力机制进行选择
    + 将上述两个注意机制结合起来，得到每个解码步骤中考虑主题信息和原始注意力信息的$\mu_i$ 。
        + $p_{i}^{s e l}=\sigma\left(\left[q_{i} ; \mu_{i}^{q} ; \mu^{t}\right] \cdot W_{P}\right)$
        + $\alpha_{i j}=\left(1-p_{i}^{s e l}\right) \cdot \alpha_{i j}^{q}+p_{i}^{s e l} \cdot \alpha_{j}^{t}$
        + $\mu_{i}=\sum_{j} \alpha_{i j}\left(x_{j} W_{V}\right)$


### 联合训练
+ 符号定义
    + $\lambda$ 用于平衡摘要模型和主题模型
    + ${L}_{T}^{C}$ 用户主题模型的损失函数
    + ${L}_{T}^{A}$ 客服主题模型的损失函数
    + ${L}_{T}$ 整体主题模型的损失函数
    + ${L}_{S}$ 摘要主题模型的损失函数

+ 联合训练的损失函数为
    + $\mathcal{L}=\mathcal{L}_{S}+\lambda\left(\mathcal{L}_{T}^{C}+\mathcal{L}_{T}^{A}+\mathcal{L}_{T}\right)$

# Reference
+ Kingma, D. P.; and Welling, M. 2014. Auto-Encoding Vari- ational Bayes. CoRR abs/1312.6114.
+ Duan, X.; Zhang, Y.; Yuan, L.; Zhou, X.; Liu, X.; Wang, T.; Wang, R.; Zhang, Q.; Sun, C.; and Wu, F. 2019b. Legal Sum- marization for Multi-role Debate Dialogue via Controversy Focus Mining and Multi-task Learning. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, 1361–1370.


## 代码分析

+ encoder
    + bert
    + Transformer
+ sent_encoder
+ hier_encoder
+ pn_decoder
+ decoder
+ pn_generator
+ generator
+ copy_generator
+ topic_model
