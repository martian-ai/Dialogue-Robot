# Multi-Turn Retrieval Chitchat

## [Multi-view Response Selection for Human-computer Conversation](https://www.aclweb.org/anthology/D16-1036.pdf)
+ 百度自然语言处理部 EMNLP2016
+ 提供了一种单轮转多轮的思路
+ 将多轮问答语句合并为一句，连接处用 _SOS_ 隔开，将整个对话历史视为一句话去匹配下一句
+ 使用 TextCNN + pooling + GRU 提取特征
+ 损失函数 使用 disagreement-loss 和 likelihood-loss， 并利用了不同程度的交互信息

+ 模型结果
![20200624184243-2020-6-24-18-42-44](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200624184243-2020-6-24-18-42-44)

+ loss 函数设计
![20200624185502-2020-6-24-18-55-2](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200624185502-2020-6-24-18-55-2)

+ ubuntu语料上结果为 
![20200624185625-2020-6-24-18-56-26](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200624185625-2020-6-24-18-56-26)


## Sequential Matching Network A New Architecture for Multi-turn Response Selection in Retrieval-Base Chatbots(SMN)
+ ACL 2017
+ https://github.com/MarkWuNLP/MultiTurnResponseSelection
+ https://arxiv.org/abs/1612.01627

+ utterance 和 response 做word embedding 表示，使用GRU 做整句信息的表示
+ word-matching 
    + 每一个 utterance 和 response 做 word embedding， 再用 dot(e_i, e_j)  计算词粒度的匹配矩阵 M1， 大小为 utterance 长度 * response 长度
+ utterance-matching
    + 每一个 utterance 和 response 做 word embedding，分别通过GRU(源码中均通过sentence_GRU) 得到 h_i  和 h_j ，然后再用 dot(h_i, A*h_j) ，得到词粒度的匹配矩阵 M2(A 为一个有待学习的变量）， 大小为 utterance 长度 * response 长度
+ 将M1，M2 视为CNN 中的双通道进行卷积和池化，之后进行全连接，之后二分类，判断respone 和对话历史的匹配程度

![20200315204536-2020-3-15-20-45-36](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200315204536-2020-3-15-20-45-36)
​
+ Ubuntu 和 豆瓣 结果为
![20200624191509-2020-6-24-19-15-10](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200624191509-2020-6-24-19-15-10)


# Modeling Multi-turn Conversation with Deep Utterance Aggregation(DUA)
+ COLING 2018
+ Multi-View 和 SMN 都是 将对话历史中的各个utterance的权重认为是平等的，但是实际对话中，对话历史的各个句子所包含的信息是不同的，所有有必要对每个utterance 赋予不同的权重

# Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network(DAM)
+ ACL 2018 百度自然语言处理部
+ 各种attention的组合
+ 多层交互

# Learning Matching Models with Weak  Supervision for Response Selection for Retrieval-based Chatbots
+ ACL 2018 SMN 作者
# Multi-Representation Fusion Network for Multi-turn Response Selection
+ https://github.com/chongyangtao/MRFN
+ 构造多种类型的信息表示， 这些表示对context 和 response 在 word，n-gram， utterance 子序列等不同粒度信息上进行语义编码，并且捕获单词之间的短期和长期依赖关系， 信息表示包括一下几类：
+ 词表示
    + char-level word embedding + CNN 解决OOV 问题
    + 原始论文中使用常规的word embedding
+ 短依赖
    + 使用GRU获取序列信息
    + 使用CNN 获取N-Gram 信息
+ 长依赖
    + Multi-head attention 使用多个self-attention 的融合来获取不同空间上的信息
    + Cross attention 获取utterance 和response之间的交互信息
+ 使用深度网络对以上信息表示进行匹配，作者提出了一种多表示融合的方法进行信息匹配，模型结构如下
    + 符号表示
        u_i : 第 i 条utterance
        r : response
        U_i^k : 第i个utterance 的第k种信息表示
        R^k : response 的 第k中信息表示
        U^*_i :  utterance 所有信息拼接的结果
        R^* : response 所有信息拼接的结果
        T^*_i ：utterance i 和 response 交互后的信息表示
        v_i : utterance i 的匹配向量
        \hat e_{i,j} ,  \hat e_{r,j} :  U_i^*  和 R^* 第 j  个词的信息表示
+ Fusing at an early stage（FES）
    + 将utterance 和 response 的K中信息拼接起来 得到 U_i^*  和 R^* 
    + 将 U_i^* 和 R^* 进行点乘后映射得到 T_i^* , 映射过程如下
    + 点乘加映射
        w_{j,k}^i = V_a^T tanh(W_a[\hat e_{i,j} \oplus \hat e_{r,k} ] + b_a）  
    + 数据归一化 
        a_{j,k}^i = \frac{exp(w_{j,k}^i)}{ \sum_{k=1}  exp(w_{j,k}^i)} 
    + 使用SUBMULT + NN 得到t_{i,j}， 方法见https://arxiv.org/pdf/1611.01747.pdf
    + 每个utterance 经过 GRU_t 得到一个匹配向量
    + 整个context 再经过 GRU_v 得到 h_m^v 
    + 再经过一个MLP得到最终的匹配分数
+ Fusing at an intermediate stage(FIS)
    + 和FES不同的是，FIS的utterance和response的交互在每个类型的表示上进行，得到K个交互表示T^{k}_{i}，拼接之后得到T^{*}_{i}
+ Fusion at an the last stage(FLS)
    + FLS将(c,r)的匹配分为K个pipeline，其中每个pipeline将一种类型的表示作为输入，并输出g^{k}(c,r)。每一种表示的U^{k}_{i}和R^{k }计算得到T^{k}_{i}，接一个GRU_{t}网络得到v^{k}_{i}，整个context接一个GRU_{v}网络和MLP得到g^{k}(c,r)
    
![20200315204636-2020-3-15-20-46-36](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200315204636-2020-3-15-20-46-36)
​
# Interactive Matching Network for Multi-Turn Response Selection
+ 受ELMo网络的影响，作者使用了多层RNN建模多个utterances，每一层RNN对应一个utterances，并创新性的加入了attention机制学习每一层的weighting（意味着每句utterance的重要性不同），作者将这个网络模块命名为attentive hierarchical recurrent encoder

![20200315204710-2020-3-15-20-47-10](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200315204710-2020-3-15-20-47-10)


+ https://arxiv.org/pdf/1901.01824.pdf
​+ https://github.com/JasonForJoy/IMN