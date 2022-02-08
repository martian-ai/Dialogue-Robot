### [Matching-LSTM](https://arxiv.org/pdf/1608.07905.pdf)

![M-LSTM](https://ws2.sinaimg.cn/large/006tNc79ly1g1t3urvg0fj318k0p4ajq.jpg)

- Abstract

  - LSTM 编码原文的上下文信息
  - Match-LSTM 匹配原文和问题
  - Answer-Pointer :　使用Ptr网络, 预测答案
    - Sequence Model :　答案是不连续的
    - Boundary Model : 答案是连续的, 在SQuAD数据集上 Boundary 比 Sequence 的效果要好

- Framework

  - LSTM Preprocessing Layer

    - 使用embedding表示Question 和 Context, 在使用单向LSTM编码,得到hidden state 表示 , l 是隐变量长度

      $$ H^P = \vec{LSTM}(P) \in R^{l *P}$$

      $$H^Q = \vec {LSTM}(Q) \in R^{l *Q}$$

  - Match-LSTM Layer

    - 类似文本蕴含:前提H, 假设T, M-LSTM序列化的经过假设的每一个词,然后预测前提是否继承自假设
    - 文本问答中, question 当做 H, context当做T, 可以看成带着问题去段落中找答案(利用soft-attention)

  - Answer Pointer Layer

    - Sequence Model
    - Boundary Model

- Results
  ![](https://img.mukewang.com/5ac37472000179ad15620702.png)