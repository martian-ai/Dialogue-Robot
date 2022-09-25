### [R-NET ACL 2017 MSRA](<https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf>)

![](https://ws3.sinaimg.cn/large/006tKfTcly1g1pcrgzkffj318p0tctc0.jpg)

- Abstract
  - 借鉴 Match-LSTM 和 Pointer Network的思想
- Framework
  - Question and Passage Encoder
    - Word Embedding(Glove) + Char Embedding 
    - Question Encoder $$u^Q_t$$
    - Passage Encoder $$u_t^Q$$
  - Gated Attention-Based Recurrent Network
  - Self-Matching Attention
  - Output Layer