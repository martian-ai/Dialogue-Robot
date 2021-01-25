### [BiDAF-ICLR2017](https://arxiv.org/pdf/1611.01603.pdf)

![](https://ws1.sinaimg.cn/large/006tNc79ly1g1t3uklag5j31ak0t8gs0.jpg)

- Abstract

  - 同时计算contex2query 和 query2context, 注意力的计算在时序上是独立的,并会flow到下一层
  - 避免了过早的summary 造成的信息丢失

- Framework

  - char embedding:使用CNN训练得到,参考Kim(有待添加)

  - word embedding:Glove

  - 字向量和词向量拼接后,过一个Highway, 得到　Context X 和 query Y

  - Contextual Embedding Layer

    - 使用Bi-LSTM整合Highway的输出，表达词间关系
    - 输出
      - contex $ H \in R^{2d*T} $
      - query $ Q \in R^{d*J} $

  - Attention Flow Layer

    - 计算context和query词和词两两之间的相似性 
      $$ S_{tj} = \alpha(H_{:t}, U_{:j}) $$
      $$ \alpha = w^T_{S} [h;u;h \odot u] $$		

    - 计算context-to-query attention, 对于context中的词,按attention系数计算query中的词的 加权和 作为当前词的 **query aware representation**

      $$\alpha_t = softmax(St:) \in R^J$$

      $$ \widetilde U_{:t} = \sum \alpha_{ij} U_{:j} R\in^{2d*J} $$

    - 计算query-to-context attention, 计算 query 和 每个 context 的最大相似度, query和context的相似度是query所有词里面和context相似度最大的, 然后计算context 的加权和

      $$ b = softmax(max_{col}(S)) $$
      $$ \widetilde{h} = \sum_t b_t H_{:t}  \in R^{2d}$$
      $$ \widetilde{H} = tile(\widetilde{h})  $$	

    - final query-aware-representation of context

      $$ G_{:t} = \beta(H_{:t}, \widetilde U_{:t}, \widetilde H_{:t} ) $$

      $$ \beta(h;\widetilde{u};\widetilde{h}) = [h;\widetilde{u};h\odot\widetilde{u};h\odot\widetilde{h}] \in R^{8d}$$	

  - Modeling Layer

    - 过Bi-LSTM 得到 M

  - Output Layer

    $$ p^1 = softmax(w^T_(p1)[G;M]$$

    $$ p^2 = softmax(w^T_(p1)[G;M_2]$$

    $$ L(\theta) = - \frac{1}{N} \sum_i^{N} log(p^1_{y^1_i}) + log(p^2_{y^2_i})$$

  - results

    - SQuAD

    ![SQuAD](https://ws2.sinaimg.cn/large/006tNc79ly1g1w7p64a07j30k006e0tm.jpg)

    - CNN/DailyMail

    ![CNN/Dialy Mail](https://ws3.sinaimg.cn/large/006tNc79ly1g1w7q415szj30k00asdh8.jpg)


+ 原文和源码
  + https://arxiv.org/abs/1611.01603
  + https://github.com/allenai/bi-att-flow
+ 思路
![20200330194903](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/20200330194903.png)
  + 传统的阅读理解方法中将段落表示成固定长度的vector，而BIDAF将段落vector流动起来，利用双向注意力流建模段落和问题之间的关系，得到一个问题感知的段落表征（即段落vector的值有段落本身和当前问题决定，不同的问题会产生不同的vector）
  + 每一个时刻， 仅仅对当前 问题 和段落进行计算，并不依赖上一时刻的attention，使得后边的attetion不受之前错误attention的影响(有待细化）
  + 计算C2Q(context2query) 和 Q2C(query2context), 认为两者相互促进
+ Demo : 基于问题感知的段落表征
  + {context, query} = {汉堡好吃, 吃啥呢} = {H, U}
  + 对应的embedding 为， 向量维度为3(为了便于演示，每个字的向量的元素都相同）
  ![20200330194949](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/20200330194949.png)
  + 计算相似矩阵 H^T * U 
  ![20200330195002](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/20200330195002.png)
  + C2Q ( 对query  进行编码）
  [ 0.36*吃 + 0.42*的 + 0.48*呢,  
    0.54*吃 + 0.63*的 + 0.72*呢,
    0.72*吃 + 0.84*的 + 0.96*呢
    0.9*吃 + 1.05*的 + 1.2*呢]
  + Q2C ( 对context 进行编码）
    + 取每一行的最大值 [0.48, 0.72, 0.96, 1.2] 作为权重，对原始query 做加权和， 并重复T次
    [0.48*汉 + 0.72*堡 + 0.96*好 + 1.2*吃] * T 
  + 拼接 H， C2Q， Q2C ， 并进行一定映射 得到 基于当前query 和 context 的 段落表示 G
  + G在通过LSTM得到M，使用G和M经过MLP 预测起始位置和终止位置 
+ 实验结果
  ![20200330195115](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/20200330195115.png)