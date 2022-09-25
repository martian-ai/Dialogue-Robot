# DSSM
+ \<Learning Deep Structured Semantic Models for Web Search using Clickthrough Data\>
+ **文章提出，PSLA LDA等模型都是非监督模型，和目标不直接挂钩，效果自然比不上监督模型**
+ 模型结构
  + word hashing 层
    + 英文处理
      + 每个词加上一个开始和结束的符号 (e.g. #good#).然后切分成n-grams 特征(e.g. letter trigrams: #go, goo, ood, od#).当然，更常见的做法是做一个预训练的word embedding。Word hashing的好处是模型是真正的end-to-end结构，不用额外维护一个word embedding lookup table，对于输入可以进行简单地映射。
      + 举个例子，假设用 letter-trigams 来切分单词（3 个字母为一组，#表示开始和结束符），boy 这个单词会被切为 #-b-o, b-o-y, o-y-#
      + ![20210814153346](https://i.loli.net/2021/08/14/yAkQVervMNqFOBL.png) 
      + 这样做的好处有两个：首先是压缩空间，50 万个词的 one-hot 向量空间可以通过 letter-trigram 压缩为一个 3 万维的向量空间。其次是增强范化能力，三个字母的表达往往能代表英文中的前缀和后缀，而前缀后缀往往具有通用的语义。
      + 这里之所以用 3 个字母的切分粒度，是综合考虑了向量空间和单词冲突：
      + ![20210814153529](https://i.loli.net/2021/08/14/Zs3bezDNhQ92E1W.png)
      + 以 50 万个单词的词库为例，2 个字母的切分粒度的单词冲突为 1192（冲突的定义：至少有两个单词的 letter-bigram 向量完全相同），而 3 个字母的单词冲突降为 22 效果很好，且转化后的向量空间 3 万维不是很大，综合考虑选择 3 个字母的切分粒度。
    
    + 中文处理
      +  中文的输入层处理方式与英文有很大不同，首先中文分词是个让所有 NLP 从业者头疼的事情，即便业界号称能做到 95%左右的分词准确性，但分词结果极为不可控，往往会在分词阶段引入误差。所以这里我们不分词，而是仿照英文的处理方式，对应到中文的最小粒度就是单字了。（曾经有人用偏旁部首切的，感兴趣的朋友可以试试）
      +  由于常用的单字为 1.5 万左右，而常用的双字大约到百万级别了，所以这里出于向量空间的考虑，采用字向量（one-hot）作为输入，向量空间约为 1.5 万维。
    + 表示/FC层：若干全连接层
      + ![20210814152823](https://i.loli.net/2021/08/14/bqw1VKQtWTmvonU.png)
      + 用 Wi 表示第 i 层的权值矩阵，bi 表示第 i 层的 bias 项。则第一隐层向量 l1（300 维），第 i 个隐层向量 li（300 维），输出向量 y（128 维）可以分别表示为
        + $l_{1}=W_{1} x$
$l_{i}=f\left(W_{i} l_{i-1}+b_{i}\right), i=2, \ldots, N-1$
$y=f\left(W_{N} l_{N-1}+b_{N}\right)$
      + 用 tanh 作为隐层和输出层的激活函数
        + $f(x)=\frac{1-e^{-2 x}}{1+e^{-2 x}}$    
    + 匹配层/Softmax层：这里算出每个文档D和查询词Q的语义向量y后，求Q和每个文档D的余弦相似度，然后接softmax层求损失函数值。
      + Query 和 Doc 的语义相似性可以用这两个语义向量(128 维) 的 cosine 距离来表示：
        + $R(Q, D)=\operatorname{cosine}\left(y_{Q}, y_{D}\right)=\frac{y_{Q}^{T} y_{D}}{\left\|y_{Q}\right\|\left\|y_{D}\right\|}$
      + 通过softmax 函数可以把Query 与正样本 Doc 的语义相似性转化为一个后验概率
        +  $P\left(D^{+} \mid Q\right)=\frac{\exp \left(\gamma R\left(Q, D^{+}\right)\right)}{\sum_{D, \in D} \exp \left(\gamma R\left(Q, D^{\prime}\right)\right)}$
        +  其中 r 为 softmax 的平滑因子，D 为 Query 下的正样本，D-为 Query 下的负样本（采取随机负采样），D 为 Query 下的整个样本空间
      + 在训练阶段，通过极大似然估计，我们最小化损失函数：
        + $L(\Lambda)=-\log \prod_{\left(Q, D^{+}\right)} P\left(D^{+} \mid Q\right)$ 
        + 残差会在表示层的 DNN 中反向传播，最终通过随机梯度下降（SGD）使模型收敛，得到各网络层的参数{Wi,bi} 
    + 采样：正样本D+和随机采样的四个不相关文档。文章指出，未发现不同采样方式对模型性能的影响。
+ 优点：DSSM 用字向量作为输入既可以减少切词的依赖，又可以提高模型的泛化能力，因为每个汉字所能表达的语义是可以复用的。另一方面，传统的输入层是用 Embedding 的方式（如 Word2Vec 的词向量）或者主题模型的方式（如 LDA 的主题向量）来直接做词的映射，再把各个词的向量累加或者拼接起来，由于 Word2Vec 和 LDA 都是无监督的训练，这样会给整个模型引入误差，DSSM 采用统一的有监督训练，不需要在中间过程做无监督模型的映射，因此精准度会比较高。

+ 缺点：上文提到 DSSM 采用词袋模型（BOW），因此丧失了语序信息和上下文信息。另一方面，DSSM 采用弱监督、端到端的模型，预测结果不可控。

# Reference
+ https://www.cnblogs.com/wmx24/p/10157154.html