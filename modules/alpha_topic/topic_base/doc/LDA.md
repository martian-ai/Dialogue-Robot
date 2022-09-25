#  LDA
  
## Part I : Introduction
+ 隐含狄利克雷分布（英语：Latent Dirichlet allocation，简称LDA），是一种主题模型，**可以将文档集中每篇文档的主题按照概率分布的形式给出**，是一种无监督学习算法，在训练时不需要手工标注的训练集，需要的仅仅是文档集以及指定主题的数量k即可。LDA首先由 David M. Blei、吴恩达和迈克尔·I·乔丹于2003年提出[1]，目前在文本挖掘领域包括文本主题识别、文本分类以及文本相似度计算方面都有应用。它是一种典型的词袋模型，即一篇文档是由一组词构成，词与词之间没有先后顺序的关系。一篇文档可以包含多个主题，文档中每一个词都由其中的一个主题生成。
+ 文档的生成过程可以简单参考下方例子
  + 事先给定四个主题 Arts、Budgets、Children、Education，**通过学习训练，获取每个主题对应的词语**，如下图

  ![20141117153816148](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20141117153816148.png)
  + 以一定概率选取一个主题，再以一定概率选取主题下的词，最终生成如下图所示的文章（其中不同颜色的词语分别对应上图中不同主题下的词）
  
  ![20141117154035285](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20141117154035285.png)
+ LDA 的数学模型较为复杂，一篇文档的生成过程如下：
  + 从参数为$\alpha$ 狄利克雷分布中采样生成文档i的主题分布$\theta_m$, **即每一篇文档的主题分布不是固定的，是从狄利克雷分布中采样出来的**
  + 从主题的多项式分布$\theta_m$中取样生成文档m第n个词的主题$z_{m,n}$, 对应上边的例子，$z_{m,n} \in $ {Arts, Budgets, Children, Education}, 比如认为$z_{m,n}$ 为Arts
  + 从参数为$\beta$狄利克雷分布中取样生成主题$z_{m,n}$对应的词语分布$\phi_{z_{m,n}}$
  + 从词语的多项式分布$\phi_{z_{m,n}}$中采样最终生成词语$w_{m,n}$
  
  <img src="https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20210627101554.png" width = "400" height = "300" alt="图片名称" align=center />

  + 上图中共会产生 M 个 文档的主题分布， K 个 词语的主题分布
  + 向量$\alpha$ 的长度是 主题的个数 K， 向量$\beta$ 的长度都是词典个数W

## Part II : Related Math

+ pass

## Part III : Model Structure
+ 根据贝叶斯理论的模式，先验分布 $\pi(\theta)+$ 样本信息 $\chi \Rightarrow$ 后验分布 $\pi(\theta \mid x)$， 意味着新观察到的样本信息将修正之前的认知(先验分布)得到新的认知(后验分布)
+ Unigram model(没有主题，只用一个狄利克雷分布来决定每个词输出的概率)
  + unigram model假设文本中的词服从Multinomial分布
  + 对于文档$w=(w_1, w_2, ..., w_N)$, 用$p(w_n)$表示词$w_n$的先验概率，则生成文档w的概率为 $p(\mathbf{w})=\prod_{n=1}^{N} p\left(w_{n}\right)$
  + 其图模型为(图中被涂色的w表示可观测变量，N表示一篇文档中总共N个单词，M表示M篇文档)

      ![20210627102006](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20210627102006.png)
  
  + 或者表示为
  
      <img src="https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20210627102053.png" width = "300" height = "200" alt="图片名称" align=center />

  + 上图中的表示在文本中观察到的第n个词，n∈[1,N]表示该文本中一共有N个单词。加上方框表示重复，即一共有N个这样的随机变量。其中，p和α是隐含未知变量：
    + p是词服从的Multinomial分布的参数
    + α是Dirichlet分布（即Multinomial分布的先验分布）的参数。
  + 一般α由经验事先给定，**p由观察到的文本中出现的词学习得到，表示文本中出现每个词的概率**
+ Mixture of unigrams model（一篇文档一个主题，在进行文档生成前会确定文档的主题）
  + 给某个文档先选择一个主题，再根据该主题生成文档，该文档中的所有词都来自一个主题。假设主题有$z_1, z_2, ..., z_k$，生成文档的概率为 $p(\mathbf{w})=p\left(z_{1}\right) \prod_{n=1}^{N} p\left(w_{n} \mid z_{1}\right)+\cdots+p\left(z_{k}\right) \prod_{n=1}^{N} p\left(w_{n} \mid z_{k}\right)=\sum_{z} p(z) \prod_{n=1}^{N} p\left(w_{n} \mid z\right)$
  + 其图模型为（图中被涂色的w表示可观测变量，未被涂色的z表示未知的隐变量，N表示一篇文档中总共N个单词，M表示M篇文档）
  + ![20210627102230](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20210627102230.png)
+ pLSA 模型(一片文档有多个主题)
  + 对于一篇文档，文档的主题分布是固定的，每个主题对应的词的分布也是固定的
  + 对于一片文档的所有词，依次在固定的主题分布中选择一个主题，根据选择到的主题分布对应的词分布来选择词。对于一篇文档中的所有词依次进行以上操作，直到完成文档生成。
  + 在这个过程中，并未关注词和词之间的出现顺序，所以pLSA是一种词袋方法
  + 相关概率定义如下：
    + $P(d_i)$表示海量文档中某篇文档被选中的概率。
    + $P(w_j|d_i)$表示词在给定文档中$d_i$出现的概率。
      + 怎么计算得到呢？针对海量文档，对所有文档进行分词后，得到一个词汇列表，这样每篇文档就是一个词语的集合。对于每个词语，用它在文档中出现的次数除以文档中词语总的数目便是它在文档中出现的概率$P(w_j|d_i)$。
    + $P(z_k|d_i)$表示具体某个主题在给定文档下出现的概率。
    + $P(w_j|z_k)$表示具体某个词在给定主题下出现的概率，与主题关系越密切的词，其条件概率越大。
  + 根据以上概率定义，PLSA的生成过程可以表示为
    + 按照概率$P(d_i)$选择一篇文档
    + 选定文档$d_i$后，从主题分布中按照概率$P(z_k|d_i)$选择一个隐含的主题类别$z_k$
    + 选定$z_k$后，从词分布中按照概率$P(w_j|z_k)$选择一个词$w_j$
  + 根据文档反推主题(生成文档的主题分布)
    + 文档和单词是可以被观察到的，主题是隐藏的
    + 如下图所示（图中被涂色的d、w表示可观测变量，未被涂色的z表示未知的隐变量，N表示一篇文档中总共N个单词，M表示M篇文档）
    
        ![20210627102157](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20210627102157.png)
    
    + 上图中，文档d和词w是我们得到的样本（样本随机，参数虽未知但固定，所以pLSA属于频率派思想。区别于下文要介绍的LDA中：样本固定，参数未知但不固定，是个随机变量，服从一定的分布，所以LDA属于贝叶斯派思想），可观测得到，所以对于任意一篇文档，其$P(w_j|d_i)$是已知的
    + 根据已知的文档-词 信息 $P(w_j|d_i)$, 训练出文档-主题$P(z_k|d_i)$ 和 主题-词项 $P(w_j|z_k)$, 计算过程如下：
      + 概率公式展开
        + $P\left(w_{j} \mid d_{i}\right)=\sum_{k=1}^{K} P\left(w_{j} \mid z_{k}\right) P\left(z_{k} \mid d_{i}\right)$
      + 文档的每个词生成的概率
        + $\begin{aligned} P\left(d_{i}, w_{j}\right) &=P\left(d_{i}\right) P\left(w_{j} \mid d_{i}\right) =P\left(d_{i}\right) \sum_{k=1}^{K} P\left(w_{j} \mid z_{k}\right) P\left(z_{k} \mid d_{i}\right) \end{aligned}$
        + 由于$P(d_i)$可以事先算出来(每个文档出现的概率)，而文档-主题$P(z_k|d_i)$ 和 主题-词项 $P(w_j|z_k)$ 两种概率是未知的，所以$\theta=\left(P\left(w_{j} \mid z_{k}\right), P\left(z_{k} \mid d_{i}\right)\right)$就是要估计的参数值
      + **常见的参数估计方法包括最大思然估计MLE，最大后验估计MAP，贝叶斯估计等，因为带估计变量中有隐变量z，所以使用EM算法估计** 
        + 使用EM算法推理 $\theta=\left(P\left(w_{j} \mid z_{k}\right), P\left(z_{k} \mid d_{i}\right)\right)$的数值
        + 用$\phi_{k,k}$表示词项$w_j$出现在主题$z_k$中的概率，即$P(w_j|z_k)=\phi_{k,j}$，用$\theta_{i,k}$表示主题$z_k$出现在文档$d_i$中的概率，即$P(z_k|d_i)=\theta_{i,k}$，从而把转换$P(w_j|z_k)$成了“主题-词项”矩阵Φ（主题生成词），把转换成了“文档-主题”矩阵Θ（文档生成主题）
        + 最终求得 $\phi_{i,k}$ 和 $\theta_{i,k}$
        + **求解PLSA 可以使用著名的 EM 算法进行求得局部最优解，可以参考 Hoffman 的原始论文，或者李航的《统计学习方法》，此处略去不讲**
  
+ LDA 模型
  + **与PLSA的比较**
    + LDA 是在 pLSA 的基础上加了一层贝叶斯框架，即LDA就是pLSA的贝叶斯版本（正因为LDA被贝叶斯化了，所以才需要考虑历史先验知识，才加的两个先验参数）
    + 在pLSA模型中，我们按照如下的步骤得到“文档-词项”的生成模型：
      + 按照概率$P(d_i)$选择一篇文档$d_i$
      + 选定文档$d_i$后，确定文章的主题分布
      + 从主题分布中按照概率$P(z_k|d_i)$选择一个隐含的主题类别$z_k$
      + 选定$z_k$后，确定主题下的词分布
      + 从词分布中按照概率$P(w_j|z_k)$选择一个词$w_j$
    + 下面，咱们对比下本文开头所述的LDA模型中一篇文档生成的方式是怎样的：
      + 按照先验概率$P(d_i)$选择一篇文档$d_i$
      + 从狄利克雷分布（即Dirichlet分布）中取样生成文档$d_i$的主题分布$\theta_i$，换言之，主题分布$\theta_i$由超参数为$\alpha$的Dirichlet分布生成
      + 从主题的多项式分布$\theta_i$中取样生成文档$d_i$第j个词的主题$z_{i,j}$
      + 从狄利克雷分布（即Dirichlet分布）$\beta$中取样生成主题$z_{i,j}$对应的词语分布$\phi_{z_{i,j}}$，换言之，词语分布$\phi_{z_{i,j}}$由参数为的Dirichlet分布生成
      + 从词语的多项式分布$\phi_{z_{i,j}}$中采样最终生成词语$w_{i,j}$
    + 从上面两个过程可以看出，LDA在PLSA的基础上，为主题分布和词分布分别加了两个Dirichlet先验。
     
  + LDA 参数估计(根据文档推断参数)
    + **在pLSA 和 原始LDA[<sup>1</sup>](#refer-anchor) 中，使用的是变分EM的方式来进行参数估计，后来有一种更好的参数估计方法就是Gibbs Sampling**
      
      <img src="https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20210627101554.png" width = "400" height = "300" alt="图片名称" align=center />

    + 给定一个文档，w是可以观察的已知变量，$\alpha$ 和 $\beta$ 是根据经验给定的先验参数，其他的变量$z$,$\theta$,$\phi$都是未知变量，需要根据观测的变量来进行规划
    + 根据LDA的图模型，可以写出所有变量的联合分布可以写为 ：$p\left(\vec{w}_{m}, \vec{z}_{m}, \vec{\theta}_{m}, \vec{\phi} \mid \vec{\alpha}, \vec{\beta}\right)=\prod^{N_{m}}_{n=1} p\left(w_{m, n} \mid \vec{\phi}_{z_{m, n}}\right) p\left(z_{m, n} \mid \vec{\theta}_{m}\right) \cdot p\left(\vec{\theta}_{m} \mid \vec{\alpha}\right) \cdot p(\vec{\phi} \mid \vec{\beta})$
        + $N_m$ 表示文档m 的单词个数
    + 因为$\alpha$产生主题分布$\theta$，主题分布$\theta$确定具体主题，且$\beta$产生词分布$\phi$、词分布$\phi$确定具体词，所以上述式子等价于下述式子所表达的联合概率分布
      + $p(\vec{w}, \vec{z} \mid \vec{\alpha}, \vec{\beta})=p(\vec{w} \mid \vec{z}, \vec{\beta}) p(\vec{z} \mid \vec{\alpha})$
    + 针对上式中的第一项$p(\vec{w} \mid \vec{z}, \vec{\beta})$表示根据主题$\vec{z}$ 和 词的先验分布 $\vec{\beta}$ 采样词的过程，第二项$p(\vec{z} \mid \vec{\alpha})$是根据主题分布的先验参数$\alpha$采用主题的过程，**这两个因子是需要计算的未知数**

    + **首先计算第一个式子**
      + $p(\vec{w} \mid \vec{z}, \vec \beta)=\prod_{i=1}^{W} p\left(w_{i} \mid z_{i}\right)=\prod_{i=1}^{W} \varphi_{z_{i}, w_{i}}$
        + W 为依次
      + 由于样本中的词服从参数为主题$z_i$的独立多项分布，这意味着可以把上面对词的乘积分解成分别对主题和对词的两层乘积：
        + $p(\vec{w} \mid \vec{z}, \Phi)=\prod_{k-1}^{K} \prod_{\left\{i: z_{i}-k\right\}} p\left(w_{i}=t \mid z_{i}=k\right)=\prod_{k=1}^{K} \prod_{t-1}^{V} \varphi_{k, t}^{n_{k}^{(t)}}$
        + k 为主题个数，V 为词典大小

    + 结合第一个式子和第二个式子的结果，得到$p(w,z)$的联合分布结果为
      + $p(\vec{z}, \vec{w} \mid \vec{\alpha}, \vec{\beta})=\prod_{z-1}^{K} \frac{\Delta\left(\vec{n}_{z}+\bar{\beta}\right)}{\Delta(\vec{\beta})} \cdot \prod_{m-1}^{M} \frac{\Delta\left(\vec{n}_{m}+\vec{\alpha}\right)}{\Delta(\vec{\alpha})}$
    + 有了联合分布，便可以通过联合分布来计算在给定可观测变量 w 下的隐变量 z 的条件分布（后验分布）$p(\vec{z} \mid \vec{w}) $来进行贝叶斯分析



<div id="refer-anchor"></div>

## References  

[1]  Blei, David M.; Ng, Andrew Y.; Jordan, Michael I. Lafferty, John , 编. Latent Dirichlet allocation. Journal of Machine Learning Research. January 2003, 3 (4–5): pp. 993–1022 [2013-07-08].  

[2] LDA 数学八卦 https://zhuanlan.zhihu.com/p/57418059   

[3] [wiki 狄利克雷分布](https://zh.wikipedia.org/wiki/%E9%9A%90%E5%90%AB%E7%8B%84%E5%88%A9%E5%85%8B%E9%9B%B7%E5%88%86%E5%B8%83) 

[4] 如何找到好的主题模型量化评价指标? https://zhuanlan.zhihu.com/p/105226228
