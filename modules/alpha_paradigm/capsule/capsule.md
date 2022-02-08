# Summary of Capsule Network

## Reference

+ https://zhuanlan.zhihu.com/p/33244896
+ https://zhuanlan.zhihu.com/p/33556066
+ Max Pechyonkin的胶囊网络入门系列：
  + [胶囊网络背后的直觉](http://link.zhihu.com/?target=https%3A//www.jqr.com/news/0088]38)

  + [胶囊如何工作](http://link.zhihu.com/?target=https%3A//www.jqr.com/news/008883)

  + [囊间动态路由算法](http://link.zhihu.com/?target=https%3A//www.jqr.com/news/009031)

  + [胶囊网络架构](http://link.zhihu.com/?target=https%3A//www.jqr.com/article/000040)



## Tips

+ Capsule 的革命在于：**它提出了一种新的“vector** **in** **vector out”的传递方案，并且这种方案在很大程度上是可解释的**

+ Demo

  ![](https://ws4.sinaimg.cn/large/006tKfTcly1g1ihxmp4z7j307e02owea.jpg)

  + 对于测试样本“帆船”，CNN的“三角形”和“四边形”的feature map都会被激活，即该图片中包含了三角形和四边形，就认为这是一个房子。所以说，CNN仅仅考虑了“有没有”的问题，没有考虑feature map的结构关系。这个结构关系包括位置，角度等等

+ Capsule 如何解决这个问题

  + Capsule layer的定义是a nested set of neural layers，即他是多层神经网络的一个集合，而非一个个孤零零的feature map

    ![](https://pic2.zhimg.com/80/v2-42daa6b0a0ae9b8a07fd270854bc4fd9_hd.jpg)

  + 结合房子和帆船的例子，模型中间Primary capsules，我们可以理解为某个capsule表征“房子”，某个capsule表征“帆船”。假设一个“帆船”测试样本输入到网络中，那么Primary capsules中对应“帆船”的capsule会被激活，而对应“房子”的capsule会被抑制，整个模型也就避免了将“帆船”错误地识别为“房子”。某种程度上说，Capsule layer提升了整个模型的表达能力，它比feature maps提取了更多的细节信息

## Framework

### What is Capsule?

+ Hinton 《Transforming Autoencoders》
+ CNN 中使用max pooling 保持神经网络的输出不变性，即稍微调整输入，由于最大池化，输出结果不变
+ 最大池化的缺陷
  + 损失了有价值的信息
  + 没有编码特征空间之间的特征关系
+ 胶囊：所有胶囊检测中的特征的状态的重要信息，都将以向量的形式被胶囊封装
  + 长度：检测出特征的概率 (有多大概率是房子)
  + 方向：检测出特征的状态(房子朝向，房子的组件间的空间结构)

### How Capsule work？

![](https://ws3.sinaimg.cn/large/006tKfTcly1g1ihyj0plfj30m80d2753.jpg)

+ 输入向量的矩阵乘法
+ 输入向量的标量加权
+ 加权输入向量之和
+ 向量到向量的非线性变换

![](https://ws1.sinaimg.cn/large/006tKfTcly1g1iin9a7w3j30rs0dudmw.jpg)

### Dynamic Route

+ 底层胶囊将输出发送给对此表示”同意”的高层胶囊

+ 算法原理

  ![](https://ws1.sinaimg.cn/large/006tKfTcly1g1ii0a1qzgj30rs086gmd.jpg)

  + 输入 : 第l层的capsule 输出状态 $\hat{u}_j$, 迭代次数$r$, 层数$l$
  + 输出 : $v_j$
  + 需要学习的参数 ：$c_{ij}$
  + 步骤
    + 首先初始化 $b_{ij}$ 为 0
    + 迭代 r 次
      + 对 $b_{ij}$ 按行做 softmax， 得到 $b_{i}$
      + 遍历 $l$ 层 的所有 capsule 对 第 $l$ + 1 层的 第j 个capsule，进行映射， $s_j = \sum_i c_{ij} \hat{u}_{j|i}$
      + 通过 squash 进行压缩，得到 $v_j$
      + 更新参数 
        + 查看了每个高层胶囊j，然后检查每个输入并根据公式更新相应的权重bij
        + 胶囊j的当前输出和从低层胶囊i处接收的输入的点积，加上旧权重，等于新权重
        + 点积检测胶囊的输入和输出之间的相似性
        + 低层胶囊将其输出发送给具有类似输出的高层胶囊， 点积刻画了这一相似性
  + 迭代次数
    + 一般三次，太多的话会导致过拟合

### Framework

+ 编码器

  ![编码器结构](https://ws1.sinaimg.cn/large/006tKfTcly1g1ijgxkgv1j30rs088gm6.jpg)

  + 卷积层

  + PrimaryCaps(主胶囊)

  + DigitCaps(数字胶囊)

  + 损失函数

    ![](https://ws4.sinaimg.cn/large/006tKfTcly1g1ijjiorr9j30rs0953z5.jpg)

+ 解码器

  ![](https://ws3.sinaimg.cn/large/006tKfTcly1g1ijl5sft1j30jg08pq39.jpg)

  + 全连接层1
  + 全连接层2
  + 全连接层3

## Next

+ 能建模更多信息，泛化能力比CNN强
+ 图片信息推理
+ 图片描述生成
+ 文本挖掘：替代n-gram 进行建模

## Drawback

