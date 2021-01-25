# Probability Graph Theory

## Outline

+ 有向图（directed graphical model）可以刻画变量间显示存在的因果关系或依赖关系，该模型又称为贝叶斯网络（bayes network）、信念网（belief network）、因果网等
+ 无向图中变量间显示的因果关系难以获得，因而只能反映变量间的相关性即关联关系，其中马尔科夫随机场（Markov Model）、因子图（factor graph）即为无向图

![20200628172414](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/nlu/20200628172414.png)

## HMM

## CRF

+ https://www.cnblogs.com/pinard/p/7048333.html
+ HMM 求解过程可能是局部最优 ，HMM/MEMM 导致标注偏置
+ CRF 全局最优， 没有标注偏置

![20200621222100](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/20200621222100.png)

## LDA

+ https://blog.csdn.net/u012771351/article/details/53032365
  + 二项分布的共轭先验分布是Beta分布，多项分布的共轭先验分布是Dirichlet分布
  + Gibbs 采样部分有较为详细的解释
  + ![20200628161252](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/20200628161252.png)
+ [LDA 基础](https://www.cnblogs.com/pinard/p/6831308.html)
+ [LDA Gibbs 采样](http://www.cnblogs.com/pinard/p/6867828.html)
+ [LDA 变分推导 EM 算法](http://www.cnblogs.com/pinard/p/6873703.html)

## Reference
+ https://zhuanlan.zhihu.com/p/54101808