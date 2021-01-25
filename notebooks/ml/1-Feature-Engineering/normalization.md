# 数据归一化
### Reference
+ [1] Keras 快速上手 P67
+ https://www.bilibili.com/video/av9770302/?p=10

### 适用情况
+ 数据值变化范围特别大, 标准化数据是一个较好的选择
+ 很多算法的初始值设定也是针对规范化后的数据更有效来设计的(???)

### 好处
+ 加快梯度下降，求最优解的速度
+ 有可能提高精度


### 常见的归一化操作
+ 重缩放(Rescaling)
	+ 加上/减去某个常量, 再乘以/除以某个常量
	+ 例如: 华氏温度到摄氏温度的转化
+ 线性归一化/规范化(Normalization)
	+ 将一个**向量**除以其范数,比如采用欧式空间距离,则用向量的方差作为范数来规范化向量
	+ 在深度学习中, 通茶使用极差(最大值-最小值)作为范数, 即将向量减去最小值并除以极差, 从而使数值范围到(0,1)之间
+ 标准化(Standardization)
	+ 将一个**向量**移除其位置和规模的度量
	+ 比如一个服从正太分布的向量, 可以减去其均值,并除以方差来标准化数据,从而获得一个服从标准正太分布的向量
+ 非线性归一化
	+ 应用在数据分化比较大的场景
	+ 通过数学函数，将原始值进行映射，如ｌｏｇ, tanh

### 深度学习中的归一化
+ 批量归一化
+ 自归一化神经网络

### 不需要归一化的算法
+ 概率模型不需要归一化, 因为它们关心的不是变量的值而是关心变量的分布和变量之间的条件概率, 如Decision Tree, Random Forest
+ 而Adaboost, svm, lr, KNN, KMeans 则需要归一化
+ 树型模型不需要归一化
### 如何选择是否进行以上几种标准化操作
+ 依情况而定
+ 一般来讲, 激活函数的值域是(0,1), 则规范化数据的范围为(0,1)是比较好的

### Batch Normalization(类似标准化Standardization)
+ https://zhuanlan.zhihu.com/p/24810318
+ ![](https://pic1.zhimg.com/80/v2-b31f7d863179f5f0b93d40c4fabbc31a_hd.jpg)
+ 如果不进行batch normalization, 如上图的上半部分, 红色阴影外的数据经过激活函数后会接近1或0(sigmoid), 
+ ![](https://pic2.zhimg.com/80/v2-083ca0bcd0749fd0f236a690b50442e6_hd.jpg)
+ 同一批次的数据减去均值，除以一个类似方差的式子

### Internal Covariate Shift[2]
+ Smaller learning reate can be helpful, but the training would be slower
+ 放在激活函数之前
+ Benefit
	+ 可以设置大一点的Learning rate
	+ 会有更少的 exploding/vanishing gradients
		+ 特别适合 sigmoid 和 tanh
			+ 原理 使结果更多的集中在斜率大的地方
	+ Learning is less affected by initialization
	+ reduces the demand for regularization