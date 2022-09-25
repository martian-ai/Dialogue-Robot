# Parameter Turning

## Normalization

+ 数据归一化

## Initialization

+ Xavier
+ MSRA

## Learning Rate

+ https://zhuanlan.zhihu.com/p/31424275

## Optimizer

## Regularization

### L1

### L2

### Lasso

### max normal

### weight deacy

### Dropout in RNN

+ https://blog.csdn.net/ningyanggege/article/details/80763289
+ https://blog.csdn.net/mydear_11000/article/details/52414342
+ https://arxiv.org/pdf/1409.2329.pdf
+ 传统的dropout在rnn中效果不是很好；dropout在rnn中使用的效果不是很好，因为rnn有放大噪音的功能，所以会反过来伤害模型的学习能力；
+ 在rnn中使用dropout要放在时间步的连接上，即cell与cell之间传递，而不是神经元；对于rnn的部分不进行dropout，也就是说从t-1时候的状态传递到t时刻进行计算时，这个中间不进行memory的dropout；仅在同一个t时刻中，多层cell之间传递信息的时候进行dropout

![](https://img-blog.csdn.net/20180621172742952?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L25pbmd5YW5nZ2VnZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### Dropout in CNN

+ https://blog.csdn.net/stdcoutzyx/article/details/49022443S
