# [Fasttext](https://fasttext.cc/)

## Tips
+ 不能跨平台，在不同的平台下要重新编译
+ Fasttext 架构与 word2vec 类似，作者都是 Facebook 科学家 Tomas Mikolov
- 适合类别非常多，且数据量大的场景
- 数据量少的场景容易过拟合

## Framework

  ![](http://www.datagrand.com/blog/wp-content/uploads/2018/01/beepress-beepress-weixin-zhihu-jianshu-plugin-2-4-2-2635-1516863566-2.jpeg)

+ Input and Output
  + 输入是文本序列(词序列 或 字序列)
  + 输出的是这个文本序列属于不同类别的概率
+ Hierachical Softmax

  + 利用霍夫曼编码的方式编码Label（尤其适用用类别不平衡的情况）

+ N-Gram
  + 原始的是Word Bog， 没有词序信息， 因此加入了N-Gram
  + 为了提高效率，低频的N-Gram 特征要去掉

# Reference
- https://blog.csdn.net/john_bh/article/details/79268850
- https://fasttext.cc/