# SummaRuNNer

## 背景
+ 抽取式文档摘要

## 模型结果
+ 1. embedding
+ 2. 原始 GRU-RNN
+ 2. 改进 CNN-RNN 
    + 由word-level CNN得到sentence的表示， 又经过sentence-level RNN得到document级别的表示 $\mathbf{d}=\tanh \left(W_{d} \frac{1}{N_{d}} \sum_{j=1}^{N^{d}}\left[\mathbf{h}_{j}^{f}, \mathbf{h}_{j}^{b}\right]+\mathbf{b}\right)$
    + s 是文本的动态表示
        + $\mathbf{s}_{j}=\sum_{i=1}^{j-1} \mathbf{h}_{i} P\left(y_{i}=1 \mid \mathbf{h}_{i}, \mathbf{s}_{i}, \mathbf{d}\right)$
+ 3. 分类
    + 第一项计算content， 第二项计算salience(显著性)， 第三项计算创新性，第四项表示绝对位置，第五项表示相对位置
$$
\begin{array}{r}
P\left(y_{j}=1 \mid \mathbf{h}_{j}, \mathbf{s}_{j}, \mathbf{d}\right)=\sigma\left(W_{c} \mathbf{h}_{j}\right. \\
+\mathbf{h}_{j}^{T} W_{s} \mathbf{d} \\
-\mathbf{h}_{j}^{T} W_{r} \tanh \left(\mathbf{s}_{\mathbf{j}}\right) \\
+W_{a p} \mathbf{p}_{j}^{a} \\
+W_{r p} \mathbf{p}_{j}^{r} \\
+b),
\end{array}
$$

## loss
+ $\begin{aligned} l(\mathbf{W}, \mathbf{b}) &=-\sum_{d=1}^{N} \sum_{j=1}^{N_{d}}\left(y_{j}^{d} \log P\left(y_{j}^{d}=1 \mid \mathbf{h}_{j}^{d}, \mathbf{s}_{j}^{d}, \mathbf{d}_{d}\right)\right.\\ &+\left(1-y_{j}^{d}\right) \log \left(1-P\left(y_{j}^{d}=1 \mid \mathbf{h}_{j}^{d}, \mathbf{s}_{j}^{d}, \mathbf{d}_{d}\right)\right) \end{aligned}$

## 训练方式
+ 由于数据集都是生成式(abstrctive)的，所以不能直接用于SummaRuNNer这个模型的抽取式训练。该论文使用贪心算法来抽取文章的句子，每次向摘要中添加一个句子，使rouge得分最高，当剩余的候选句子都没有改善Rouge分数时，停止添加。