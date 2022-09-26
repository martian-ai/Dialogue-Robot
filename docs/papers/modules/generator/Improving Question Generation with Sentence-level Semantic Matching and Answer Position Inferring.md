# 

# Introduction
+ 以答案及其上下文为输入，Seq2Seq模型在问题生成方面取得了长足的进步。然而，我们注意到这些方法 经常生成错误的问题词或关键字并复制 回答输入中不相关的单词。我们认为，缺乏全局问题语义和没有很好地利用答案位置意识是关键的根本原因。在本文中，我们 提出了一个包含两个具体模块的神经问题生成模型：句子级语义匹配和答案 位置推断。此外，我们还增强了 译码器，利用应答感知的选通融合机制。实验结果表明，我们的模型在阵容和战术上都优于最先进的（SOTA）模型 马可数据集。由于其通用性，我们的工作也大大改进了现有的模型。


# Proposed Model

## Sentence-Level Semantic Matching

## Answer Position Inferring

## Loss 
+ $L\left(\theta_{a l l}\right)=L\left(\theta_{s 2 s}\right)+\alpha * L\left(\theta_{s m}, \theta_{s 2 s}\right)+\beta * L\left(\theta_{a p}, \theta_{s 2 s}\right)$


https://www.zhihu.com/search?type=content&q=Improving%20Question%20Generation%20with%20Sentence-level%20Semantic%20Matching%20and%20Answer%20Position%20Inferring