# Machine Reading Comprehension Summary

![20200330204239](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/20200330204239.png)

## 1. 背景

+ 机器阅读理解，广泛的范围是指构建机器阅读文档，并理解相关内容，通过不断的阅读和理解来建立逻辑能力，来回答相关问题
+ 根据领域范围可分为 封闭领域(Close Domain, Community), 开发领域(Open Domain)
+ 根据任务要求可分为 完形填空(Close Type), 多项选择(Mulit Choice), 答案截取(Question Extraction), 答案生成(Question Generation), 问题生成(Question Generation), 问答对挖掘(QA mining)
+ 根据文档结构可分为 单文档(Single Passage), 多文档(Multi Passage)
+ 根据依赖资源可分为 知识库问答(KBQA), 网络问答(WebQA)
+ Knowledge Mining (???)

## 2. 数据

+ [MRC-Dataset](https://github.com/martian-ai/Prototype-Robot/blob/master/resources/corpus/mrc/MRC-Dataset.md)

## 3. 领域划分

### Close Type

+ Community QA

### Open Domain

+ DrQA
+ ORQA

## 4. 任务划分

### 4.1 Close Type

+ Teach Machine to Read and Comprehension

### 4.2 Question Extraction

#### 4.2.1 single span extract

+ BiDAF
+ Match LSTM
+ AoA
+ spanBert

#### 4.2.2 multi-span extract

+ Tag-based Multi-Span Extraction in Reading Comprehension
+ A Multi-Type Multi-Span Network for Reading Comprehension that Requires Discrete Reasoning

### 4.3 Answer Generation

### 4.4 Question Generation

## 4.5 QA Mining

+ web crawler + answer extraction
+ web crawler + answer generation
+ question generation + answer extraction
+ question generation + answer generation

## 5. 文档结构

### 5.1 多文档(Multi-Passage)

+ Doing

## 最新进展

+ 基于知识的机器阅读理解
+ 不可回答问题
+ 多段式机器阅读理解
+ 对话问答

## 未解决问题

+ 外部知识的整合
+ MRC系统的鲁棒性
+ 给定上下文的局限性
+ 推理能力不足

## Reference

+ https://github.com/thunlp/RCPapers
+ Neural Machine Reading Comprehension: Methods and Trends
  + https://arxiv.org/abs/1907.01118v1
  + [MRC综述: Neural MRC: Methods and Trends](https://zhuanlan.zhihu.com/p/75825101)
+ A STUDY OF THE TASKS AND MODELS IN MACHINE READING COMPREHENSION
  + https://arxiv.org/pdf/2001.08635.pdf
+ [Sogou Machine Reading Comprehension Toolkit](https://arxiv.org/pdf/1903.11848v2.pdf)  
+ 最AI的小PAI：机器阅读理解探索与实践
+ https://stacks.stanford.edu/file/druid:gd576xb1833/thesis-augmented.pdf
+ Danqi Chen: From Reading Comprehension to Open-Domain Question Answering
+ [后bert时代的机器阅读理解](https://zhuanlan.zhihu.com/p/68893946)
+ [DGCNN 在机器阅读理解上的应用](https://zhuanlan.zhihu.com/p/35755367)
+ Joint QA & QG
    + [Dual Ask-Answer Network for Machine Reading Comprehension](https://arxiv.org/pdf/1809.01997v2.pdf)
+ Dataset-transfor
    + MULTIQA An Empirical Investigation of Generalization and Transfer in Reading Comprehension
    + 探索了不同阅读理解数据间的迁移特性
+ A Multi-Type Multi-Span Network for Reading Comprehension that Requires Discrete Reasoning


