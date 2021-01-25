# Prototype Robot

![ds_pic_1.png](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/ds_pic_1.png)

## 构建思路

+ 1. 语料库(resources) : 持续整理相关的数据，语料，构建语料库，已经整理的语料库如下：
  + 1.1 [闲聊](https://github.com/martian-ai/Prototype-Robot/blob/master/resources/corpus/chitchat/Chatbot-Dataset.md)
  + 1.2 [阅读理解](https://github.com/martian-ai/Prototype-Robot/blob/master/resources/corpus/mrc/MRC-Dataset.md)
+ 2. 算法原理(notebooks):
  + pass
+ 3. 工程(solutions) : 借鉴以下方式进行代码构建
  + [transformers](https://github.com/huggingface/transformers)
  + [absa-pytorch](https://github.com/songyouwei/ABSA-PyTorch)
+ 4. 示例代码(examples)
  + pass

+ 在现有的轮子可以满足要求的情况下，优先调用现有轮子；当前轮子不满足要求时，按照一定工程框架开发轮子
+ 重点整理阅读理解相关部分的相关数据，算法，论文和解决方案，集中时间和精力来处理

## 相关计划

+ 1. 完善基本NLP能力的构建(1-30)
  + 词法分析
  + 句法分析
  + 篇章分析
  + 分类
  + 抽取/摘要
  + 搜索匹配/召回排序
+ 2. 搭建ORQA(Open domain Retrieval Question Answer)系统， 完善文档处理，文档召回，文档截取相关模块开发(1.1-1.15)
  + 2.1 完善问句改写/召回/排序 等功能(当前方式为es)
  + 2.2 调试答案截取功能
  + 2.3 尝试答案生成功能
  + 2.4 尝试问题生成功能
+ 3. 搭建KBQA(Knowledge base Question Answer)系统，完善知识挖掘，知识库构建，知识推理相关能力调研(1.16-1.30)
  + 3.1 整理现有知识库资源
  + 3.2 思考知识和逻辑的表示方式，构建知识和逻辑的表示形式
  + 3.3 构建知识挖掘相关方法，动态维护知识库，并进行逻辑进化
  + 3.4 利用知识库进行知识推理
+ 4. 搭建Chatbot系统，调优多轮检索和多轮生成相关能力(2.1-2.15)
  + 4.1 尝试强化学习(Policy-based/Vaule-based/A2C)等方式
  + 4.2 尝试GAN(SeqGAN/...)等生成模型
  + 4.3 多轮检索模型迭代优化
+ 5. 搭建DM(Dialog Manger)系统，完善对话状态跟踪，对话策略学习等能力，对QRQA，KBQA，Chatbot 进行调度
+ 6. 完善用户画像，系统设定等方面能力
+ 7. 实现一定程度上的人机对话
+ 8. 基于flask 等工具进行部署，对外提供服务

## Reading

+ [Never-Ending Learning for Open-Domain Question Answering
over Knowledge Bases](https://myahya.org/publications/neqa-abujabal-www2018.pdf)
+ [ELI5:Long From Question Answering](https://arxiv.org/pdf/1907.09190.pdf)
+ [thu rc papers](https://github.com/thunlp/RCPapers)
+ [A Large-Scale Chinese Short-Text Conversation Dataset](https://arxiv.org/pdf/2008.03946.pdf)
+ [UniLM](https://github.com/microsoft/unilm)

## Readed

+ [thu OpenQA](https://github.com/thunlp/OpenQA)
+ [Multi-Passage Machine Reading Comprehension with Cross-Passage Answer Verification](https://arxiv.org/pdf/1805.02220.pdf)
