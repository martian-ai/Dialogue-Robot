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
+ [ChineseSemanticKB](https://github.com/liuhuanyong/ChineseSemanticKB)
+ [QA-Survey](https://github.com/BDBC-KG-NLP/QA-Survey)
+ https://github.com/WenDesi/lihang_book_algorithm
+ https://github.com/kushalj001/pytorch-question-answering 形式不错，使用ipynb 做原理解释和代码展示
+ https://github.com/luopeixiang/named_entity_recognition sequence labeling 的一些整理，方法比较全
+ https://github.com/quincyliang/nlp-data-augmentation 语料增强
+ [DialoGTP](https://github.com/microsoft/DialoGPT) 
+ [NN-Brain](https://github.com/gyyang/nn-brain)
+ [ACL 2020 Tutorial:Open-Domain Question Answering](https://github.com/danqi/acl2020-openqa-tutorial)
+ [statistical learning method](https://github.com/Dod-o/Statistical-Learning-Method_Code)
+ [CDial-GPT](https://github.com/thu-coai/CDial-GPT)
+ [MRPC](https://github.com/shuohangwang/mprc) 检索和阅读放一起
+ [Awesome-QG](https://github.com/bisheng/Awesome-QG)
+ [Joint-MRC](https://github.com/caishiqing/joint-mrc) 检索和阅读放一起
+ [Capsule-MRC](https://github.com/freefuiiismyname/capsule-mrc)
+ [DRCD](https://github.com/DRCKnowledgeTeam/DRCD)
+ [刘建平博客代码](https://github.com/ljpzzz/machinelearning)
+ [SemEval-2016 Task 9](https://github.com/HIT-SCIR/SemEval-2016)
+ [TopicModelling](https://github.com/balikasg/topicModelling)
+ [lightKG](https://github.com/smilelight/lightKG)
+ [CLUEPretrainedModels](https://github.com/CLUEbenchmark/CLUEPretrainedModels)
+ [CLUEDatasetSearch](https://github.com/CLUEbenchmark/CLUEDatasetSearch)
+ [OpenBookQA](https://github.com/allenai/OpenBookQA)
+ [CMRC2018](https://github.com/ymcui/cmrc2018)
+ [CMRC2017](https://github.com/ymcui/cmrc2017)
+ [PRML](https://github.com/ctgk/PRML)
+ [AWESOME-VAES](https://github.com/matthewvowels1/Awesome-VAEs)
+ [PYGCN](https://github.com/tkipf/pygcn)
+ [IMN](https://github.com/JasonForJoy/IMN)
+ [HarvestText](https://github.com/blmoistawinde/HarvestText)
+ [Chatbot_CN](https://github.com/charlesXu86/Chatbot_CN) 完整例子
+ [Chinese2Digits](https://github.com/Wall-ee/chinese2digits) 中文转数字
+ [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
+ [huggingface/tokenizers](https://github.com/huggingface/tokenizers)
+ [correction](https://github.com/ccheng16/correction)
+ [K-BERT](https://github.com/autoliuweijie/K-BERT)
+ [Multi Step Reasoning for Open Domain Question Answering](https://github.com/rajarshd/Multi-Step-Reasoning)
+ [Chinese-NLP-Corpus](https://github.com/OYE93/Chinese-NLP-Corpus)
+ [CLUE](https://github.com/CLUEbenchmark/CLGE)
+ [hanzi char featurizer](https://github.com/howl-anderson/hanzi_char_featurizer)
+ [abbreviation](https://github.com/zhangyics/Chinese-abbreviation-dataset)
+ [awesome-nlg](https://github.com/tokenmill/awesome-nlg)
+ [GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)
+ [ChatterBot Language Training Corpus](https://github.com/gunthercox/chatterbot-corpus)
+ [Seq2seqChatbots](https://github.com/ricsinaruto/Seq2seqChatbots) tensor2tensor
+ [Kernelized Bayesian Softmax for Text Generation](https://github.com/NingMiao/KerBS)
+ [Awesome-Chatbot](https://github.com/fendouai/Awesome-Chatbot)
+ [KnowledgeGraph](https://github.com/ownthink/KnowledgeGraph)
+ [CTRL](https://github.com/salesforce/ctrl)
+ [TG-Reading-List](https://github.com/Apollo2Mars?after=Y3Vyc29yOnYyOpK5MjAxOS0wOS0yM1QxNTo1ODo1MCswODowMM4LLtIm&tab=stars) 清华text generation paper list
+ [MulitQA](https://github.com/alontalmor/MultiQA) 
+ [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)

## Readed

+ [thu OpenQA](https://github.com/thunlp/OpenQA)
+ [Multi-Passage Machine Reading Comprehension with Cross-Passage Answer Verification](https://arxiv.org/pdf/1805.02220.pdf)
