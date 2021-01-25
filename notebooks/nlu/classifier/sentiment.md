
# Sentiment Analysis

![20200421170211](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/nlu/20200421170211.png)

## Tips

- 文本分类更侧重与文本得客观性，情感分类更侧重主观性

## Solutions

| Model                                                        | Tips                                                |
| ------------------------------------------------------------ | --------------------------------------------------- |
| TD-LSTM [paper](https://link.zhihu.com/?target=http%3A//www.aclweb.org/anthology/C16-1311) [code](https://link.zhihu.com/?target=https%3A//github.com/jimmyyfeng/TD-LSTM) | COLING 2016；两个LSTM 分别编码 context 和 target    |
| TC-LSTM [paper]() [blog](https://zhuanlan.zhihu.com/p/43100493) | 两个LSTM 分别添加 target words 的 average embedding |
| AT-LSTM                                                      | softmax 之前加入 aspect embedding                   |
| ATAE-LSTM [paper](Wang, Yequan, Minlie Huang, and Li Zhao. "Attention-based lstm for aspect-level sentiment classification." Proceedings of the 2016 conference on empirical methods in natural language processing. 2016) [source-code](http://coai.cs.tsinghua.edu.cn/media/files/atae-lstm_uVgRmdb.rar) | EMNLP 2016；输入端加入 aspect embedding             |
| BERT-SPC [paper](https://arxiv.org/pdf/1810.04805.pdf) [code](https://github.com/songyouwei/ABSA-PyTorch) |                                                     |
| MGAN [paper](http://aclweb.org/anthology/D18-1380) [code](https://github.com/songyouwei/ABSA-PyTorch) | ACL 2018                                            |
| AEN-BERT [paper](https://arxiv.org/pdf/1902.09314.pdf) [code](https://github.com/songyouwei/ABSA-PyTorch) | ACL 2019                                            |
| AOA                                                          |                                                     |
| TNet                                                         |                                                     |
| Cabasc                                                       |                                                     |
| RAM                                                          | EMNLP 2017                                          |
| MemNet                                                       | EMNLP 2016                                          |
| IAN                                                          |                                                     |

## Aspect Extraction

- 频繁出现的名次或名次短语
  - PMI
- 分析opinion 和 target 的关系
  - 依存关系：如果opinion 已知， sentiment word可以通过依存关系知道
  - 另一种思想：使用依存树找到aspect 和 opinion word 对，然后使用树结构的分类方法来学习，aspect从得分最高的pair 得到(???)
- 有监督
  - 序列标注，HMM/CRF
- 主题模型
  - pLSA 和 LDA

## ABSA

### AT-LSTM

- https://arxiv.org/pdf/1512.01100
- 每一时刻输入word embedding，LSTM的状态更新，将隐层状态和aspect embedding结合起来，aspect embedding作为模型参数一起训练，得到句子在给定aspect下的权重表示r

### ATAE-LSTM

- https://aclweb.org/anthology/D16-1058
- AT-LSTM在计算attention权重的过程中引入了aspect的信息，为了更充分的利用aspect的信息，作者在AT-LSTM模型的基础上又提出了ATAE-LSTM模型，在输入端将aspect embedding和word embedding结合起来

### BERT-SPC

- Bert Pre-training of deep bidirectional transformers for language understanding

### MGAN

http://aclweb.org/anthology/D18-1380
![20200627221549](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/nlu/20200627221549.png)

### ANE-BERT

![20200627221328](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/nlu/20200627221328.png)

## Views

### Papers

### Benchmarking Multimodal Sentiment Analysis

- 多模态情感分析目前还有很多难点，该文提出了一个基于 CNN 的多模态融合框架，融合表情，语音，文本等信息做情感分析，情绪识别。
- 论文链接：https://www.paperweekly.site/papers/1306

### Learning Sentiment Memories for Sentiment Modification without Parallel

- Sentiment Modification : 将某种情感极性的文本转移到另一种文本
- 由attention weight 做指示获得情感词, 得到 neutralized context(中性的文本)
- 根据情感词构建sentiment momory
- 通过该memory对Seq2Seq中的Decoder的initial state 进行初始化, 帮助其生成另一种极性的文本

### ABSA-BERT-pair

- <https://github.com/HSLCY/ABSA-BERT-pair>
- <https://arxiv.org/pdf/1903.09588.pdf>

### [Deep Learning for sentiment Analysis - A survey](<https://arxiv.org/pdf/1801.07883.pdf>)

- Date:201801
  - Tips
    - A survey of deep learning approach on sentiment analysis
    - Introduce various types of Sentiment Analysis 
      - Document level
      - Sentence level
      - Aspect level
      - Aspect Extraction and categorization
      - Opinion Expression Extraction
      - Sentiment Composition
      - Opinion Holder Extraction
      - Temporal Opinion Mining
      - SARCASM Analysis(讽刺)
      - Emotion Analysis
      - Mulitmodal Data for Sentiment Analysis
      - Resource-poor language and multilingual sentiment anslysis
  
- 统计自然语言处理 P431
  
  - https://zhuanlan.zhihu.com/p/23615176
  
- Attention-based LSTM for Aspect-level Sentiment Classification
  
  - [http://zhaoxueli.win/2017/03/06/%E5%9F%BA%E4%BA%8E-Aspect-%E7%9A%84%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90/](http://zhaoxueli.win/2017/03/06/基于-Aspect-的情感分析/)
  
### Projects

- <https://github.com/songyouwei/ABSA-PyTorch>
- https://github.com/12190143/Deep-Learning-for-Aspect-Level-Sentiment-Classification-Baselines

### Challenge

### [SemEval 2014](http://alt.qcri.org/semeval2014/task4/](http://alt.qcri.org/semeval2014/task4/))

- [Data v2.0]([http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-train-data-v20-annotation-guidelines/683b709298b811e3a0e2842b2b6a04d7c7a19307f18a4940beef6a6143f937f0/](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-train-data-v20-annotation-guidelines/683b709298b811e3a0e2842b2b6a04d7c7a19307f18a4940beef6a6143f937f0/))

- [data demo]([http://alt.qcri.org/semeval2014/task4/data/uploads/laptops-trial.xml](http://alt.qcri.org/semeval2014/task4/data/uploads/laptops-trial.xml))

- subtask