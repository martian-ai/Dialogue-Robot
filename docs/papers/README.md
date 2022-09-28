# Paper List

## Part I Math

+ Neural Variational Inference for Text Processing
+ [Probabilistic &amp; Unsupervised Learning Approximate InferenceFactored Variational Approximationsand Variational Bayes](http://www.gatsby.ucl.ac.uk/teaching/courses/ml1/lect8-handout.pdf)
+ [VAE](https://arxiv.org/abs/1606.05908)

## Part II Modules

### Transformers

+ base
+ reformer
+ Self-Attention with Relative Position Representations
+ xl

### Embedding

+ BERT

  + embedding/BERT.md
+ BART

  + 只有英文
  + https://zhuanlan.zhihu.com/p/270925325
  + https://arxiv.org/abs/1910.13461
  + https://github.com/huggingface/transformers/blob/v3.0.2/src/transformers/modeling_bart.py
+ CPT: A Pre-Trained Unbalanced Transformer for Both Chinese Language Understanding and Generation

  + https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2109.05729.pdf
  + https://link.zhihu.com/?target=https%3A//github.com/fastnlp/CPT
  + 有中文
+ T5

  + 有中文

### topic

### generation

+ summary

  + [awesome-text-generatioin](https://github.com/ChenChengKuan/awesome-text-generation#gan-based)
  + Pretrain language models for text generation : a survey [zhihu](https://zhuanlan.zhihu.com/p/417117371)

    + 基于预训练语言模型的文本生成研究综述
  + [A Survey of Knowledge-Enhanced Text Generation]()  [zhihu](https://zhuanlan.zhihu.com/p/356399466)
+ **[Seq2Seq] Sequence to Sequence Learning with Neural Networks** , in NeurIPS 2014. [[pdf]](https://arxiv.org/pdf/1409.3215.pdf)
+ **[SeqAttn] Neural Machine Translation by Jointly Learning to Align and Translate** , in ICLR 2015. [[pdf]](https://arxiv.org/pdf/1409.0473.pdf)
+ **[CopyNet] Incorporating Copying Mechanism in Sequence-to-Sequence Learning** , in ACL 2016. [[pdf]](https://arxiv.org/abs/1603.06393)
+ **[PointerNet] Get To The Point: Summarization with Pointer-Generator Networks** , in ACL 2017. [[pdf]](https://arxiv.org/abs/1704.04368)
+ **[Transformer] Attention Is All You Need** , in NeurIPS 2017. [[pdf]](https://arxiv.org/abs/1706.03762)
+ 结构化生成

  + Text summarization with pretrained encoders, in EMNLP, 2019. [pdf](https://arxiv.org/abs/1908.08345) [code](https://github.com/nlpyang/PreSumm)
  + Sentence centrality revisited for unsupervised summarization, in ACL, 2019.
  + Pre-trained language model representations for language generation, in NAACL-HLT, 2019.
  + Multi-granularity interaction network for extractive and abstractive multi-document summarization, in ACL, 2020.
  + HIBERT: document level pretraining of hierarchical bidirectional transformers for document summarization, in ACL, 2019.
  + Unsupervised extractive summarization by pre-training hierarchical transformers, in EMNLP, 2020.
  + Discourse-Aware Neural Extractive Text Summarization, in ACL, 2020. [[pdf]](https://www.aclweb.org/anthology/2020.acl-main.451/) [[code]](https://github.com/jiacheng-xu/DiscoBERT)
  + Cross-lingual language model pretraining, in NeurIPS, 2019.
  + Multilingual denoising pretraining for neural machine translation, in TACL, 2020.
  + Unsupervised cross-lingual word embedding by multilingual neural language models, in arXiv preprint arXiv:1809.02306, 2018.
+ 非结构化生成

Control NLG

+ Prabhumoye S, Black A W, Salakhutdinov R. Exploring Controllable Text Generation Techniques[J]. arXiv preprint arXiv:2005.01822, 2020.
+ https://baijiahao.baidu.com/s?id=1712128443813723814&wfr=spider&for=pc
  + 目前可控文本生成已有大量的相关研究，比较有趣的研究有，SongNet（Tencent）控制输出诗词歌赋的字数、平仄和押韵；StylePTB（CMU）按照控制信号改变句子的语法结构、单词形式、语义等；CTRL（Salesforce）在预训练阶段加入了 control codes 与 prompts 作为控制信号，影响文本的领域、主题、实体和风格。可控文本生成模型等方案也多种多样，此处按照进行可控的着手点和切入角度，将可控文本生成方案分为：构造 Control Codes、设计 Prompt、加入解码策略（Decoding Strategy），以及 Write-then-Edit 共四类。

Long Input NLG

Long Output NLG

Multi Passage NLG

### Sum

+ fastSum https://github.com/fastnlp/fastSum

## Part III Solutions

### FAQ

+ RocketQA
  + https://zhuanlan.zhihu.com/p/532242020
  + 《DuReader_retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine》
  + 《RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering》
  + 《RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking》

# QA Mining

+ [Improving Unsupervised Question Answering via Summarization-Informed Question Generation](https://arxiv.org/abs/2109.07954)
  + We make use of freely available news summary data, transforming declarative summary sentences into appropriate questions using heuristics informed by dependency parsing, named entity recognition and semantic role labeling. The resulting questions are then combined with the original news articles to train an end-to-end neural QG model
+ Summarize-then-Answer: Generating Concise Explanations for Multi-hop Reading Comprehension
  + HotpotQA
  + https://github.com/StonyBrookNLP/suqa

# DocQA

+ Building Goal-oriented Document-grounded Dialogue Systems
+ End-to-End Training of Multi-Document Reader and Retriever for Open-Domain Question Answering

# Todo

+ https://github.com/wyu97/KENLG-Reading
