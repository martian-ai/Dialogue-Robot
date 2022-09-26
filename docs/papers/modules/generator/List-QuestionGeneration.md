# Question Generation

##  **Part I Seq2seq QG**

### [Learning to Ask Neural Question Generation for Reading Comprehension](https://www.aclweb.org/anthology/P17-1123.pdf)
+ ACL 2017 / Xinya Du / Cornell
+ 代码数据
    + https://github.com/xinyadu/nqg
    + 利用 SQuAD 构造的 sentence-question 对（MS MARCO）训练集大概有86k 条
    + 数据截图
        + **左侧 context， 右侧 question， 没有指定 answer**
        + ![20220404190602](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220404190602.png)
+ 原理
    + **基于attention的seq2seq模型，基本跟之前的基于检索的模型最好效果差不多**, 使用encoder-decoder 架构进行QG
    + encoder 部分有 Attention-based sentence encoder 和   Paragraph encoder 两种
        + Attention-based sentence encoder 只编码输入话术
        + Paragraph encoder  编码输入话术的同时，编码相关的段落（数据格式为squad 格式）
    + Decoder
        + LSTM

+ 实验
    + 结果
        + ![20220405153101](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220405153101.png)
    + 结果分析
        + 在SQuAD 数据中挑选人工挑选 可以提问的句子，构造  sentence-question 对，利用这些 sentence-pair 对 训练 seq2seq 模型

### [Neural Question Generation from Text：A Preliminary Study](https://arxiv.org/pdf/1704.01792.pdf)
+ EMNLP 2017 / / HIT MSRA BUAA
+ 代码数据
    + https://github.com/magic282/NQG(torch 0.4.0， 代码和数据较为完整)
    + 提供了数据下载的地址
    + 提供的数据中，利用 SQuAD构造的 sentence-question 对(没有使用整个段落，而是人工挑选出来了要用来产生问题的sentence)，并提取了 pos，ner 等特征
    + ![20220405153447](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220405153447.png)
+ 原理
    + 使用了answer position和lexical features 
    + ![20220406221832](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406221832.png)

+ 实验
    + 结果
        + ![20220406222254](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406222254.png)
    + 结果分析
        + 在SQuAD 数据中人工挑选 可以提问的句子，构造  sentence-question 对，利用这些 sentence-pair 对 训练 seq2seq 模型，没有使用answer 信息
        + 模型训练中 使用了 位置特征和词法特征

## Part II Pretrain LM QG

### UniLM
+ 代码数据
    + https://github.com/microsoft/unilm/tree/master/unilm-v1
+ 原理
+ 实验
    + 根据数据切分方式的不同
        + Du 2017 https://github.com/xinyadu/nqg/tree/master/data
        + Zhao 2018 dev-test setup
            + ![20220406222603](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406222603.png)
    + 结果
        + 需要添加
    + 使用 与之前 Learning to Ask: Neural Question Generation for Reading Comprehension 相同的切分，BLEU-4 有7个点的提高，数据构造使用的了 sentence-answer-passage

### CopyBERT
+ 代码数据
    + pass
+ 原理
    + pass
实验
    + 结果
        + ![20220406222738](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406222738.png)

## Part III Document QG

### View
+ Deep Questions without deep understanding ACL2015
+ Leveraging Context Information for Natural Question Generation. Linfeng Song, Zhiguo Wang, Wael Hamza, Yue Zhang, Daniel Gildea. ACL, 2018
+ Paragraph-level Neural Question Generation with Maxout Pointer and Gated Self-attention Networks. Yao Zhao, Xiaochuan Ni, Yuanyuan Ding, Qifa Ke. EMNLP, 2018.
+ Capturing Greater Context for Question Generation. Luu Anh Tuan, Darsh J Shah, Regina Barzilay. arxiv, 2019.


### Part 3.1 Context NQG


### Capturing Greater Context for Question Generation
+ 代码数据
    + suqad
    ![20220406223423](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406223423.png)
+ 原理
    + 图中绿色的句子是答案（1985）所在的sentence，但是其中的this unit具体指代的内容是另外一句话中的内容（红色）。因此使用answer-passage这样一次的attention方式很可能关注不到红色的部分，导致生成的问题定位不到红色部分
    + 作者解决这个问题使用了两次attention，即首先answer-passage做attention，得到的结果再跟passage attention一次
    + ![20220406232703](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406232703.png)
+ 实验
    + 结果
        + ![20220406232807](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406232807.png)
        + split1 和 split2 是SQuAD 数据两种切分方式

### How to Ask Good Questions? Try to Leverage Paraphrases
+ ACL 2020 / / 
+ 数据代码
    + SQuAD  and MARCO 
+ 原理
    + 构造释义生成 PG paraphrase generation 的辅助任务（通过back-translate 对sentence 和 question 都进行扩充）
    + 对sentence-sentecen paraphrase，可以进行paraphrase generation（PG）的训练，作为辅助任务给QG提供paraphrase信息，PG和QG共享encoder，二者的输入都是sentence，只不过训练的目标一个是sentence
    paraphrase，一个是question。在二者的decoder上，使用soft sharing加强互相的影响。PG和QG的loss加权和作为整体模型的loss
    + 对于question 和question paraphrase，二者都可作为正确的reference question使用，因此在QG过程中，计算loss时会分别跟二者计算，然后取较小值作为当前的loss
    + ![20220406232954](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406232954.png)
    + ![20220406233003](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406233003.png)
+ 实验
    + 结果
        + ![20220406233101](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406233101.png)

### Leveraging Context Information for Natural Question Generation
+ 代码数据
    + https://github.com/freesunshine0316/MPQG?utm_source=catalyzex.com
+ 原理
    + 在研究如何更好的利用较长passage中的信息，不同的是作者使用answer的表示去跟passage的每一个hidden算相似度，总共三种方式，第一种是用anwer的最后一个hidden，第二种是把answer 的所有hiddens取加权和，第三种是取max answer hidden。通过这三种方式跟passage进行计算相似度，最后把三种方式的表示拼接作为encoder的输入
    + ![20220406233201](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406233201.png)


### Part 3.2 Coref NQG

### CorefNQG : Harvesting paragraph-level question-answer pairs from wikipedia
+ ACL 2018 / Xinya Du and Claire Cardie  / Cornell
+ 数据代码
    + https://github.com/xinyadu/HarvestingQA 只有数据
    + ![20220406234444](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406234444.png)
+ 原理
    + 创新点：通过指代消解来强化生成效果， 共指信息指的是上下文的代词指的是同一个人，上图中斜体的 his,he 指的都是一个人
    + candidate answer extraction
        + 序列标注
    + answer-specific question generation
    + ![20220406234534](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406234534.png)
+ 实验
    + 结果
        + 答案抽取
            + ![20220406234614](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406234614.png)
        + 问题生成
            + ![20220406234637](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406234637.png)
    + 结果分析
        + 没有对应的中文数据，中文的指代关系较难构建
        + 引入了sequence labeling 进行答案标注，之后使用标注的到的答案进行 qg-with-answer 模型训练
        + 从论文和实际操作中可以看到，识别答案的sequence labling 效果不佳 Exact-F1 大约 0.3

### Part 3.3 Important Sentence Select

### Identifying where to focus in reading comprehension for neural queston generation
+ ACL 2018 / Xinya Du / Cornell
+ 数据代码
    + 无代码
    + 数据为使用SQuAD 的原始数据，按照段落进行切分，train/dev/test = 8:1:1
    + 没有使用答案位置信息
    + Note that generating answer-specific questions would be easy for this architecture — we can append answer location features to the vectors of tokens in the sentence. To better mimic the real life case (where questions are generated with no prior knowledge of the desired answers), we do not use such location features in our experiments
    + ![20220406234955](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406234955.png)
+ 原理 
    + Import Sentence Select
        + sentence encoder : sum or cnn
        + sentence-level sequence labeling : bi-lstm
        + ![20220406235037](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406235037.png)
    + sentence-level QG
        + 同NQG

+ 实验
    + 句子选择的模型和精度
        + Random 随机选择句子
        + Majority Baseline  认为所有的句子权重一致
        + CNN 使用CNN对每个句子提取特征，然后做二分类，判断是不是重要的句子
        + LREGw/ BOW is the logistic regression model with bag-of-words features.
        + LREGw/ para-level is the feature-rich LREG model designed by Cheng and Lapata (2016)
            + the features include: sentence length, position of sentence, number of named entities in the sentence, number of sentences in the paragraph, sentence-tosentence cohesion, and sentence-to-paragraph relevance
            + Sentence-to-sentence cohesion is obtained
    + Ours 本论文提出的方法
        + sum/cnn 提取句子特征 
        + RNN + softmax 进行 sequence labeling
        + ![20220406235250](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406235250.png)
    + 问题生成
        + ![20220406235313](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406235313.png)
        + 指标计算方法
            + ![20220406235412](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406235412.png)
            + if neither the gold data nor prediction include the sentence, then the sentence is discarded from the evaluation; if the gold data includes the sentence while the prediction does not, we assign a score of 0 for it;  and if gold data does not include the sentence while prediction does, the generated question gets a 0 for conservative, while it gets full score for liberal evaluation

### Question-Worthy Sentence Selection for Question Generation
+ 代码数据
+ 原理
    + fearute-baase sentence extraction
        + 特征工程
            + To select sentences for question generation, in [4], different textual features, such as sentence length, sentence position, the total number of entity types, the total number of entities, hardness, novelty, and LexRank measure [7] are individually used to extract question-worthy sentences for a comparison purpose. Here, we train a sentense selection classifier by using multiple features including both context-based and sentence-based features.
        + 分类器
            + Context-aware question generation
            + encoder-decoder
            + ![20220406235559](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406235559.png)
+ 实验
    + 句子选择
        + FS-SM-Pos: A version of FS-SM whose classifier is trained by considering just the POS-tag features 
        + FS-SM-IM: A version of FS-SM whose classifier is trained by considering just the sentence importance features 
        + FS-SM-Rank: A version of FS-SM whose classifier is trained by considering just the rank features
        + ![20220406235638](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406235638.png)


### Part IV Reward QG

### Exploring Question-Specific Rewards for Generating Deep Questions
+ 代码数据
    + https://github.com/YuxiXie/RL-for-Question-Generation
+ 原理
+ 实验
    + 结果
        + ![20220406235810](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220406235810.png)


### Part V Diversity QG

### Mixture Content Selection for Diverse Sequence Generation
+ EMNLP 2019 / / 
+ 代码数据
    + https://github.com/clovaai/FocusSeq2Seq?utm_source=catalyzex.com
    + https://www.zhihu.com/search?type=content&q=Mixture%20Content%20Selection%20for%20Diverse%20Sequence%20Generation.%20EMNLP%2C%202019.%20

### Learning to Generate Questions by Learning What not to Generate
+ WWW 2019 / / 
+ 代码数据
    + https://github.com/BangLiu/QG
+ 算法原理
    + 使用GCN的工作出现了。作者发现对于一个输入sentence，可以从不同的方向提出一个符合要求的问题，比如图中的例子，answer是奥巴马，但是提问方式可以是谁在（某个时间）做了啥，也可以是谁（在某个地点）做了啥，两种提问方式都是切合上下文，并且能被answer回答的。因此作者想在生成问题前先确定要针对哪一部分进行提问
    + 方式就是通过GCN进行一波分类，把分类为1的词打个标签，作为feature embedding提供给后边的seq2seq生成
+ 实验
    + 结果
        + ![20220407000115](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220407000115.png)



### Part VI QA&QG
+ Syn-QG: Syntactic and Shallow Semantic Rules for Question Generation ACL 2020 
    + https://bitbucket.org/kaustubhdhole/syn-qg/src
    + https://bitbucket.org/kaustubhdhole/syn-qg/src/master/
    + Syn-QG is a Java-based question generator which generates questions from multiple sources: 1. Dependency Parsing 2. Semantic Role Labeling 3. NER templates 4. VerbNet Predicates 5. PropBank Roleset Argument Descriptions 6. Custom Rules 7. Implication Rules
+ AnswerQuest: A System for Generating Question-Answer Items from Multi-Paragraph Documents. EACL Demo, 2021.
    + 有的QA 和 QG 的代码，有配套的Flask 工程
    + 没有QG 的训练代码
    + 没有针对SQuAD 的复现结果


### Part VII Semantic QG

### Semantics-based Question Generation and Implementation
+ / / 
+ 代码数据
    + https://github.com/delph-in/pydelphin
    + https://en.wikipedia.org/wiki/DELPH-IN
+ 原理
    + A semantics-based system, MrsQG, is developed to tackle these problems by corresponding solutions: mrs transformation for simple sentences, mrs decomposition for complex sentences and automatic generation with rankings
    + The proposed system consists of multiple syntactic and semantic processing components based on the delph-in tool-chain (e.g. erg/lkb/pet), while the theoretical support comes from Minimal Recursion Semantics
        + primarily in the HPSG syntactic and MRS semantic frameworks
    + ![20220407000558](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220407000558.png)
+ 实验

### Question generation from concept maps
+ ![20220407000638](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220407000638.png)

### Semantic Graphs for Generating Deep Questions
+ ACL 2020 / / 
+ 代码数据
    + code
        + https://github.com/WING-NUS/SG-Deep-Question-Generation
    + data
        + https://www.aclweb.org/anthology/D18-1259.pdf
        + https://hotpotqa.github.io/
+ 原理
    + a document encoder to encode the input document
    + a semantic graph encoder to embed the document-level semantic graph via Att-GGNN
    + a content selector to select relevant question-worthy contents from the semantic graph
    + a question decoder to generate question from the semantic-enriched document representation
    + ![20220407000828](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220407000828.png)

### A Methodology for Creating Question Answering Corpora Using Inverse Data Annotation

+ 实验

### Part VIII Syntax QG

### Good Question！Statistical Ranking for Question Generation 
+ ACL 2010 / / 
+ 代码数据
+ 原理
    + step 1 ：Sentence Simplification  对原始句子进行简化（去掉从句等修饰成分）
    + step 2：Answer Phrase Selection  选择要被提问的答案 （？？？）（当前代码中是对句子中的每一部分都进行了提问）
    + step 3：Main Verb Decomposition 找到句子的主谓宾，对句子进行简化
    + step 4:  Movement and Insertion of Question Phrase 产生问句
    + step 5:  将产生的问句通过ranking 模块，若ranking 分数超过阈值则输出产生的问句，低于阈值则不输出产生的问句
    + ![20220407001023](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220407001023.png)
+ 实验

### Harvesting and Refining Question-Answer Pairs for Unsupervised QA 
+ ACL 2020 / / 
+ 代码数据
    + dataset
        + SQuAD (Rajpurkar et al., 2016, 2018), NewsQA (Trischler et al., 2017) and TriviaQA (Joshi et al., 2017)
    + code
        + https://github.com/Neutralzz/RefQA
        + https://github.com/facebookresearch/UnsupervisedQA
+ 原理
    + step 1 : context generate 将wikipedia 的引用链接到的文档当作context (主要是为了构造 QAS 的数据， 进行MRC， 当前DocQA可以不使用step 1)
    + step 2 : answer generate 通过NER 产生需要问的答案
        人名 == 》 who
        地名 == 》 where
        机构名 == 》 ？？？
        时间/日期 ==》 when
        其他定义？？？
        定义方式参考 Patrick Lewis, Ludovic Denoyer, and Sebastian Riedel. 2019. Unsupervised question answering by cloze translation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4896–4910, Florence,
    + step 3 : close generation  将找到的answer token 用 mask token 替代， 如下图中的 "Elysium" 用 "[THING]"代替 （替代方式由NER 识别的类比来决定， e.g., PRODUCT corresponding to THING, LOC corresponding to PLACE.）
        + ![20220407001220](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220407001220.png)
    + step 4 : Translate Clozes to Natural Questions 生成问题
        + step i : 保留mask token node 的右侧节点， 删除左侧节点 We keep the right child nodes of the answer and prune its lefts
        + step ii : 对于解析树中的每一个节点， 如果其子节点包含mask token node， 则将该子节点移动成为当前节点的第一个子节点(如下图中的 about 从crashed 的最后一个子节点移动成为第一个子节点），通过此步骤完成解析树的重建
        + step III : 通过中序遍历对解析树进行遍历 
        + step iv : 通过规则的方式完成疑问词的映射(Lewis et al. (2019)), 例如 "THING" 这个类别被映射成 “What”
        + ![20220407001315](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220407001315.png)
    + step 5:  Iterative Data Refinement 通过QA模型对挖掘到的数据进行精炼
        + step i : 使用 RefQA 的数据训练 BERT-Based QA model 
        + step ii : 使用 QA model 对没见过的数据进行预测， 获得预测的答案和概率， 并通过阈值对预测出来的答案进行限制
        + step iii : 对于上一步预测出来的答案，如果预测出来的答案和 标注答案一致，则保留原始问题；如果预测出来的答案和标注答案不一致，则认为预测出来的答案是一个新 answer candidate(适合做为答案，但是不是当前问题的答案），针对answer candidate 使用step 4 产生新的问题（理论上认为Bert-Based QA model 有一定的能力可以提取出来当前文本中那些部分适合做答案）
        + step iv : step iii 中会产生很多新的 QA pairs ， 将这些新产生的 QA pairs 加入 QA model 的训练数据，可以提高QA model 的效果
        + ![20220407001406](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/folder/20220407001406.png)

### Part IX Template QG

### Generating Questions Automatically from Informational Text
+ / / 
+ 数据代码
+ 原理
+ 实验
+ This work was further expanded by Chen et al. (2009) with 4 more templates to generate What-wouldhappen-if, When-would-x-happen, What-would-happen-when and Why-x questions from informational text questions. Template-based approaches are mostly suitable for applications with a special purpose, which sometimes come with a closed-domain. The trade-off between coverage and cost is hard to balance because human labor is required to produce high-quality templates. Thus, it is generally considered unsuitable for open-domain general purpose applications