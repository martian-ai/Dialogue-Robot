
# 1. 同义词方案

## 1.1 知识库

### 1.1.1《哈工大信息检索研究室同义词词林扩展版》

+ 资源 https://github.com/yaleimeng/Final_word_Similarity/tree/master/cilin/V3
    + 23575 个条目
+ 处理脚本 https://github.com/yaleimeng/Final_word_Similarity/blob/master/cilin/V3/ciLin.py

### 1.1.2 HowNet
+ 资源 https://raw.githubusercontent.com/yaleimeng/Final_word_Similarity/master/hownet/glossary.txt
    + 66182 个条目

+ 处理脚本 
    + https://github.com/yaleimeng/Final_word_Similarity/blob/master/hownet/howNet.py


### 1.1.3 搜索引擎(百度百科)

![20200814150530-2020-8-14-15-5-30](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200814150530-2020-8-14-15-5-30)

![20200817143434-2020-8-17-14-34-34](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200817143434-2020-8-17-14-34-34)

+ 处理脚本
    + https://github.com/tigerchen52/synonym_detection/blob/master/source/baike_crawler_model.py
    + 实测可用


## 1.2 上下文相关性
+ 弱监督同义词挖掘: [DPE](https://arxiv.org/pdf/1706.08186.pdf)

## 1.3 语义网络
+ https://github.com/tigerchen52/synonym_detection/blob/master/source/semantic_network_model.py

### 1.4 文本相似性
+ word2vec
+ Levenshtein distance
+ Longest common subsequence

# 2. 词义消歧

+ 找到带消歧词(如何寻找)

+ 主要通过百度百科，将待输入的消歧词进行查询，从而得到相关义项，并将消歧句子与各个义项所表示的句子进行相似性计算，从而得到与之相关的该消歧词的意思，有点远程监督的味道
![20200817154109-2020-8-17-15-41-10](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200817154109-2020-8-17-15-41-10)

+ https://arxiv.org/pdf/1805.04032.pdf
    +  非监督
        + 基于聚类
        + 基于共同训练
        + 基于上下文语境

# 3. Tools
+ synonyms 

# Reference
+ https://www.jianshu.com/p/94ab423d0772
+ HIT-CIR Tongyici Cilin  
    + file:///Users/sunhongchao/Zotero/storage/NMEV3IY8/Sharing_Plan.html
+ Hownet
    + https://openhownet.thunlp.org/download
+ https://blog.csdn.net/Zh823275484/article/details/88115041
+ https://github.com/yaleimeng/Final_word_Similarity
+ https://github.com/tigerchen52/synonym_detection
+ 词义消歧
    + https://www.jianshu.com/p/5331af742076
    + https://arxiv.org/pdf/1805.04032.pdf