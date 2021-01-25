# Syntax Analysis

## 句法分析的任务/短语结构分析/成分句法分析
+ 判断输出的字符串是否属于某种语言
+ 消除输入句子中词法和结构等方面的歧义
+ 分析输入句子的内部结构，如成分构成、上下文关系等

## 句法结构分析
+ CFG
+ PCFG
+ PCFG-LA

![PCFG 和 PCFG-LA 的对比](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200609184132-2020-6-9-18-41-33)

## 浅层语法分析(局部语法分析)
### 语块的识别和分析(chunking)
+ base NP 属于语块的一种
### 语块之间的依附关系分析

## 依存句法分析

![依存分析的三种形式](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200609184232-2020-6-9-18-42-32)

+ [LTP依存句法分析标注集](https://ltp.readthedocs.io/zh_CN/latest/appendix.html#id5)

![20200518155949-2020-5-18-15-59-50](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200518155949-2020-5-18-15-59-50)

## 语义角色标注

+ [LTP语义角色标注标注集](https://ltp.readthedocs.io/zh_CN/latest/appendix.html#id5)

![20200609182301-2020-6-9-18-23-2](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20200609182301-2020-6-9-18-23-2)

## 语义依存分析(Semantic Dependancy Parsing, SDP)
+ http://www.ltp-cloud.com/intro#sdp_how



## 深层文法句法分析
+ 即利用深层文法，例如词汇化树邻接文法（Lexicalized Tree Adjoining Grammar， LTAG）、词汇功能文法（Lexical Functional Grammar， LFG）、组合范畴文法（Combinatory Categorial Grammar， CCG）等，对句子进行深层的句法以及语义分析

## Reference
+ https://www.jianshu.com/p/fb408b6a0904
+ 使用句法分析抽取事实三元组
    + https://github.com/twjiang/fact_triple_extraction
+ https://zhuanlan.zhihu.com/p/51186364