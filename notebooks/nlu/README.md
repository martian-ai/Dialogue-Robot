
# Summary of NLU in Dialogue System

## 0. 问句改写

### 0.1 纠错

+ FASPell
  + https://github.com/iqiyi/FASPell
+ pycorrector
  + https://blog.csdn.net/yiminghd2861/article/details/84181349

+ 基于transformer 的通用纠错


### 0.2 专有名替换/缩略词扩充

+ 针对具体应用场景对专有名词和缩略词进行处理
+ https://github.com/zhangyics/Chinese-abbreviation-dataset

### 0.3 文本归一化

+ 数据补充，相关正则表达式添加
+ 可以用算法处理吗？
+ 时间/单位/数

### 0.4 繁简转化

+ snow nlp

### 0.5 翻译为中文

+ 调用翻译接口

### 0.6 去除非中文

## 1. 词法分析

### 1.1 分词

+ jieba/ltp

## 1.2 词性标注

+ ltp

## 1.3 命名实体识别

+ ltp

## 2. 句法分析

+ ltp

## 3. 篇章分析

### 3.1 分句

### 3.2 长句压缩

+ TextRank
+ snowlp
+ 语法树分析 加 关键词典
+ 抽取方式
+ 生成方式
+ ByteCup 基于transformer

### 3.3 指代消解

+ standford core nlp
+ (PAI)Mention Pair models：将所有的指代词（短语）与所有被指代的词（短语）视作一系列pair，对每个pair二分类决策成立与否。
+ (PAI)Mention ranking models：显式地将mention作为query，对所有candidate做rank，得分最高的就被认为是指代消解项。
+ (PAI)Entity-Mention models：一种更优雅的模型，找出所有的entity及其对话上下文。根据对话上下文聚类，在同一个类中的mention消解为同一个entity。但这种方法其实用得不多。

+ 分类
  + https://blog.csdn.net/u013378306/article/details/64441596
+ 零指代消解
  + https://www.jiqizhixin.com/articles/2018-07-28-8
+ https://zhuanlan.zhihu.com/p/53550123
+ https://zhuanlan.zhihu.com/p/103794289

## 4. 语义分析

+ 词义消歧
+ 语义角色标注