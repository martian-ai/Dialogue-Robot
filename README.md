# Prototype Robot

![ds_pic_1.png](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/ds_pic_1.png)

## 1. 非结构化数据挖掘

+ 目标：对非结构化信息进行数据挖掘

  + 非结构化信息包括 文本，文档等
+ 挖掘方式

  + QA Pair Mining
  + KB Mining
  + 文档结构化
+ 结果保存

  + ES 数据库

## 2. 利用挖掘信息进行对话服务

+ 利用挖掘结果完成对话能力

  + FAQ
  + ORQA
  + KBQA

## 3. 多轮对话

## 4. 个人助手
+ ORQA

## 版本管理

+ v 0.1.0
  + application/doc_qa_mining

    + 基础问答对挖掘能力开发
    + 使用 textrank 选择重要句子作为摘要 Summary， 遍历重要句子作为答案 Answer，通过重要句子 和 摘要 来生生成问题 Question ，挖掘结果为 {Sumary， Question， Answer}
+ v 0.1.1

## 数据资源

+ 文档资源 resouces/corpus
  + 刘慈欣小说 云盘地址
  + wiki 数据
+ 网络资源 resouces/crawler
  + 爬虫
+ 常见数据资源整理 resouces/dataset
  + chrome


# TODO
+ generate
  + vocab
  + GenerateRNN
+ resources 


# 对话服务能力图

```flowchart
  start=>start: 用户请求QUERY
  query_nlu=>operation: QUERY理解

  faq_clf=>condition: FAQ 前置分类
  faq_domain_clf=>operation: FAQ场景分类
  docqa_clf=>condition: DocQA 前置分类

  faq_recall=>operation: 相似问召回
  faq_match=>operation: 相似问匹配
  
  docqa_recall=>operation: 文档域检索
  orqa_recall=>operation: 开放域检索

  multi_passage_rc=>operation: 多文档答案应答

  other=>operation: 其他逻辑

  end=>end: 输出答案

  start->query_nlu->faq_clf
  faq_clf(yes)->faq_domain_clf
  faq_clf(no)->docqa_clf

  faq_domain_clf->faq_recall

  docqa_clf(yes)->docqa_recall
  docqa_clf(no)->orqa_recall

  faq_recall->faq_match
  docqa_recall->multi_passage_rc
  orqa_recall->multi_passage_rc

  multi_passage_rc->end
  faq_match->end
```

