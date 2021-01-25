
# Problems for NLU
+ Few Data
    + 冷启动

+ Few Label
    + 生成标注数据
    + Activate Learning : 少量数据训练模型， 使用模型预测一批数据，人工修改模型预测的数据，使用新的数据进行训练，依次迭代

+ Corrupter Label(标注错误较多)
    + 更好的发现错误的指标(confusion matrix)
    + 针对错误类别重点回滚

+ Unbalance Diversity(数据多样性差)


+ Unbalance Classification(分类问题中的不平衡性对性能有影响，针对不平衡的问题做如下处理
    + 更好的评测不平衡问题 （marco, multi class ROC-AUC， sensitivity， specificity, mAP, Precision@Rank k）
    + 针对较差的类别生成数据

# Solutions : Unbalanced and Few Data Problem

## Data Augemens
- [4.1. Synonyms](#41-synonyms)
    - [4.1.1. BERT](#411-bert)
    - [4.1.2. ERNIE](#412-ernie)
    - [4.1.3. BERT-WWW](#413-bert-www)
    - [4.1.4. synonyms](#414-synonyms)
- [4.2. Random Insertion](#42-random-insertion)
- [4.3. Random Swap](#43-random-swap)
- [4.4. Random  Deletion](#44-random--deletion)
- [4.5. 回译](#45-回译)
- [4.6. 文档剪辑（长文本）](#46-文档剪辑长文本)
- [4.7. 文本生成](#47-文本生成)
- [4.8. 预训练语言模型](#48-预训练语言模型)
- [4.9. 文本更正](#49-文本更正)
    

    - 检查一致性

## Training Model
- [6.5.2. change weight of loss](#652-change-weight-of-loss)
    - [6.5.2.1. weight loss](#6521-weight-loss)
    - [6.5.2.2. Focal Loss](#6522-focal-loss)
    - [6.5.2.3. Learning weight](#6523-learning-weight)
- [6.5.3. EDA](#653-eda)
- [6.5.4. UDA ](#654-uda)
- [6.5.5. Ensemble ](#655-有监督的集成学习)

![20191024181928.png](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20191024181928.png)


# Reference

## Papers
![20191024185255.png](https://blog-picture-bed.oss-cn-beijing.aliyuncs.com/blog/upload/20191024185255.png)

## Links
+ 各自优缺点
    + https://www.zhihu.com/question/30643044
+ 严重不平衡
    + https://www.zhihu.com/question/59236897
+ 数据挖掘中的异常检测方法
    + https://www.zhihu.com/question/280696035/answer/417091151