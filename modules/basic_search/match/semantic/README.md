+ 背景
  + 本项目将持续更新，对比目前业界主流文本匹配模型在中文的效果

+ 数据
  + 数据集采用LCQMC数据（将一个句对进行分类，判断两个句子的语义是否相同（二分类任务）），因数据存在侵权嫌疑，故不提供下载，需要者可向官方提出数据申请http://icrc.hitsz.edu.cn/info/1037/1146.htm ，并将数据解压到data文件夹即可。模型评测指标为：ACC，AUC以及预测总共耗时。

+ 开发中
  + CDSSM

+ 测试结果  

|                                    模型                                     |  ACC   |  AUC   | 耗时（s）（备注：环境1070TI) |
| :-------------------------------------------------------------------------: | :----: | :----: | :--------------------------: |
|                [ABCNN](https://arxiv.org/pdf/1512.05193.pdf)                | 0.8081 | 0.9059 |            4.6260            |
|             [Albert](https://openreview.net/pdf?id=H1eA7AEtvS)              | 0.8522 | 0.9475 |           52.3823            |
|                [Bert](https://arxiv.org/pdf/1810.04805.pdf)                 | 0.8714 | 0.9544 |           61.2800            |
|                [BIMPM](https://arxiv.org/pdf/1702.03814.pdf)                | 0.8359 | 0.9375 |           18.8210            |
|        [DecomposableAttention](https://arxiv.org/pdf/1606.01933.pdf)        | 0.8068 | 0.9334 |            3.7170            |
|             [DistilBert](https://arxiv.org/pdf/1910.01108.pdf)              | 0.8450 | 0.9403 |           31.1680            |
|                [ESIM](https://arxiv.org/pdf/1609.06038.pdf)                 | 0.8385 | 0.9311 |            2.7410            |
|            [RE2](https://www.aclweb.org/anthology/P19-1465.pdf)             | 0.8391 | 0.9196 |            5.2200            |
|               [Roberta](https://arxiv.org/pdf/1907.11692.pdf)               | 0.8726 | 0.9591 |           61.3130            |
| [SiaGRU](https://aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12195/12023) | 0.8281 | 0.9336 |            3.5500            |
|                [XlNet](https://arxiv.org/pdf/1906.08237.pdf)                | 0.8694 | 0.9601 |           89.8090            |

+ 参考
部分模型，借鉴了  
https://github.com/alibaba-edu/simple-effective-text-matching-pytorch  
https://github.com/pengshuang/Text-Similarity  
https://github.com/MachineLP/TextMatch
等项目

+ #TODO
  + utils 整理
