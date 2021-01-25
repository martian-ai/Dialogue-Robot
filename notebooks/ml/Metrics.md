# Summary of Evaluation

## Accuracy, Precision, Recall
+ TP, TN, FP, FN的定义
    + TP: 预测为1(Positive)，实际也为1(Truth-预测对了)
    + TN: 预测为0(Negative)，实际也为0(Truth-预测对了)
    + FP: 预测为1(Positive)，实际为0(False-预测错了)
    + FN: 预测为0(Negative)，实际为1(False-预测错了)
    + 总的样本个数为：TP+TN+FP+FN

|  |Real=1 | Real=0|
|---|---|---|
|Predict=1 | TP | FP |
|Predict=0 | FN | TN|

+ 计算
    + Accuracy = (预测正确的样本数)/(总样本数)=(TP+TN)/(TP+TN+FP+FN)
    + Precision = (预测为1且正确预测的样本数)/(所有预测为1的样本数) = TP/(TP+FP)
    + Recall = (预测为1且正确预测的样本数)/(所有真实情况为1的样本数) = TP/(TP+FN)
+ 示意图

![](https://pic4.zhimg.com/80/v2-76b9176719868e9b85bedf5192e722d3_hd.png)

+ recall是相对真实的答案而言： true positive ／ golden set 。假设测试集里面有100个正例，你的模型能预测覆盖到多少，如果你的模型预测到了40个正例，那你的recall就是40%
+ precision是相对你自己的模型预测而言：true positive ／retrieved set。假设你的模型一共预测了100个正例，而其中80个是对的正例，那么你的precision就是80%。我们可以把precision也理解为，当你的模型作出一个新的预测时，它的confidence score 是多少，或者它做的这个预测是对的的可能性是多少
+ 一般来说呢，鱼与熊掌不可兼得。如果你的模型很贪婪，想要覆盖更多的sample，那么它就更有可能犯错。在这种情况下，你会有很高的recall，但是较低的precision。如果你的模型很保守，只对它很sure的sample作出预测，那么你的precision会很高，但是recall会相对低。
 
## Demo 二分类
+ 训练一个二分类器,判断地震是否发生
+ 地震预测(假如 每10000 天有1 天发生地震)
+ 预测一百天, 假如90天预测为不发生, 10天预测为发生, 结果全都没有发生
+ 准确率 Accuracy = 90/100
+ 不发生
    + Precision 90/90
    + Recall 90/100
+ 发生
    + Precision 0/10
    + Recall 0/0   
+ accuracy paradox

## Demo 多分类

|  |Target| 预测结果| 预测结果中正确的 TP|
|---|--- |---|---|
|足球| 9  | 10| 7 |
|篮球| 10 | 10| 8 | 
|网球| 11 | 10| 9 | 

+ Accuracy (7+8+9)/(9+10+11)

+ 足球
    + Precision 7/10
    + Recall 7/9
+ 篮球
    + Precision 8/10
    + Recall 8/10
+ 网球
    + Precision 9/10
    + Recall 9/11

# ROC
+ 我们计算两个指标TPR（True positive rate）和FPR（False positive rate），TPR=TP/(TP+FN)=Recall，TPR就是召回率。FPR=FP/(FP+TN)，FPR即为实际为好人的人中，预测为坏人的人占比。我们以FPR为x轴，TPR为y轴画图，就得到了ROC曲线。

![](https://pic3.zhimg.com/80/v2-70f8ba09d21ca845a1ec0c107e41212e_hd.png)

# PR 

# Micro vs Macro Average vs Weighted

![](https://images2018.cnblogs.com/blog/1366679/201804/1366679-20180413180201734-1205637646.png)
![](https://images2018.cnblogs.com/blog/1366679/201804/1366679-20180413180147402-1119858334.png)

+ micro average 在样本不平衡的时候较为有效
