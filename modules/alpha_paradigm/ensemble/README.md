# Summary of Ensemble Learning

## 集成学习理论

+ 强弱学习器
  + 打靶
+ 计算学习理论

## 基学习器

### 决策树

+ 分类树
  + 信息增益
  + 信息熵
  + 基尼系数
+ 回归树
  + 回归树 squre loss 的缺点 https://zhuanlan.zhihu.com/p/42740654
+ 停止条件
+ 减枝
+ 常见的树
  + ID3
  + C4.5
  + CART
    + 分类树 Gini 系数
    + 回归树 最小二乘

### 模型

+ 其他模型

## 组合策略

### Bagging

### Boosting

### Mixture of Experts

### Bayesian Optimial Classifier

### Bayesian Model Averaging

### Bayesian Model Combination

### Bucket of Models

### Stacking

![20200627154219](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/20200627154219.png)

+ https://zhuanlan.zhihu.com/p/26890738

## 常见的集成模型

### Random Forest

### Adaboost

![20200620161345](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/20200620161345.png)

### GBDT

+ 以CART为基分类器
+ Gradient Boosting for Regression
![20200620162612](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/20200620162612.png)

+ Gradient Boosting for Classification
  + https://zhuanlan.zhihu.com/p/46445201
![20200620171939](https://blog-picture-new.oss-cn-beijing.aliyuncs.com/dialog/20200620171939.png)

### XGBoost

+ GBDT 以CART作为基分类器， xgboost 还支持线性分类器(#TODO)， 带有L1 和 L2 正则化项
+ GBDT 只用到一阶导数信息，xgboost 用二阶泰勒展开，同时使用一阶和二阶导数
+ xgboost 在代价函数里加入了正则项，用于控制模型复杂度
+ shrinkage 学习速率
+ 列抽样 与随机森林类似，可以降低过拟合，并且提高速度
+ 缺失值 自动学习分裂方向(#TODO)
+ 特征粒度上并行 xgboost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量
+ 近似直方图，加快计算速度
+ https://xgboost.readthedocs.io/en/latest/tutorials/model.html

### LightGBM