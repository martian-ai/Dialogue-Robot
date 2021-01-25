# -*- coding:utf-8 -*-
# https://www.jianshu.com/p/9caef967ec0a
import xgboost as xgb

# 数据格式 : 第一列是 label，后面的特征，用 DMatrix。
# DMatrix有set_group方法，调用设置 groupId。(groupId 的概念在 rank 中广泛适用，只有同一个 group 中的样本才有排序的意义。对于IR任务来说，不同 query对应不同group。)

from xgboost import DMatrix, train

import pickle


def data_generation(input_dict):
    """
    data generation
    :param input_dict:
    :return: group_list, data_list, target_list
    """
    group_list = []
    data_list = []
    target_list = []

    for (k, v) in input_dict.items():
        label_and_response_list = v
        group_list.append(len(label_and_response_list))

        for item in label_and_response_list:
            target_list.append(item[0])
            data_list.append(item[1:])

    return group_list, data_list, target_list


with open('output/feature_extract_train.pkl', 'rb') as f:
    train_dict = pickle.load(f)

with open('output/feature_extract_eval.pkl', 'rb') as f:
    eval_dict = pickle.load(f)

with open('output/feature_extract_test.pkl', 'rb') as f:
    test_dict = pickle.load(f)

print("data load done!!!")

xgb_rank_params2 = {
    'bst:max_depth': 5,  # 构建树的深度，越大越容易过拟合
    'bst:eta': 0.1,  # 如同学习率
    'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    'objective': 'rank:pairwise',
    'nthread': 8,  # cpu 线程数
    'eval_metric': 'ndcg@10-',
    'metric': 'ndcg@10-'
}

train_group_list, train_data_list, train_target_list = data_generation(train_dict)
xgbTrain = DMatrix(train_data_list, label=train_target_list)
xgbTrain.set_group(train_group_list)


eval_group_list, eval_data_list, eval_target_list = data_generation(eval_dict)
xgbEval = DMatrix(eval_data_list, label=eval_target_list)
xgbEval.set_group(eval_group_list)

# get evallist
evallist = [(xgbTrain, 'train'), (xgbEval, 'eval')]

# train
rankModel = train(xgb_rank_params2, xgbTrain, num_boost_round=5, evals=evallist)

# get predict
test_group_list, test_data_list, test_target_list = data_generation(test_dict)
xgbTest = DMatrix(test_data_list, label=test_target_list)
xgbTest.set_group(test_group_list)

# predict
results = rankModel.predict(xgbTest)
print(results)

import numpy

print(numpy.mean(results))
print(numpy.var(results))

with open('xgb_output/test_result.pkl', 'wb') as f:
    pickle.dump(results, f)


# xgb_rank_params1 = {
#     'booster': 'gbtree',
#     'eta': 0.1,
#     'gamma': 1.0,
#     'min_child_weight': 0.1,
#     'objective': 'rank:pairwise',
#     'eval_metric': 'merror',
#     'max_depth': 6,
#     'num_boost_round': 10,
#     'save_period': 0
# }