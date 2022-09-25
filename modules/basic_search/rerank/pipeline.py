# -*- coding:utf-8 -*-
import xgboost as xgb
import pandas as pd
import pickle
import os, sys
import numpy as np
from collections import Counter

# 数据格式 : 第一列是 label，后面的特征，用 DMatrix。
# DMatrix有set_group方法，调用设置 groupId。(groupId 的概念在 rank 中广泛适用，只有同一个 group 中的样本才有排序的意义。对于IR任务来说，不同 query对应不同group。)

from xgboost import DMatrix, train

sys.path.append('..')
from nlu.feature import feature_extract

def read_csv(path):
    df = pd.read_csv(path, header=None)
    return df

def data_generation(input_dict):
    file_path = "/Users/sunhongchao/Desktop/qg2.csv"
    df = read_csv(file_path)
    target_list = [ int(item) for item in df.iloc[:, 0][1:].tolist()]
    data_list = df.iloc[:, 1][1:].tolist()
    data_list = [feature_extract(item) for item in data_list]
    origin_list = df.iloc[:, 3][1:].tolist()
    c = Counter(origin_list)
    group_list = [c[key] for key in dict.fromkeys(origin_list).keys()]
    return group_list, data_list, target_list

def test_data_generation(text_list):
    # df = df.dropna(axis=0) 
    # print(df)
    target_list = [0]*len(text_list)
    group_list = [1]*len(text_list)
    data_list = [feature_extract(text) for text in text_list]
    return group_list, data_list, target_list

xgb_rank_params2 = {
    'bst:max_depth': 10,  # 构建树的深度，越大越容易过拟合
    'bst:eta': 0.01,  # 如同学习率
    'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    'objective': 'rank:pairwise',
    'nthread': 8,  # cpu 线程数
    'eval_metric': 'ndcg@5-',
    'metric': 'ndcg@5-'
}

def train_ranking():
    train_group_list, train_data_list, train_target_list = data_generation({})
    test_group_list, test_data_list, test_target_list = train_group_list, train_data_list, train_target_list
    eval_group_list, eval_data_list, eval_target_list = train_group_list, train_data_list, train_target_list

    xgbTrain = DMatrix(np.asmatrix(train_data_list), label=train_target_list)
    xgbTrain.set_group(train_group_list)

    xgbEval = DMatrix(np.asmatrix(eval_data_list), label=eval_target_list)
    xgbEval.set_group(eval_group_list)
    evallist = [(xgbTrain, 'train'), (xgbEval, 'eval')]

    rankModel = train(xgb_rank_params2, xgbTrain, num_boost_round=50, evals=evallist)
    rankModel.save_model('xgb.model')
    loaded_model = xgb.Booster(model_file='xgb.model')
    xgbTest = DMatrix(np.asmatrix(test_data_list), label=test_target_list)
    xgbTest.set_group(test_group_list)
    results = loaded_model.predict(xgbTest)

    with open('results.txt', mode='w', encoding='utf-8') as f:
        for item in results:
           f.write(str(item) + '\n')

def get_pairs_rank_score(loaded_model, text_list):
    test_group_list, test_data_list, test_target_list = test_data_generation(text_list)
    # print(test_group_list, '\n*******test_group_list************')
    # print(test_data_list, '\n*********test_data_list**********')
    # print(test_target_list, '\n*********test_target_list**********')
    xgbTest = DMatrix(np.asmatrix(test_data_list), label=test_target_list)
    xgbTest.set_group(test_group_list)
    results = loaded_model.predict(xgbTest)
    return results

if __name__ == "__main__":
    #train_ranking()
    xgb_model = xgb.Booster(model_file='xgb.model')

    print(get_pairs_rank_score(xgb_model, ['你好', '你好啊']))
