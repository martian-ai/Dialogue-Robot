from collections import defaultdict
import math
import operator
import os
import jieba

"""
函数说明：特征选择TF-IDF算法
Parameters:
     list_words:词列表
Returns:
     dict_feature_select:特征选择词字典
"""
def feature_select(list_words):
    #总词频统计
    doc_frequency=defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[i]+=1
 
    #计算每个词的TF值
    word_tf={}  #存储没个词的tf值
    for i in doc_frequency:
        word_tf[i]=doc_frequency[i]/sum(doc_frequency.values())
 
    #计算每个词的IDF值
    doc_num=len(list_words)
    word_idf={} #存储每个词的idf值
    word_doc=defaultdict(int) #存储包含该词的文档数
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[i]+=1
    for i in doc_frequency:
        word_idf[i]=math.log(doc_num/(word_doc[i]+1))
 
    #计算每个词的TF*IDF的值
    word_tf_idf={}
    for i in doc_frequency:
        word_tf_idf[i]=word_tf[i]*word_idf[i]
 
    # 对字典按值由大到小排序
    dict_feature_select=sorted(word_tf_idf.items(),key=operator.itemgetter(1),reverse=True)
    return dict_feature_select
 

def test(filename):

    def check(item_1, list_1):
        for item_2 in list_1:
            if len(set(list(item_2)).intersection(set(list(item_1))))>0 :
                return True
        else:
            return False

    def get_p_r_f1(ground_list, output_list, type):
        correct = 0
        if type == 'strict':
            correct = len(set(ground_list).intersection(set(output_list)))
        elif type == 'span':
            for item_1 in output_list:
                if check(item_1, ground_list):
                    correct += 1

        if len(output_list) > 0:
            precision = correct /len(output_list)
            recall = correct / len(ground_list)
            return precision, recall
        else:
            return 0, 0

    with open(filename, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        precision_list, recall_list, f1_list = [], [], []
        for item in lines:
            cut_item = item.split('\t')
            ground_list = cut_item[1].split('###')
            inputText = cut_item[0]
            print("*"*100)
            print("text :", inputText)
            print("ground :", ground_list)
            # print(inputText)
            output_list = predict(inputText, topK=6)
            output_list = [item[0] for item in output_list]
            print("output :", output_list)
            tmp_p, tmp_r = get_p_r_f1(ground_list, output_list, 'strict')
            print("precision", tmp_p)
            print("recall", tmp_r)
            precision_list.append(tmp_p)
            recall_list.append(tmp_r)
            if tmp_p + tmp_r > 0:
                f1_list.append(2*tmp_p*tmp_r/(tmp_p + tmp_r))
            else:
                f1_list.append(0)

        return {'precision': sum(precision_list)/len(precision_list), 'recall': sum(recall_list)/len(recall_list), 'f1': sum(f1_list)/len(f1_list) }


def predict(inputText, key_words_dict, topK=5):
    cut_list = list(jieba.cut(inputText))
    results = []
    for item in cut_list:
        if item.isdigit():
            continue
        if item in key_words_dict.keys():
            results.append((item, key_words_dict[item]))
    results = list(set(results))
    sorted_results = sorted(results, key=lambda x:x[1], reverse=True)
    return sorted_results[:topK]

if __name__=='__main__':

    # 训练模型
    data_list = []
    with open("../../../resources/corpus/document/三体.txt", mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    data_list = lines[:1000]

    data_list_cut = [list(jieba.cut(item)) for item in data_list]
    features=feature_select(data_list_cut) #所有词的TF-IDF值

    with open("../../../resources/corpus/lexical/HIT_stopwords.txt", mode='r', encoding='utf-8') as f:
        lines = f.readlines() 
        stop_words = [item.strip() for item in lines]
    
    key_words_dict = {}
    for item in features:
        if len(item[0]) < 2:
            pass
        elif item[0] in stop_words:
            pass
        else:
            key_words_dict[item[0]] = item[1]

    print(predict('叶文洁的妹妹是谁', key_words_dict))