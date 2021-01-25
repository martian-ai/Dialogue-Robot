
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, Birch, MiniBatchKMeans, SpectralClustering
from sklearn.metrics import  silhouette_score, adjusted_rand_score, calinski_harabasz_score
import jieba
import math
import time
import matplotlib.pyplot as plt

segment_jieba = lambda text: " ".join(jieba.cut(text))

def feature_tfidf(lines):
    corpus = []
    for line in lines:
        line = line.strip()
        tmp = segment_jieba(line)
        corpus.append(tmp)

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # word = vectorizer.get_feature_names()
    return tfidf # 高维稀疏矩阵

def feature_bert(lines):
    pass

def kmeans(features, lines):
    # K-Means的主要优点有：
    # 1）原理比较简单，实现也是很容易，收敛速度快。
    # 2）聚类效果较优。
    # 3）算法的可解释度比较强。
    # 4）主要需要调参的参数仅仅是簇数k。
    # K-Means的主要缺点有：
    # 1）K值的选取不好把握
    # 2）对于不是凸的数据集比较难收敛
    # 3）如果各隐含类别的数据不平衡，比如各隐含类别的数据量严重失衡，或者各隐含类别的方差不同，则聚类效果不佳。
    # 4）采用迭代方法，得到的结果只是局部最优。
    # 5）对噪音和异常点比较的敏感。
    n_clusters=int(math.sqrt(len(lines)/2))
    kmeans = KMeans(n_clusters)
    kmeans.fit(features)

    labels = kmeans.labels_
    ss = silhouette_score(features.A, labels, metric='euclidean')
    ch = calinski_harabasz_score(features.A, labels)

    result = {}
    for index, label in enumerate(kmeans.labels_, 0):
        if label in result.keys():
            tmp_list = result[label]
            tmp_list.extend([lines[index]])
            result[label] = tmp_list
        else:
            result[label] = [lines[index]]
    return result, {'ss':ss, 'ch':ch }

def mini_batch_kmeans(features, lines):
    n_clusters=int(math.sqrt(len(lines)/2))
    kmeans = MiniBatchKMeans(n_clusters)
    kmeans.fit(features)
    
    labels = kmeans.labels_
    ss = silhouette_score(features.A, labels, metric='euclidean')
    ch = calinski_harabasz_score(features.A, labels)

    result = {}
    for index, label in enumerate(kmeans.labels_, 0):
        if label in result.keys():
            tmp_list = result[label]
            tmp_list.extend([lines[index]])
            result[label] = tmp_list
        else:
            result[label] = [lines[index]]
    return result, {'ss':ss, 'ch':ch }

def dbscan(features, lines):
    # eps 默认 0.5; 邻接阈值 
    # min_samples 默认 10; 最小样本数
    # DBSCAN的主要优点有：
    # 1） 可以对任意形状的稠密数据集进行聚类，相对的，K-Means之类的聚类算法一般只适用于凸数据集。
    # 2） 可以在聚类的同时发现异常点，对数据集中的异常点不敏感。
    # 3） 聚类结果没有偏倚，相对的，K-Means之类的聚类算法初始值对聚类结果有很大影响。
    # DBSCAN的主要缺点有：
    # 1）如果样本集的密度不均匀、聚类间距差相差很大时，聚类质量较差，这时用DBSCAN聚类一般不适合。
    # 2） 如果样本集较大时，聚类收敛时间较长，此时可以对搜索最近邻时建立的KD树或者球树进行规模限制来改进。
    # 3） 调参相对于传统的K-Means之类的聚类算法稍复杂，主要需要对距离阈值𝜖，邻域样本数阈值MinPts联合调参，不同的参数组合对最后的聚类效果有较大影响。

    min_samples = 50

    y_pred = DBSCAN(eps=0.5, min_samples=min_samples).fit_predict(features)
    ss = silhouette_score(features.A, y_pred, metric='euclidean')
    ch = calinski_harabasz_score(features.A, y_pred)

    result = {}
    for index, label in enumerate(y_pred, 0):
        if label in result.keys():
            tmp_list = result[label]
            tmp_list.extend([lines[index]])
            result[label] = tmp_list
        else:
            result[label] = [lines[index]]
    return result, {'ss':ss, 'ch':ch }

def birch(features, lines):
    # threshold 默认值 0.5; 最大样本半径阈值, 越小, CF Tree 建立的规模越大, 样本方差较大需要增大这个数值 
    # branching_factor 默认值 50; CF Tree 内部节点的最大CF数, 如果样本量大, 需要增大这个数值
    # n_cluster 默认值 None; 类别数
    # BIRCH算法的主要优点有：
    # 1) 节约内存，所有的样本都在磁盘上，CF Tree仅仅存了CF节点和对应的指针。
    # 2) 聚类速度快，只需要一遍扫描训练集就可以建立CF Tree，CF Tree的增删改都很快。
    # 3) 可以识别噪音点，还可以对数据集进行初步分类的预处理
    # BIRCH算法的主要缺点有：
    # 1) 由于CF Tree对每个节点的CF个数有限制，导致聚类的结果可能和真实的类别分布不同.
    # 2) 对高维特征的数据聚类效果不好。此时可以选择Mini Batch K-Means
    # 3) 如果数据集的分布簇不是类似于超球体，或者说不是凸的，则聚类效果不好。

    y_pred = Birch(threshold=0.5, branching_factor=50).fit_predict(features)
    ss = silhouette_score(features.A, y_pred, metric='euclidean')
    ch = calinski_harabasz_score(features.A, y_pred)

    result = {}
    for index, label in enumerate(y_pred, 0):
        if label in result.keys():
            tmp_list = result[label]
            tmp_list.extend([lines[index]])
            result[label] = tmp_list
        else:
            result[label] = [lines[index]]
    return result, {'ss':ss, 'ch':ch }

def spectral(features, lines):
    y_pred = SpectralClustering().fit_predict(features)
    ss = silhouette_score(features.A, y_pred, metric='euclidean')
    ch = calinski_harabasz_score(features.A, y_pred)

    result = {}
    for index, label in enumerate(y_pred, 0):
        if label in result.keys():
            tmp_list = result[label]
            tmp_list.extend([lines[index]])
            result[label] = tmp_list
        else:
            result[label] = [lines[index]]
    return result,  {'ss':ss, 'ch':ch }

if __name__ == "__main__":

    # lines = [item.strip() for item in lines.split("\n")]
    # lines = lines*10
    labels, lines = [], []
    with open('../others/evaluate/output_all.txt', mode='r', encoding='utf-8') as f:
        tmp_lines = f.readlines()
        print('origin length')
        print(len(tmp_lines))
        tmp_lines = list(set(tmp_lines))
        print('after length')
        print(len(tmp_lines))
        tmp_lines = [item.strip() for item in tmp_lines]
        for line in tmp_lines:
            labels.append(line.split('\t')[0])
            lines.append(line.split('\t')[1])
    print(labels[:5])
    print(lines[:5])

    print (time.strftime("%H:%M:%S", time.localtime()))

    # 特征提取
    features = feature_tfidf(lines)

    # print('*'*10, 'kmeans', '*'*10)
    # print (time.strftime("%H:%M:%S", time.localtime()))
    # result_kmeans = kmeans(features, lines)
    # print(result_kmeans[1])

    # print('*'*10, 'mini-batch-kmeans', '*'*10)
    # print (time.strftime("%H:%M:%S", time.localtime()))
    # result_mini_batch_kmeans = mini_batch_kmeans(features, lines)
    # print(result_mini_batch_kmeans[1])

    # print('*'*10, 'birch', '*'*10)
    # print (time.strftime("%H:%M:%S", time.localtime()))
    # result_birch = birch(features, lines)
    # print(result_birch[1])

    print('*'*10, 'dbscan', '*'*10)
    print (time.strftime("%H:%M:%S", time.localtime()))
    result_descan = dbscan(features, lines)
    print(result_descan[1])

    # print('*'*10, 'spectral', '*'*10)
    # print (time.strftime("%H:%M:%S", time.localtime()))
    # result_spectral = spectral(features, lines)
    # print(result_spectral[1])
    # print (time.strftime("%H:%M:%S", time.localtime()))

# ********** kmeans **********
# 17:42:16
# {'ss': 0.014227844166874415, 'ch': 9.324828041343949}
# ********** mini-batch-kmeans **********
# 17:44:09
# {'ss': -0.02768273319886474, 'ch': 3.503082241815677}
# ********** birch **********
# 17:45:18
# {'ss': 0.0010565242316500497, 'ch': 24.852898441142536}
# ********** dbscan **********
# 18:29:03

# ********** spectral **********
# 18:40:53
# {'ss': 0.004556970997817553, 'ch': 26.37083580960393}
# 18:42:17