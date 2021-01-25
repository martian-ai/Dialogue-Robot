import codecs
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import corpora, models
import numpy as np
import jieba
import pickle

def get_dict():
    train = []
    fp = codecs.open('/export/home/sunhongchao1/Workspace-of-NLU/corpus/clf/THUCnews/test.txt', 'r', encoding='utf8') # 文本文件，输入需要提取主题的文档
    stopwords = codecs.open('/export/home/sunhongchao1/Workspace-of-NLU/resources/stopwords.txt', 'r', encoding='utf8').readlines() # 取出停用词
    for line in fp:
        line = list(jieba.cut(line))
        train.append([w for w in line if w not in stopwords])

    dictionary = Dictionary(train)
    print('get dict done')

    with open('dictionary.pkl', 'wb') as f:
        pickle.dump(dictionary, f)
    with open('train.pkl', 'wb') as f:
        pickle.dump(train, f)

def train_model():
    dictionary=pickle.load(codecs.open('dictionary.pkl'))
    train=pickle.load(codecs.open('train.pkl'))
    corpus = [ dictionary.doc2bow(text) for text in train ]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=100)
    #模型的保存/ 加载
    lda.save('test_lda.model')
#计算两个文档的相似度
def lda_sim(s1,s2):
    lda = models.ldamodel.LdaModel.load('test_lda.model')
    test_doc = list(jieba.cut(s1))  # 新文档进行分词
    dictionary = pickle.load(codecs.open('dictionary.pkl'))
    doc_bow = dictionary.doc2bow(test_doc)  # 文档转换成bow
    doc_lda = lda[doc_bow]  # 得到新文档的主题分布
    # 输出新文档的主题分布
    print(doc_lda)
    list_doc1 = [i[1] for i in doc_lda]
    print('list_doc1',list_doc1)

    test_doc2 = list(jieba.cut(s2))  # 新文档进行分词
    doc_bow2 = dictionary.doc2bow(test_doc2)  # 文档转换成bow
    doc_lda2 = lda[doc_bow2]  # 得到新文档的主题分布
    # 输出新文档的主题分布
    print(doc_lda)
    list_doc2 = [i[1] for i in doc_lda2]
    print('list_doc2',list_doc2)
    try:
        sim = np.dot(list_doc1, list_doc2) / (np.linalg.norm(list_doc1) * np.linalg.norm(list_doc2))
    except ValueError:
        sim=0
    #得到文档之间的相似度，越大表示越相近
    return sim

if __name__ == "__main__":
    get_dict()
    train_model()
    doc_1 = '北京的路上又开始堵了'
    doc_2 = '上海的路况还不错啊'
    doc_3 = '吃吃吃，胖死拉到'
    print(lda_sim(doc_1, doc_2))
    print(lda_sim(doc_2, doc_3))
    print(lda_sim(doc_1, doc_3))
