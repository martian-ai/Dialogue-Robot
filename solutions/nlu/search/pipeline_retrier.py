import pickle
import numpy as np
import sys, os
from bert_serving.client import BertClient
from annoy import AnnoyIndex
from elasticsearch import RequestsHttpConnection, Elasticsearch
from elasticsearch.helpers import bulk

"parameters for es"
es_ip = ''
es_user = ''
es_pwd = ''
es_port = ''
es_document_size = 5
es_index_name = 'jd_index_name' # ES 6.3
es = Elasticsearch([es_ip],http_auth=(es_user, es_pwd), port=es_port)
" bert-as-service init"
bc_embedding_path = '../../resources/clue-pair'
if os.path.exists(bc_embedding_path) is False:
     raise RuntimeError('embedding file is not exist')
os.system('bert-serving-start -model_dir ' + bc_embedding_path + ' -num_worker=4 ')
bc_embedding_size = 312
bc_document_size = 5
bc = BertClient()
" annoy parameters"
annoy_tree_number = 100

"""
Method 1 : Literal Search by Elastic Search
"""

def elastic_search_by_qq(input_text):
    """
    use in dialogue system
    use elastic search to make query-question pair retrieval 
    """
    doc = {"question": {"bool": {'should':[{'match':{'question':input_text}}]}}}
    searched = es.search(index=es_index_name, body=doc, size=es_document_size)
    document_list = []
    for hit in searched['hits']['hits']:
        document_list.append(hit['_source']['document'])
    return(document_list)

def elastic_search_by_qa(input_text):
    """
    use in dialogue system
    use elastic search to make query-answer pair retrieval 
    """
    # doc = {"question":{'match_parse':{'documents':input_text}}} # 精确匹配
    doc = {"question":{'match':{'documents':input_text}}} # 精确匹配
    searched = es.search(index=es_index_name, body=doc, size=es_document_size)
    document_list = []
    for hit in searched['hits']['hits']:
        document_list.append(hit['_source']['document'])
    return(document_list)

"""
Method 2 : Semantic Search based on Embedding Vector
sentence represent by bert-as-service
used in short-short text similarity
"""

def text_encode(input_text):
    return bc.encode([input_text])[0]

def sentences_encode(input_list):
    return bc.encode(input_list)

def paragraphs_query_encode(input_list, input_term):
    """
    todo : how to represent paragraph
    current solution : mean of all sentence in current parapraph
    """
    return np.mean([ sentences_encode(item) for item in input_list], axis=1)

def annoy(encode_list, encode_item, text_list):
    ann = AnnoyIndex(bc_embedding_size, 'angular') # 312 为向量维度
    for idx, item in enumerate(encode_list):
        embd = item.tolist() 
        ann.add_item(idx, embd)
    ann.build(annoy_tree_number )
    idx_list = ann.get_nns_by_vector(encode_item, bc_document_size)
    return [ text_list[tmp_idx ]for tmp_idx in idx_list ]

def eculd(encode_list, encode_item, text_list):

    def distEclud(vecA, vecB):
        return np.sqrt(sum(np.power(vecA - vecB, 2)))
    score_list = []
    for value in encode_list:
        score_list.append(distEclud(encode_item, value))

    zipped = zip(score_list, text_list)
    sorted_zipped = np.asarray(sorted(zipped, reverse=False, key=lambda x: (x[0], x[1])))
    return sorted_zipped[:bc_document_size][:, 1]

"""
Method 3 : Semantic Search based on Topic Model and Embedding Vector
short text represent by bert-as-service
long text represent by topic model (LDA/LSI)
used in short-short text similarity
"""


"""
Method 4: Semantic Search based on Topic Model
long text represent by topic model(LDA/LSI)
used in long-long text similarity
"""


if __name__ == "__main__":
    query = '你喜欢打篮球吗'
    sentence_list = ['乔丹打球厉害吗', '姚明是谁', '火箭队的主场在哪里', '怎么了', '吃了没']
    paragraph_list = ['', '']

    """
    Method 1 test
    """
    # pass

    """
    Method 2 
    """
    query_encode = text_encode(query)
    sentences_encode = sentences_encode(sentence_list)

    print(eculd(sentences_encode, query_encode, sentence_list))
    # ['火箭队的主场在哪里', '乔丹打球厉害吗', '姚明是谁', '怎么了', '吃了没']
    print(annoy(sentences_encode, query_encode, sentence_list))
    # ['火箭队的主场在哪里', '乔丹打球厉害吗', '姚明是谁', '怎么了', '吃了没']