"""
es 相关构建情况
1. 本地mac 三体小说 index 为 test-index

ToDo
es bulk 方式添加
es 其他函数整理
"""

import sys, os
from datetime import datetime
from elasticsearch import Elasticsearch
sys.path.append("../../..")
from solutions.utils.format.document import get_data, paragraph_chunking


es = Elasticsearch()


def elastic_search_by_qq(input_text, es_index_name, es_document_size):
    """
    use in dialogue system
    use elastic search to make query-question pair retrieval 
    """
    doc = {"question": {"bool": {'should':[{'match':{'question':input_text}}]}}} # ToDo
    searched = es.search(index=es_index_name, body=doc, size=es_document_size)
    document_list = []
    for hit in searched['hits']['hits']:
        document_list.append(hit['_source']['document'])
    return(document_list)

def elastic_search_by_qa(input_text, es_index_name, es_document_size):
    """
    use in dialogue system
    use elastic search to make query-answer pair retrieval 
    """
    # doc = {"question":{'match_parse':{'documents':input_text}}} # 精确匹配
    doc = {"question":{'match':{'documents':input_text}}}
    searched = es.search(index=es_index_name, body=doc, size=es_document_size)
    document_list = []
    for hit in searched['hits']['hits']:
        document_list.append(hit['_source']['document'])
    return(document_list)

def es_insert(items, es_index_name):
    for item in items:
        doc = {
            'text': str(" ".join(item))
        }
        es.index(index=es_index_name, body=doc)

def es_delete(es_index_name):
    print(es.indices.get_alias("*"))
	# es.delete_by_query(index=app.config['ES_INDEX'], doc_type=app.config['ES_DOC_TYPE'], body={"query": {"match_all": {}}})
	# es.delete_by_query(index=app.config['ES_INDEX_PARA'], doc_type=app.config['ES_DOC_TYPE'], body={"query": {"match_all": {}}})
    es.delete_by_query(index=es_index_name, body={"query": {"match_all": {}}})

def es_search(search_query, search_size, es_index_name):
    es.indices.refresh(index=es_index_name)
    res = es.search(index=es_index_name, size=search_size, body={"query": {"bool": {'should':[{'match':{'text':search_query}}]}}})
    return res

if __name__ == "__main__":
    # ES_TEST_INDEX = 'test-index'
    ES_TEST_INDEX = 'three-body'
    path = '../../../resources/corpus/document/三体.txt'
    lines = get_data(path)
    paras = paragraph_chunking(lines, 384)
    es_insert(paras, ES_TEST_INDEX)
    print(es_search('云天明送给程心的星球叫什么名字', 3, ES_TEST_INDEX))
    #es_delete(ES_TEST_INDEX)
