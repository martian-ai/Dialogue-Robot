from elasticsearch import Elasticsearch

es = Elasticsearch()

# 插入
def es_insert(items, es_index_name):
    for item in items:
        item_list = item.split("\t")
        doc = { 'question': item_list[0], 'answer': item_list[1]}  # 插入结构 'question', 'answer'
        es.index(index=es_index_name, body=doc)

# 删除
def es_delete(es_index_name):
	print(es.indices.get_alias("*"))
	es.delete_by_query(index=es_index_name, body={"query": {"match_all": {}}})

# 修改

# 查询
def es_search(search_query, search_size, es_index_name):
	es.indices.refresh(index=es_index_name)
	res = es.search(index=es_index_name, size=search_size, body={"query": {"bool": {'should':[{'match':{'question':search_query}}]}}}) # 匹配 'question'
	return res

def get_search_results(query, search_size, search_engine='offline-es', es_index='ES_INDEX_Chitchat_V1'): 
    if search_engine == 'offline-es':
        results_search = es_search(query, search_size, es_index)
        score_list, text_list = [], []
        for idx, item in enumerate(results_search['hits']['hits']):
            print("### Search item  ", idx, "  #"*20)
            print(item['_score'])
            print(item['_source']['question']) # 匹配 question
            score_list.append(item['_score'])
            text_list.append(item['_source']['question']) # 匹配 question
        return score_list, text_list

