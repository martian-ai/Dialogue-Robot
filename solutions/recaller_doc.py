from elasticsearch import Elasticsearch

from modules.basic_search.recall.recaller import EsRecaller


class DocRecaller(EsRecaller):
    def __init__(self, ip, port):
        super.__init__(ip, port)

    def search_by_qq(self, query, index, match_item, search_size):
        res = self.search_by_match(query, index, match_item, search_size)
        score_list, question_list = []
        for hit in res:
            score_list.append(hit['_score'])
            question_list.append(hit['_source']['quesiton'])
        return(question_list)
    
    def search_by_qa(self, query, index, match_item, search_size):
        res = self.search_by_match(query, index, match_item, search_size)
        score_list, answer_list = []
        for hit in res:
            score_list.append(hit['_score'])
            answer_list.append(hit['_source']['answer'])
        return(answer_list)


if __name__ == '__main__':
    faq_recaller = DocRecaller(ip='10.92.22.160', port='9200')
    faq_recaller.indices.get_alias("*")
    faq_recaller.search_by_qq('你好', 'doc0', "question", 3)






# 插叙
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


def es_search(search_query, search_size, es_index_name):
	es.indices.refresh(index=es_index_name)
	res = es.search(index=es_index_name, size=search_size, body={"query": {"bool": {'should':[{'match':{'text':search_query}}]}}})
	return res

def get_search_results(query, search_size, search_engine='offline-es', es_index='ES_INDEX_DOC_ThreeBody'):

    if search_engine == 'offline-es':
        results_search = es_search(query, search_size, es_index)
        score_list, text_list = [], []
        for idx, item in enumerate(results_search['hits']['hits']):
            print("### Search item  ", idx, "  #"*20)
            print(item['_score'])
            print(item['_source']['text'])
            score_list.append(item['_score'])
            text_list.append(item['_source']['text'])
        return score_list, text_list


def build_doc_es(index_name):
    # index_name = app.config['ES_INDEX_DOC_ThreeBody']
	path = 'resources/corpus/document/三体.txt'
	lines = get_data(path)
	paras = paragraph_chunking(lines, 384)
	es_insert(paras, index_name)
	print(es_search('云天明送给程心的星球叫什么名字', index_name))
    # return {'statue':200}
	#es_delete(ES_TEST_INDEX)
    # Todo 返回函数设计