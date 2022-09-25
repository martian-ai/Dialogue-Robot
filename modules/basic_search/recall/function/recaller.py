
from elasticsearch import Elasticsearch, helpers


class EsRecaller(object):
    def __init__(self, ip, port):
        self.es = Elasticsearch(hosts=ip, port=port) # TODO 后续可以换成 集群
    def create_mapping(self, index, mapping):
        """  build es init mapping

        Args:
            index (string): _description_
            mapping (json):  es mapping, demo like {"mappings": { "_source": { "enabled": true }, "properties": { "botid": { "type": "keyword" }, "question": { "type": "text" }}}}

        """
        res = self.es.indices.create(index = index, body =mapping)
        return res

    def show(self):
        return self.es.indices.get_alias("*")

    def delete(self, index):
        self.es.delete_by_query(index=index, body={"query": {"match_all": {}}})

    def search_by_match(self, query, index, match_item, size=3):
        self.es.indices.refresh(index=index)
        res = self.es.search(index=index, size=size, body={"query": {"bool": {'should':[{'match':{match_item:query}}]}}})
        return res['hits']['hits']

    # def search_by_must(self, query, index, match_item, size=3):
        # self.es.indices.refresh(index=index)
        # res = self.es.search(index=index, size=size, body={"query": {"bool": {'should':[{'match':{match_item:query}}]}}})
        # #res = self.es.search(index=index, size=size, body={"query": {"bool": {'should':[{'match':{'text':query}}]}}})
        # return res

    def insert(self, items, index):
        for item in items:
            self.es.index(index=index, body=item)

    def insert_bulk(self, index, items):
        actions = []
        for item in items:
            actions.append(item)
        helpers.bulk(self.es, actions)
