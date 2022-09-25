
import sys

sys.path.append('/Users/sunhongchao/Documents/Bot')  # TODO

from modules.basic_search.recall.function.recaller import EsRecaller


class FAQRecall(EsRecaller):
    def __init__(self, ip, port):
        super(FAQRecall, self).__init__(ip, port)

    def search_by_qq(self, query, index, match_item, search_size):
        res = self.search_by_match(query, index, match_item, search_size)
        score_list, question_list, answer_list = [], [], []
        for hit in res:
            score_list.append(hit['_score'])
            question_list.append(hit['_source']['question'])
            answer_list.append(hit['_source']['answer'])
        return(score_list, question_list, answer_list)

    def search_by_qa(self, query, index, match_item, search_size):
        res = self.search_by_match(query, index, match_item, search_size)
        score_list, answer_list = []
        for hit in res:
            score_list.append(hit['_score'])
            answer_list.append(hit['_source']['answer'])
        return(answer_list)

def pipeline():
    # init
    faq_recaller = FAQRecall(ip='10.9.227.9', port='8200')
    # create mapping
    mapping = {"mappings": {"_source": {"enabled": True}, "properties": {"botid": {"type": "keyword"}, "question": {"type": "text"}}}}
    bot_id = 0
    index = "demo" + str(bot_id)
    faq_recaller.create_mapping(index, mapping)
    # insert
    items = []
    with open('resouces/corpus/solutions/faq/demo.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            _bot_id, _question_id, _question, _answer = line.strip().split()
            body = {'bot_id': _bot_id, 'question_id': _question_id, 'question': _question, 'answer': _answer}
            items.append(body)
    faq_recaller.insert(index, items)
    # search
    faq_recaller.es.indices.get_alias("*")
    res = faq_recaller.search_by_qq('车', 'demo_0', "question", 1)
    return res


if __name__ == '__main__':
    pipeline()