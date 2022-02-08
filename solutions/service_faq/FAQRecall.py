"""
Copyright (c) 2022 Martian.AI, All Rights Reserved.

FAQ Recall 

Authors: apollo2mars(apollo2mars@gmail..com)
Date: 2022/11/22 17:23:06
"""


import sys

sys.path.append('/Users/sunhongchao/Documents/Bot')  # TODO

from modules.basic_search.recall.function.recaller import EsRecaller


class FAQRecall(EsRecaller):
    """_summary_

    Args:
        EsRecaller (_type_): _description_
    """

    def __init__(self, ip, port):
        """init"""
        super(FAQRecall, self).__init__(ip, port)

    def search_by_qq(self, query, index, match_item, search_size):
        """_summary_

        Args:
            query (_type_): _description_
            index (_type_): _description_
            match_item (_type_): _description_
            search_size (_type_): _description_
        """
        res = self.search_by_match(query, index, match_item, search_size)
        score_list, question_list, answer_list = [], [], []
        for hit in res:
            score_list.append(hit['_score'])
            question_list.append(hit['_source']['question'])
            answer_list.append(hit['_source']['answer'])
        return(score_list, question_list, answer_list)

    def search_by_qa(self, query, index, match_item, search_size):
        """_summary_

        Args:
            query (_type_): _description_
            index (_type_): _description_
            match_item (_type_): _description_
            search_size (_type_): _description_
        """
        res = self.search_by_match(query, index, match_item, search_size)
        score_list, answer_list = []
        for hit in res:
            score_list.append(hit['_score'])
            answer_list.append(hit['_source']['answer'])
        return(answer_list)


@FAQRecall  # 这个语法糖相当于Student = SingleTon(Student),即Student是SingleTon的实例对象
class SingleTonRecallerFAQ:
    """_summary_
    """
    def __init__(self, es_ip, es_port, ):
        self.ip = es_ip
        self.port = es_port


recallerFAQ = SingleTonRecallerFAQ(ip='127.0.0.1', port='8200')

