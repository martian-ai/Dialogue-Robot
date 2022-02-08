"""
Copyright (c) 2022 Martian.AI, All Rights Reserved.

main application interface for Dialogue Robot BotMVP

Authors: apollo2mars(apollo2mars@gmail..com)
"""

from collections import OrderedDict
from typing import List

from modules.structure_paragraph.keysentence.pipeline import \
    paragraph_keysentences

# from solutions.summarizer.pipeline import importance_sentence_select
from solutions.service_docqa.pipeline import docQuestionGeneration, docQuestionAnswering

# from modules.discourse.extractor.phrases.pipeline import extract_phrase_ckpe


def sentence_answer_select_simple(sentence):
    answers = [sentence]  # 几种答案拼接
    return answers


def sentence_answer_select(sentence):
    # keywords_answers = [item[0] for item in doc_keywords(sentence, topK=5)]  # 句子的关键词
    # keyphrases_ckpe_answers = paragraph_keysentences(sentence, topK=5)  # 句子的关键短语
    # keyphrases_dp_answers = extract_phrase_dp(tmp_sentence, topK=5) # 句法
    # retall_senteneces = retall(tmp_sentence) # back translate
    # answers = keywords_answers + keyphrases_ckpe_answers + keyphrases_dp_answers + [tmp_sentence] + retall_senteneces # 几种答案拼接
    answers = [sentence]  # 几种答案拼接
    # answers = keywords_answers + keyphrases_ckpe_answers  + [sentence]  # 几种答案拼接
    # 添加其他答案筛选逻辑
    # answers = [item for item in answers if len(item) > 3]
    return answers


class BaseDoc(object):
    """Basic Document Data Strucute and Algorithms.

    Args:
        object (_type_): _description_
    """
    def __init__(self, text: str, summary: list, keysents: list, keywords: list, faqs: list, kb: object) -> None:
        """_summary_

        Args:
            text (str): _description_
            summary (list): _description_
            keysents (list): _description_
            keywords (list): _description_
            faqs (list): _description_
            kb (object): _description_
        """
        super().__init__()
        self.text = text
        self.summary = summary  # Title
        self.keysents = keysents
        self.keywords = keywords
        self.faqs = faqs

    def set_summary(self):
        """Set summary by origin text or key sentences."""
        tmp = ''
        if len(self.text) < 500:  # TODO 500 or 512
            tmp = self.text
        else:
            tmp = ' '.join(self.keysents)[:500]
        self.summary = tmp

    def set_summary(self, sum):
        """Set summary by input sum.

        Args:
            sum (_type_): Input sum
        """
        self.summary = sum

    def set_keysents(self, keysents):
        """Set Key Sentence.

        #TODO check keysents in not null

        Args:
            keysents (_type_): _description_
        """
        self.keysents = keysents

    def set_keywords(self, keywords):
        """Set key Words.

        #TODO check key word is not null

        Args:
            keywords (_type_): _description_
        """
        self.keywords = keywords

    def get_sum(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if self.summary:
            return self.summary
        else:
            self.set_summary()
            return self.summary

    def doc_qa(self, question: str, topK=3)-> List:
        """_summary_

        Args:
            question (str): _description_
            topK (int, optional): _description_. Defaults to 3.

        Returns:
            List: _description_
        """
        res = docQuestionAnswering(question)
        print(res)

        # TODO: 测试 doc_qa doc_qg 移动出去


    def doc_qg(self, answer, topK=3):
        """
        """
        """
        Way II
        提供问题{当前需要额外输入}
        根据问题生成答案
        """
        # qa_results = []

        # for tmp_sentence in select_sentences:
        #     candidate_questions = sentence_question_select(tmp_sentence)

    def doc_qa_mining(self, topK=3):
        # TODO key sentences 和 summary 如何 使用
        select_sentences = paragraph_keysentences(self.text)

        print(select_sentences)
        self.set_keysents(select_sentences)
        self.set_summary("".join(select_sentences))

        """
        Way I
        选择重要的句子，通过规则进行生成
        TODO
            指代消解(摘要后指代消解)
        """

        """
        Way III
        提供答案{重要句子，重要短语，重要单词}
        根据提供的答案生成问题
        """
        qg_results = []

        for tmp_sentence in select_sentences:
            candidate_answers = sentence_answer_select_simple(tmp_sentence)
            print(candidate_answers)
            for answer in candidate_answers:  # 返回所有结果
                question = docQuestionGeneration(self.summary, answer)
                tmp_result = OrderedDict()
                if answer in question: # if question contain answer, pass mininig
                    pass
                else:
                    tmp_result['context'] = self.summary
                    tmp_result['retall_context'] = ''
                    tmp_result['context_length'] = len(tmp_result['context'])
                    tmp_result['question'] = question
                    tmp_result['question_score'] = -1.0
                    tmp_result['question_ppl'] = -1.0
                    tmp_result['answer'] = answer
                    tmp_result['answer_score'] = -1.0
                    tmp_result['answer_ppl'] = -1.0
                    tmp_result['question_answer_match_score'] = -1.0
                    qg_results.append(tmp_result)

        all_results = qg_results

        all_faqs = [[item['question'], item['answer']]for item in all_results]

        """
        qa match 
        """
        faqs = all_faqs  # TODO

        """
        文档 faqs 更新
        """
        self.set_qa_pairs(faqs)

    def set_qa_pairs(self, faqs):
        self.faqs = faqs

    def get_qa_paris(self):
        if self.faqs:
            return self.faqs
        else:
            self.doc_qa_mining()
            return self.faqs
    
    def upload_qa_pairs(self):
        pass

