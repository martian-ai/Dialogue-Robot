# TODO smoothnlp 改造为operation

import re
import jieba
import smoothnlp
import sys
sys.path.append('../..')
from solutions.qg.utils import line_cleanser, safe_add

def question_generate_by_rule(line, exist_questions=set(), law=''):
    questions = []
    answers =  []
    clean_line = line_cleanser(line)
    # regex = "(?<=。|\\s)([^。]*?)(是指|指的是)(.*?)。"
    regex = "([^。]*?)(是指|指的是)(.*?)。"
    m = re.findall(regex, clean_line)
    for (q, n, a) in m:
        question = q+n+'什么？'
        safe_add(question, a, exist_questions, questions, answers)
    regex_aim = '.*(为\w.*制定.*。)'
    aim = re.findall(regex_aim, clean_line)
    for a in aim:
        question = '为什么制定'+law
        safe_add(question, a, exist_questions, questions, answers)
        question = '制定'+law+'的目的是？'
        safe_add(question, a, exist_questions, questions, answers)
        question = '制定'+law+'的背景是什么'
        safe_add(question, a, exist_questions, questions, answers)
        question = law+'的制定目的是什么'
        safe_add(question, a, exist_questions, questions, answers)
    # xx 负责 oo (话术)
    seg_words = jieba.lcut(clean_line)
    if '负责' in seg_words and len(clean_line.split('负责')) == 2:
        role, duty = clean_line.split('负责')
        duty = duty.replace('。' , '')
        entries = smoothnlp.postag(role) # TODO smoothnlp 换成 ltp
        postags = [entry['postag'] for entry in entries]
        tags_lists = sorted(list(set(postags)))
        if tags_lists == ['NN'] or tags_lists == ['CC','NN']:
            question = '谁负责'+duty+"？"
            safe_add(question, line, exist_questions, questions, answers)
            question = '谁负责'+duty+"？"
            safe_add(question, line, exist_questions, questions, answers)
            question = duty+"由谁负责？"
            safe_add(question, line, exist_questions, questions, answers)
            if law:
                question = '在'+law+'中'+role+'起什么作用？'
                safe_add(question, line, exist_questions, questions, answers)
            else:
                question = role+'起什么作用？'
                safe_add(question, line, exist_questions, questions, answers)
    
    return questions, answers
