import re
import jieba
import sys
sys.path.append('../..')

from solutions.qg.utils import line_cleanser, safe_add
from solutions.nlu.lexical import lexical_analysis
from solutions.nlu.syntax import dp

def question_generate_by_syntax(line, exist_questions=set(), law=''):
    """
    TODO
    """
    questions = []
    answers = []
    clean_line = line_cleanser(line)
    words, pos, ner = lexical_analysis(clean_line)
    prefix = '' if not law else '根据《'+law+'》，'
    if '本办法' in line or '本法规' in line or '本提案' in line or  '本条例' in line or '本规定' in line or '该办法' in line or '该法规'  in line or '该提案' in line or '该条例' in line or '该规定' in line:
        prefix = ''
    if '应当' in words:
        first, second = clean_line.split('应当')[:2]
        first_arcs = dp(first, words, pos) # 句法分析

        if 'SBV' in [arc.relation for arc in first_arcs]:
            question = prefix+first+'应当怎么办？'
            # question = prefix+first+'###应当怎么办？###'
            safe_add(question, line, exist_questions, questions, answers)
            # question = prefix+first+'###该如何处理？###'
            question = prefix+first+'该如何处理？'
            safe_add(question, line, exist_questions, questions, answers)
        pos_counter_idx = {}
        """
        对于 句子中含有应当的句子：
        如果前面只含有一个谓语，则可以讲 问句 的如何进行提前
        例如：国家治理河流应该按照本法规
        国家###应该如何###治理河流？
        """
        first_seg = []
        first_pos = []
        i = 0
        while words[i] != '应当':
            first_seg.append(words[i])
            first_pos.append(pos[i])
            i += 1
        pos = first_pos
        for i in range(len(pos)):
            pos_counter_idx[pos[i]] = (pos_counter_idx.get(pos[i], (0, i))[0]+1, i)
        if pos_counter_idx.get('v', (0,0))[0] == 1:
            idx = pos_counter_idx.get('v', 0)[1]
            if idx != 0:
                # first_seg = first_seg[:idx] + ["###应该如何###"] + first_seg[idx:] 
                first_seg = first_seg[:idx] + ["应该如何"] + first_seg[idx:] 
                question = prefix+''.join(first_seg)+'？'
                safe_add(question, line, exist_questions, questions, answers)
    elif '负责' in words:
        first, second = clean_line.split('负责')[:2]
        second_arcs = dp(second, words, pos) # 句法分析
        if 'SBV' in [arc.relation for arc in second_arcs]:
            # question = prefix+first+'####负责什么？###'
            question = prefix+first+'负责什么？'
            safe_add(question, line, exist_questions, questions, answers)
            # question = prefix+second+'###由谁/哪个部门负责？###'
            question = prefix+second+'由谁/哪个部门负责？'
            safe_add(question, line, exist_questions, questions, answers)
            # question = prefix+second+'###是哪个部门管的？###'
            question = prefix+second+'是哪个部门管的？'
            safe_add(question, line, exist_questions, questions, answers)

    return questions, answers

if __name__ == '__main__':
    question_generate_by_syntax('国务院规定农民工的工资不能拖欠')