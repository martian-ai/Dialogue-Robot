import sys
sys.path.append('../..')

from solutions.qg.utils import line_cleanser, safe_add
from solutions.nlu.lexical import lexical_analysis
from solutions.nlu.semantic import srl


def question_generate_by_semantic(line, exist_questions=set(), law=''):
    """
    TODO
    """
    questions = []
    answers = []
    clean_line = line_cleanser(line)
    words, pos, ner = lexical_analysis(clean_line)
    roles = srl(clean_line, words, pos)
    prefix = '' if not law else '根据《'+law+'》，'

    for role in roles:
        arg_names = [role.name for role in role.arguments]
        if arg_names == ['A0'] or set(arg_names) == set(['A0', 'A1']):
            for arg in role.arguments:
                question = clean_line.replace(''.join(words[arg.range.start:arg.range.end+1]), '谁/哪个部门')+'?'
                if arg.name == 'A0' and question not in questions:
                    question = prefix+question
                    safe_add(question, line, exist_questions, questions, answers)
        for arg in role.arguments:
            before = ''.join(words[0:arg.range.start]) 
            after = ''.join(words[arg.range.end+1::]) 
            answer = ''.join(words[arg.range.start:arg.range.end+1])
            if arg.name == 'A1' and len(answer) >= 4:
                replace = ['什么']
            elif arg.name == 'TMP':
                replace = ['什么时候']
            elif arg.name == 'MNR':
                replace = ['以什么方式', '如何', '怎么']
            elif arg.name == 'LOC':
                replace = ['在哪里', '于何处']
            else:
                continue
            if replace:
                for _replace in replace:
                    tmp_results = before + _replace + after + '?'
                    if tmp_results not in questions:
                        question = tmp_results
                        safe_add(question, line, exist_questions, questions, answers)
    return questions, answers

# def question_generate_by_semantic_1(text):
#     """
#     利用语义角色标注确定 主动词
#     利用依存句法分析确定 主语 和 谓语
#     """
#     words, pos, _ = lexical_analysis(text)
#     arcs = syntax_analysis(text, words, pos) # TODO
#     roles = srl(text, words, pos)

#     roles_index = [role.index for role in roles]

#     rely_id = [arc.head for arc in arcs]# 提取依存父节点id
#     relation = [arc.relation for arc in arcs]# 提取依存关系
#     heads = ['Root' if id ==0 else words[id-1]for id in rely_id]# 匹配依存父节点词语

#     for tmp in roles_index:
#         subject, predicate, object = '', '', ''
#         for i in range(len(words)):
#             if relation[i] == 'VOB' and heads[i] == words[tmp]:
#                 object = words[i]
#                 print(relation[i] +'(' + words[i] +', ' + heads[i] +')')
#             elif relation[i] == 'SBV' and heads[i] == words[tmp]:
#                 subject = words[i]
#                 print(relation[i] +'(' + words[i] +', ' + heads[i] +')')
#             if len(subject) > 0 and len(object) > 0 :
#                 predicate = words[tmp]
#                 print(subject + '###' + predicate + '###' + object)
