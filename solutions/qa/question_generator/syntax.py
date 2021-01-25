import re
import jieba
from utils import line_cleanser, safe_add

def question_generate_by_syntax(line, exist_questions, law=''):
    questions = []
    answers = []
    
    clean_line = line_cleanser(line)
    words, pos, ner = lexical_analysis(clean_line)
    prefix = '' if not law else '根据《'+law+'》，'
    if '本办法' in line or '本法规' in line or '本提案' in line or \
        '本条例' in line or '本规定' in line or '该办法' in line or '该法规' \
        in line or '该提案' in line or '该条例' in line or '该规定' in line:
        prefix = ''
    if '应当' in words:
        first, second = clean_line.split('应当')[:2]
        first_arcs = syntax_analysis(first, words, pos)

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
        second_arcs = syntax_analysis(second, words, pos)
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

    roles = semantic_analysis(clean_line, words, pos)

    for role in roles:
        arg_names = [role.name for role in role.arguments]
        if arg_names == ['A0'] or set(arg_names) == set(['A0', 'A1']):
            for arg in role.arguments:
                question = clean_line.replace(''.join(words[arg.range.start:arg.range.end+1]), '谁/哪个部门')+'?'
                # question = clean_line.replace(''.join(words[arg.range.start:arg.range.end+1]), '###谁/哪个部门###')+'?'
                if arg.name == 'A0' and question not in questions:
                    question = prefix+question
                    safe_add(question, line, exist_questions, questions, answers)

        for arg in role.arguments:

            before = ''.join(words[0:arg.range.start]) 
            after = ''.join(words[arg.range.end+1::]) 
            answer = ''.join(words[arg.range.start:arg.range.end+1])
            if arg.name == 'A1' and len(answer) >= 4:
                # replace = ['###什么###']
                replace = ['什么']
            # elif arg.name == 'A2':
            ##外国投资者、外商投资企业因违反信息报告义务受到商务主管部门行政处罚的，商务主管部门可将相关情况在外商投资信息报告系统公示平台上予以公示，并按照国家有关规定纳入信用信息系统。
            #     replace = ['###怎么办###', '###如何处理###']
            elif arg.name == 'TMP':
                # replace = ['###什么时候###']
                replace = ['什么时候']
            elif arg.name == 'MNR':
                # replace = ['###以什么方式####', '###如何###', '###怎么###']
                replace = ['以什么方式', '如何', '怎么']
            elif arg.name == 'LOC':
                # replace = ['###在哪里####', '###于何处###']
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
