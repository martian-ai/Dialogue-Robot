
def question_generate_by_semantic(text):
    """
    利用语义角色标注确定 主动词
    利用依存句法分析确定 主语 和 谓语
    """
    words, pos, _ = lexical_analysis(text)
    arcs = syntax_analysis(text, words, pos) # TODO
    roles = semantic_analysis(text, words, pos)

    roles_index = [role.index for role in roles]
    # print(roles_index)
    # roles_index = roles_index[0]

    rely_id = [arc.head for arc in arcs]# 提取依存父节点id
    relation = [arc.relation for arc in arcs]# 提取依存关系
    heads = ['Root' if id ==0 else words[id-1]for id in rely_id]# 匹配依存父节点词语

    for tmp in roles_index:
        subject, predicate, object = '', '', ''
        for i in range(len(words)):
            if relation[i] == 'VOB' and heads[i] == words[tmp]:
                object = words[i]
                # print(relation[i] +'(' + words[i] +', ' + heads[i] +')')

            elif relation[i] == 'SBV' and heads[i] == words[tmp]:
                subject = words[i]
                # print(relation[i] +'(' + words[i] +', ' + heads[i] +')')
            
            if len(subject) > 0 and len(object) > 0 :
                predicate = words[tmp]
                # print(subject + '###' + predicate + '###' + object)
