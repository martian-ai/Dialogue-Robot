from snownlp import SnowNLP
from solutions.nlu.lexical import seg, pos, ner
from solutions.nlu.syntax import dp

def sentence_summary(sentences):
    s = SnowNLP(sentences)
    print(s.summary(2))

def sentence_simple(words, rely_id):

    class Node(object):
        def __init__(self, head, idx):
            self.head = head
            self.idx = idx
            self.next = None

    class Tree(object):
        def __init__(self):
            self.root = None
        def build(self, input_list):

            if self.root is None:
                self.root = Node(words[rely_id.index(-1)], input_list.index(-1))
            
            all_node_list = [self.root]
            while len(all_node_list) > 0:
                pop_node = all_node_list.pop(0)
                tmp_idx = pop_node.idx
                node_list = [Node(words[idx], idx) for idx, val in enumerate(input_list) if val == tmp_idx]
                pop_node.next = node_list
                # if node_list is None:
                #     continue
                # else :
                all_node_list.extend(pop_node.next)
        def traverse(self):  # 层次遍历
            if self.root is None:
                return None
            all_node_list = [self.root]
            res = [self.root.head]
            while len(all_node_list) > 0:
                pop_node = all_node_list.pop(0)
                if pop_node.next is not None:
                    all_node_list.extend(pop_node.next)
                res.append([tmp.head for tmp in pop_node.next])

            return res

    root = Tree()
    rely_id = [ item - 1 for item in rely_id ]
    root.build(rely_id)

    results = root.traverse()
    tmp_results = []
    for item in results:
        if type(item) is list:
            tmp_results.extend(item)
        else:
            tmp_results.append(item)

    cut_number = int(len(tmp_results)/2) if len(tmp_results) > 12 else 6
    save_words = tmp_results[:cut_number]

    from jieba import analyse
    textrank = analyse.textrank
    key_words = textrank(''.join(words), topK=3)
    for item in key_words:
        save_words.append(item)

    results_words = [ item for item in words if item in save_words]

    return results_words

def long_sentence_compression_by_syntax_delete_att(text):
    res = []
    words, pos, _ = lexical_analysis(text)
    arcs = parser.parse(words, pos)  # 句法分析
    # print([(arc.head, arc.relation) for arc in arcs])
    relations = [arc.relation for arc in arcs]
    rely_ids = [arc.head for arc in arcs]# 提取依存父节点id
    skip = False

    NO_SKIP_WORDS = ['本', '该', '不', '本办法', '本法规', '本提案', '本条例', '违反', '违背']
    has_law = False
    has_coo = False
    id_word_syn_att = {}
    for _id, (word, rel, rely_id) in enumerate(zip(words, relations, rely_ids)):
        id_word_syn_att[_id+1] = (word, rel, rely_id)

    for _id, (word, rel, rely_id) in enumerate(zip(words, relations, rely_ids)):
        # print(_id+1, word, rel, rely_id)
        """
            保证《xx法规》不会被删去
        """
        has_skip_content = False
        if word == '（':
            has_skip_content = True
        elif word == '）':
            has_skip_content = False
            continue
        if has_skip_content:
            continue
        if word == '《':
            has_law = True
        elif word == '》':
            has_law = False
        if not has_law:
            if rel == 'SBV':
                skip = True
            if (rel =='ATT' or rel == 'RAD') and skip and word not in NO_SKIP_WORDS :
                continue
            # if word == '、' or rel == 'LAD' or (rel=='COO'):
            if word == '、' or rel == 'LAD':
                continue
            if rel == 'COO':
                parent_rel = rel
                parent_id = _id+1
                while parent_rel == 'COO':
                    p_word, parent_rel, parent_id = id_word_syn_att[parent_id]
                if parent_rel in ['ATT', 'VOB', 'FOB']  :
                    continue
        res.append(word)
        # res.append('，')
    return ''.join(res)

def long_sentence_compression_by_syntax_keywords(input):
    cut_text = input.split('，')
    all_results = []
    try:
        for text in cut_text:
            words = seg(text)
            pos = pos(text)
            arcs = dp(text, words, pos) # TODO
            rely_id = [arc.head for arc in arcs]# 提取依存父节点id
            tmp_results = sentence_simple(words, rely_id)
            all_results.append(''.join(tmp_results))

    except:
        return input
    
    return '，'.join(all_results)