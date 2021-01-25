from solutions.nlu.lexical import seg, pos, ner
from solutions.nlu.syntax import dp
from solutions.nlu.semantic import srl
from collections import Counter

def get_dict(word_list, char_list = None):
    word_dict = {value:key for key,value in enumerate(word_list) }
    if char_list:
        char_dict = {value:key for key,value in enumerate(char_list) }

    all_dict = Counter(word_dict) + Counter(char_dict)
    
    pos_dict = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'g':5, 'h':6, 'i':7, 'j':8, 'k':9,
                'm':10, 'n':11, 'nd':12, 'nh':13, 'ni':14, 'nl':15, 'ns':16, 'nt':17, 'nz':18, 'o':19,
                'p':20, 'q':21, 'r':22, 'u':23, 'v':24, 'wp':25, 'ws':26, 'x':27, 'x':18 } 
    ner_dict = {'O':0, 'B-Nh':1, 'I-Nh':2, 'E-Nh':3, 'S-Nh':4, 'B-Ns':5, 'I-Ns':6, 'E-Ns':7, 'S-Ns':8,
                'B-Ni':9, 'I-Ni':10, 'E-Ni':11, 'S-Ni':12}
    relation_dict = {'SBV':0, 'VOB':1, 'IOB':2, 'FOB':3, 'DBL':4, 'ATT':5, 'ADV':6, 'CMP':7, 
                     'COO':8, 'POB':9, 'LAD':10, 'RAD':11, 'IS':12, 'WP':13, 'HED':14}

    return all_dict, pos_dict, ner_dict, relation_dict

def feature_extract(text, words_dict, pos_dict, ner_dict, relation_dict, max_length = 16):
    words, pos, ner = seg(text), pos(text), ner(text)
    arcs = dp(text, words, pos) 
    roles = srl(text, words, pos)

    relations = [arc.relation for arc in arcs]
    rely_ids = [arc.head for arc in arcs]# 提取依存父节点id
    # roles_index = [role.index for role in roles]

    if len(words) > max_length :
        words = words[:max_length]
        pos = pos[:max_length]
        ner = ner[:max_length]
        # arcs = arcs[:max_length]
        relations = relations[:max_length]
        rely_ids = rely_ids[:max_length]
        # roles_index = roles_index[:max_length]
    else:
        padding_length = max_length - len(words)
        words = words + [-1]*padding_length
        pos = pos + [-1]*padding_length
        ner = ner + [-1]*padding_length
        # arcs = arcs + [-1]*padding_length
        relations = relations + [-1]*padding_length
        rely_ids = rely_ids + [-1]*padding_length
        # roles_index = roles_index + [-1]*padding_length

    words = [ word_dict[item] for item in words]
    pos = [ pos_dict[item] if item is not -1 else -1 for item in pos ]
    ner = [ ner_dict[item] if item is not -1 else -1 for item in ner ]
    relations = [ relation_dict[item] if item is not -1 else -1 for item in relations ]

    tmp_results = [words, pos, ner, relations, rely_ids] 
    results = []
    for item in tmp_results:
        results.extend(item)
    return results