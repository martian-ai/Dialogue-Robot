from pyltp import SementicRoleLabeller
from solutions.nlu import label_model_path
from solutions.nlu.lexical import seg, pos, ner
from solutions.nlu.syntax import dp

labeler = SementicRoleLabeller()
labeler.load(label_model_path)

def func():
    # 词义消歧
    # 词-释义 表
    # 1. 探索语料 有语料 有监督 2. 少量语料半监督 3. 无语料 无监督
    pass

def srl(input_text:str, words=None, pos=None, show=False):
    if not words:
        words = seg(input_text)
    if not pos:
        pos = pos(input_text)
    arcs = dp(input_text)
    roles = labeler.label(words, pos, arcs)

    if show:
        for role in roles:
            print(words[role.index], end=' ') # 谓语索引
            print(role.index, "".join(["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
        
    return roles