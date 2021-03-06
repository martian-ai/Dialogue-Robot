from solutions.nlu.lexical import segment, pos_tagging, name_entity
from solutions.nlu import parser_model_path
from pyltp import Parser

parser = Parser() 
parser.load(parser_model_path)

def func():
    # 句法结构分析
    pass

def dp(input_text:str, words = None, pos = None, show=False):
    if not words:
        words = segment(input_text)
    if not pos:
        pos = pos_tagging(input_text)

    arcs = parser.parse(words, pos)  
    if show:
        print('parser list', '\t'.join('%d: %s' %(arc.head, arc.relation) for arc in arcs))

    return arcs