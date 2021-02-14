from snownlp import SnowNLP
from pyltp import SentenceSplitter

def discourse_analysis(input_text:str):
    """
    # 长句压缩/消除冗余/文本摘要
    #   TextRank 
    #       1. 找语料 2. 找其他方法
    #   snownlp
    # 指代消解
    #   探索方法(Transformer)
    """
    # 分句
    sentences = list(SentenceSplitter.split(input_text))

    # 摘要
    s = SnowNLP(input_text)
    summary_list = s.summary(limit=3)

    # 指代消解 : 使用段落上文信息进行指代消解 零指代
    # 
    pass

    return sentences, summary_list