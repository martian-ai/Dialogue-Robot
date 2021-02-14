from snownlp import SnowNLP

"""
繁简转化 
"""
def traditional_to_simple(text):
    s = SnowNLP(text)
    return s.han