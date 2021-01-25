import re

"""
非中文去除 
"""
def nochinese_delete(text):
    return re.sub(r'[^\u4e00-\u9fa5]+','', text)



# from zhon.hanzi import punctuation
# print(type(punctuation))
