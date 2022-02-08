import re

from snownlp import SnowNLP

"""
繁简转化 
"""
def traditional_to_simple(text):
    s = SnowNLP(text)
    return s.han


def text_segment(text):
    return re.split('[?!。？！]', text)


"""
非中文去除 
"""
def string_upper(text):
    return text.upper()

def string_lower(text):
    return text.lower()

def string_title(text):
    return text.title()


def nochinese_delete(text):
    return re.sub(r'[^\u4e00-\u9fa5]+','', text)

def DBC2SBC(ustring):
    '''
    全角转半角
    '''
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if not (0x0021 <= inside_code and inside_code <= 0x7e):
            rstring += uchar
            continue
        rstring += chr(inside_code)
    return rstring
 
def SBC2DBC(ustring):
    '''
    半角转全角
    '''
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x0020:
            inside_code = 0x3000
        else:
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
            continue
        inside_code += 0xfee0
        rstring += chr(inside_code)
    return rstring

