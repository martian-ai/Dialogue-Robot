# -*- coding:utf-8 -*-

import jieba
from xpinyin import Pinyin
import re

pin = Pinyin()

def zh_extract(text):
    """
    找到文本中的所有中文
    """
    pattern = "[\u4e00-\u9fa5]+"
    regex = re.compile(pattern)
    results_list = regex.findall(text)
    return results_list

def get_char_pinyin(input_text):
    """
    找到所有中文，如果不是中文，不转化未拼音，即跳过
    :param input_text: 原始文本，可能包括中文，英文或其他语言的字符
    :return: 中文部分 字符 拼音 list
    """
    return_char_list = []
    zh_part_list = zh_extract(input_text)
    for item_c in zh_part_list:
        item_char_list = pin.get_pinyin(item_c).split('-')
        return_char_list.extend(item_char_list)
    return return_char_list

def get_word_pinyin(input_text):
    """
    找到所有中文，如果不是中文，不转化未拼音，即跳过
    :param input_text string 原始文本，可能包括中文，英文或其他语言的字符
    :return: 中文部分 分词后 拼音 list
    """
    return_word_list = []
    zh_part_list = zh_extract(input_text)
    for item_zh in zh_part_list:
        item_zh_cut_list = list(jieba.cut(item_zh, cut_all=True))  # 全分词
        for item_tmp in item_zh_cut_list:  # 分词后的各个部分， 转化为词的拼音
            item_tmp_str = ''.join(pin.get_pinyin(item_tmp).split('-'))
            return_word_list.append(item_tmp_str)
    return return_word_list

if __name__ == '__main__':
    input_text = "中华人民共和国, <+, China, スタッフ, 직원 国家权力"
    print(zh_extract(input_text))
    print(get_char_pinyin(input_text))
    print(get_word_pinyin(input_text))