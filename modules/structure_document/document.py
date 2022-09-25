import re
from zhon import hanzi


def doc_segment_by_punc(document):

    rst = re.findall(hanzi.sentence, document)
    if rst == []:
        return [document]
    return rst


def get_data(filename):
    lines = []
    with open(filename, encoding='utf-8', mode='r') as f:
        lines = f.readlines()
    return lines


def paragraph_chunking(lines, max_char_length):
    """
    当前仅实现简单版本
    TODO 复杂版本调研，chunking 模块和后续模块联合调用
    TODO 返回结果使用generator
    """
    from functools import reduce

    def count_all_paragraph(paras):
        count = 0
        for item in paras:
            count += len(item)
        return count

    all_results = []
    tmp_list = []
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        if len(line) > max_char_length:
            # print("a")
            all_results.append(line)
        elif count_all_paragraph(tmp_list) <= max_char_length and count_all_paragraph(tmp_list + [line]) > max_char_length:
            # print("b")
            # print(count_all_paragraph(tmp_list))
            # print(count_all_paragraph(tmp_list + [line]))
            # print(len(tmp_list))
            all_results.append(tmp_list.copy())
            tmp_list = []
            tmp_list.append(line)
        else:
            # print("c")
            # print(count_all_paragraph(tmp_list))
            tmp_list.append(line)

    return all_results


if __name__ == "__main__":
    MAX_CHAR_COUNT = 384
    lines = get_data('resources/corpus/document/三体.txt')
    print(len(lines))
    paras = paragraph_chunking(lines, MAX_CHAR_COUNT)
    print(len(paras))
    print(paras[0])
    print(paras[-1])
