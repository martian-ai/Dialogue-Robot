# -*- coding=utf-8 -*-
# tokenizer for all system
# input : text
# output : (cut_list, span_list)
import jieba

class JieBaTokenizer(object):
    def __init__(self):
        self.tokenizer = jieba

    def tokenize(self, doc):
        tokens = list(self.tokenizer.cut(doc))
        start, token_spans = 0, []
        for token in tokens:
            token_spans.append((start, start + len(token)))
            start += len(token)
        return tokens, token_spans

if __name__ == '__main__':
    jtok = JieBaTokenizer()
    print(jtok.tokenize('中午在西坝河吃了个午饭'))
    # (['中午', '在', '西坝河', '吃', '了', '个', '午饭'], [(0, 2), (2, 3), (3, 6), (6, 7), (7, 8), (8, 9), (9, 11)])



