import jieba

from modules.structure_sequence.keyword import key_words_dict
from modules.utils.tokenizer.segment import text_segment


def line_keywords(inputText, topK=5):
    cut_list = list(jieba.cut(inputText))
    results = []
    for item in cut_list:
        if item.isdigit():
            continue
        if item in key_words_dict.keys():
            results.append((item, key_words_dict[item]))
    results = list(set(results))
    sorted_results = sorted(results, key=lambda x:x[1], reverse=True)
    return sorted_results[:topK]

def doc_keywords(inputDoc, topK=5):
    """
    前后64 个字取关键词
    """
    inputText = '' 
    if len(inputDoc) > 128:
        text_seg = text_segment(inputDoc) 
        if len(text_seg) == 1:
            inputText = text_seg[0][:128]
        elif len(text_seg) == 0:
            inputText = ''
        else:
            inputText = text_seg[0][:64] + ' ' + text_seg[-1][-64:]
    
    word_dict = line_keywords(inputText, topK=topK)
    return word_dict

if __name__ == "__main__":

    import sys
    sys.path.append("..")

    def check(item_1, list_1):
        for item_2 in list_1:
            if len(set(list(item_2)).intersection(set(list(item_1))))>0 :
                return True
        else:
            return False

    def get_p_r_f1(ground_list, output_list, type):
        correct = 0
        if type == 'strict':
            correct = len(set(ground_list).intersection(set(output_list)))
        elif type == 'span':
            for item_1 in output_list:
                if check(item_1, ground_list):
                    correct += 1

        if len(output_list) > 0:
            precision = correct /len(output_list)
            recall = correct / len(ground_list)
            return precision, recall
        else:
            return 0, 0

    with open("data.txt", mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        precision_list, recall_list, f1_list = [], [], []
        for item in lines:
            cut_item = item.split('\t')
            ground_list = cut_item[1].split('###')
            inputText = cut_item[3]
            print("*"*100)
            print("text :", inputText)
            print("ground :", ground_list)
            # print(inputText)
            output_list = line_keywords(inputText, topK=6)
            output_list = [item[0] for item in output_list]
            print("output :", output_list)
            tmp_p, tmp_r = get_p_r_f1(ground_list, output_list, 'strict')
            print("precision", tmp_p)
            print("recall", tmp_r)
            precision_list.append(tmp_p)
            recall_list.append(tmp_r)
            if tmp_p + tmp_r > 0:
                f1_list.append(2*tmp_p*tmp_r/(tmp_p + tmp_r))
            else:
                f1_list.append(0)

        print("*"*100)
        print('all results')
        print("average precesion :", sum(precision_list)/len(precision_list))
        print("average recall :", sum(recall_list)/len(recall_list))
        print("average f1 :", sum(f1_list)/len(f1_list))


