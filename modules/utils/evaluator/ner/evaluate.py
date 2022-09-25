import argparse

from .metric  import calc_partial_match_evaluation_per_line, calc_overall_evaluation


def process_boundary(tag: list, sent: list):
    """
    将 按字 输入的 list 转化为 entity list
    :param tag: [B,I,O,O,B,I,O]
    :param sent:[外，观，不，错，手，感，好]
    :return:
    """

    entity_val = ""
    tup_list = []
    entity_tag = None
    for i, tag in enumerate(tag):
        tok = sent[i]
        tag = "O" if tag==0 else tag
        # filter out "O"
        try:
            if tag.startswith('S-'):
                entity_tag = tag[2:]
                entity_val = tok
                tup_list.append((entity_tag, entity_val))

            if tag.startswith('B-'):
                if len(entity_val) > 0:
                    tup_list.append((entity_tag, entity_val))
                entity_tag = tag[2:]
                entity_val = tok
            elif tag.startswith("I-") and entity_tag == tag[2:]:
                entity_val += tok
            elif tag.startswith("E-") and entity_tag == tag[2:]:
                entity_val += tok
            elif tag.startswith("I-") and entity_tag != tag[2:]:
                if len(entity_val) > 0:
                    tup_list.append((entity_tag, entity_val))
                entity_tag = tag[2:]
                entity_val = tok
            elif tag.startswith("E-") and entity_tag != tag[2:]:
                if len(entity_val) > 0:
                    tup_list.append((entity_tag, entity_val))
                entity_tag = tag[2:]
                entity_val = tok
            elif tag in [0, 'O']:
                if len(str(entity_val)) > 0:
                    tup_list.append((entity_tag, entity_val))
                entity_val = ""
                entity_tag = None

        except Exception as e:
            pass
    if len(str(entity_val)) > 0:
        tup_list.append((entity_tag, entity_val))

    return tup_list


def cut_result_2_sentence_for_file(text_list, ground_list, predict_list):
    text_sentence_list = []
    ground_sentence_list = []
    predict_sentence_list = []
    
    tmp_t = []
    tmp_g = []
    tmp_p = []

    idx = 0

    for item_t, item_g, item_p in zip(text_list, ground_list, predict_list):

        #if len(item_g.strip()) == 0 and len(item_p.strip()) != 0:
        #    print('index', idx)
        #    raise Exception("Error")
        #elif len(item_g.strip()) != 0 and len(item_p.strip()) == 0:
        #   print('index', idx)
        #   raise Exception("Error")
        if len(item_g) == 0 and len(item_p) == 0:
            text_sentence_list.append(tmp_t.copy())
            ground_sentence_list.append(tmp_g.copy())
            predict_sentence_list.append(tmp_p.copy())
            tmp_t = []
            tmp_g = []
            tmp_p = []
        else:
            tmp_t.append(item_t)
            tmp_g.append(item_g)
            tmp_p.append(item_p)
        idx += 1 

    return text_sentence_list, ground_sentence_list, predict_sentence_list


def sentence_evaluate(char_list, tag_ground_list, tag_predict_list):
    
    print(tag_predict_list)
    print(tag_ground_list)
    print(char_list)

    entity_predict_list, entity_ground_list = process_boundary(tag_predict_list, char_list), process_boundary(tag_ground_list, char_list)
    print("%" * 20)
    print("entity ground list", entity_ground_list)
    print("entity predict list", entity_predict_list)
    print("%" * 20)

    #if entity_predict_list != entity_ground_list:
    #    print("###")
    #    print(char_list)
    #    print(tag_predict_list)
    #    print(tag_ground_list)
    
    #    print('predict###', entity_predict_list)
    #    print('ground###', entity_ground_list)
    
    char_list = [ item for item in char_list if type(item) is str]

    text = ''.join(char_list)

    calc_partial_match_evaluation_per_line(entity_predict_list, entity_ground_list, text, "NER")


def get_results_by_line(text_lines, ground_lines, predict_lines):


    #print("text lines top 3", text_lines)
    #print("ground lines top 3", ground_lines)
    #print("predict lines top 3", predict_lines)
    

    assert len(text_lines) == len(ground_lines) == len(predict_lines)


    # assert len(text_lines[0]) == len(ground_lines[0]) == len(predict_lines[0])

    count_predict = 0
    count_ground = 0
    for item in predict_lines:
        if len(item) == 0:
            count_predict += 1

    for item in ground_lines:
        if len(item) == 0:
            count_ground += 1
    assert count_predict == count_predict

    for item_t, item_g, item_p in zip(text_lines, ground_lines, predict_lines):
        print("#" * 100)
        print(" item text", item_t)
        print(" item ground", item_g)
        print(" item predict", item_p)
        sentence_evaluate(item_t, item_g, item_p)

    cnt_dict = {'NER': len(text_lines)}
    overall_res = calc_overall_evaluation(cnt_dict)
    p = overall_res['NER']['strict']['precision']
    r = overall_res['NER']['strict']['recall']
    f1 = overall_res['NER']['strict']['f1_score']

    return p, r, f1


def get_results_by_file(ground_text_path, predict_label_path):
    text_lines = []
    ground_lines = []

    with open(ground_text_path, mode='r', encoding='utf-8') as f:
        for item in f.readlines():
            cut_list = item.strip().split("\t")
            if len(cut_list) is 3:
                text_lines.append(cut_list[0])
                ground_lines.append(cut_list[1])
            else:
                text_lines.append("")
                ground_lines.append("")

    with open(predict_label_path, mode='r', encoding='utf-8') as f:
        predict_lines = f.readlines()

    count_predict = 0
    count_ground = 0
    for item in predict_lines:
        if len(item) == 0:
            count_predict += 1

    for item in ground_lines:
        if len(item) == 0:
            count_ground += 1
    assert count_predict == count_predict

    text_list, ground_list, predict_list = cut_result_2_sentence_for_file(text_lines, ground_lines, predict_lines)

    for item_t, item_g, item_p in zip(text_list, ground_list, predict_list):
        sentence_evaluate(item_t, item_g, item_p)

    cnt_dict = {'NER': len(text_list)}
    overall_res = calc_overall_evaluation(cnt_dict)
    p = overall_res['NER']['strict']['precision']
    r = overall_res['NER']['strict']['recall']
    f1 = overall_res['NER']['strict']['f1_score']

    return p, r, f1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_text_path', type=str)
    parser.add_argument('--predict_label_path', type=str)

    args = parser.parse_args()

    p, r, f1 = get_results_by_file(args.ground_text_path, args.predict_label_path)
    print('precision : {}, recall : {}, f1 score : {}'.format(p, r, f1))
