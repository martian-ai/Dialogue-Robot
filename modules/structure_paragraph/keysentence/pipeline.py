from joblib import parallel_backend
from modules.structure_paragraph.keysentence.function.textrank import \
    keysentences_extraction


def paragraph_keysentences(inputPara, topK=2, minLen=1):
    res = keysentences_extraction(inputPara, topK, minLen)

    para_list = []
    for item in res:
        para_list.append(item["sentence"])

    return para_list


if __name__ == "__main__":
    pass
    # TODO
