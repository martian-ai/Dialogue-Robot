import os
import numpy as np
from bert_serving.client import BertClient
from annoy import AnnoyIndex

bc_embedding_path = '../../resources/clue-pair'
if os.path.exists(bc_embedding_path) is False:
     raise RuntimeError('embedding file is not exist')
os.system('bert-serving-start -model_dir ' + bc_embedding_path + ' -num_worker=4 ')
bc_embedding_size = 312
bc_document_size = 5
bc = BertClient()
" annoy parameters"
annoy_tree_number = 100


def text_encode(input_text):
    return bc.encode([input_text])[0]

def sentences_encode(input_list):
    return bc.encode(input_list)

def paragraphs_query_encode(input_list, input_term):
    """
    todo : how to represent paragraph
    current solution : mean of all sentence in current parapraph
    """
    return np.mean([ sentences_encode(item) for item in input_list], axis=1)

def annoy(encode_list, encode_item, text_list):
    ann = AnnoyIndex(bc_embedding_size, 'angular') # 312 为向量维度
    for idx, item in enumerate(encode_list):
        embd = item.tolist() 
        ann.add_item(idx, embd)
    ann.build(annoy_tree_number )
    idx_list = ann.get_nns_by_vector(encode_item, bc_document_size)
    return [ text_list[tmp_idx ]for tmp_idx in idx_list ]

def eculd(encode_list, encode_item, text_list):

    def distEclud(vecA, vecB):
        return np.sqrt(sum(np.power(vecA - vecB, 2)))
    score_list = []
    for value in encode_list:
        score_list.append(distEclud(encode_item, value))

    zipped = zip(score_list, text_list)
    sorted_zipped = np.asarray(sorted(zipped, reverse=False, key=lambda x: (x[0], x[1])))
    return sorted_zipped[:bc_document_size][:, 1]