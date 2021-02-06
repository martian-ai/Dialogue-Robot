import os
import sys
sys.path.append('../..')
RESOURCE_DIR = 'resources/'
LTP_DATA_DIR = os.path.join(RESOURCE_DIR, 'ltp_data_v3.4.0')

cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词，
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标，
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别
parser_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 句法分析
label_model_path = os.path.join(LTP_DATA_DIR, "pisrl.model") # 语义角色标注

stopwords_path = os.path.join(RESOURCE_DIR, 'HIT_stop_words.txt')
abbreviation_path = os.path.join(RESOURCE_DIR, 'abbreviations.txt')