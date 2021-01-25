import os
# LTP_DATA_DIR = '/Users/sunhongchao/Documents/craft/09_Dialogue/resources/ltp_data_v3.4.0'
# RESOURCE_DIR = '/Users/sunhongchao/Documents/craft/09_Dialogue/resources/'
# vocab_tencent = '/Users/sunhongchao/Documents/craft/09_Dialogue/resources/vocab_tencent.pkl'
LTP_DATA_DIR = '/export/resources/ltp_data_v3.4.0'
RESOURCE_DIR = '/export/resources/'
vocab_tencent = '/export/mrc_flask_deploy/app/resources/vocab_tencent.pkl'

cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词，
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标，
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别
parser_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 句法分析
label_model_path = os.path.join(LTP_DATA_DIR, "pisrl.model")
stopwords_path = os.path.join(RESOURCE_DIR, 'HIT_stop_words.txt')
abbreviation_path = os.path.join(RESOURCE_DIR, 'abbreviations.txt')