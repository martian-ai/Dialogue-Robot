import pickle

with open('resources/model/extractor/keywords/key_words_tf_idf.pkl', mode='rb') as f:
    key_words = pickle.load(f)

with open("resources/vocab/stopwords_hit.txt", mode='r', encoding='utf-8') as f:
    lines = f.readlines() 
    stop_words = [item.strip() for item in lines]
    
key_words_dict = {}
for item in key_words:
    if len(item[0]) < 2:
        pass
    elif item[0] in {"\n", "lt", "gt", ";", "&", "blockquote", '"', ' ', "class", "bp3"}:
        pass
    elif item[0] in ["进行", "或者", "可以", "根据", "包括", "管理", "标准", "批准"] + stop_words:
        pass
    else:
        key_words_dict[item[0]] = item[1]