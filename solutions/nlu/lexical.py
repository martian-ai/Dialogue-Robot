import os
import sys

from pyltp import Segmentor, Postagger, NamedEntityRecognizer, Parser
from snownlp import SnowNLP
from solutions.nlu import cws_model_path, pos_model_path, ner_model_path

segmentor = Segmentor()  
segmentor.load(cws_model_path)  
postagger = Postagger()  
postagger.load(pos_model_path)  
recognizer = NamedEntityRecognizer()  
recognizer.load(ner_model_path)  

def seg(input_text:str):
    words = segmentor.segment(input_text) 
    return list(words) 

def pos(input_text, words=None):
    if not words:
        words = seg(input_text)
        pos_tags = postagger.postag(words)  
        return list(pos_tags) 
    else:
        pos_tags = postagger.postag(words)  
        return list(pos_tags) 

def ner(input_text, _words=None, _pos=None):
    if not _words and _pos:
        _words = seg(input_text)
        return list(recognizer.recognize(_words, _pos))
    elif not _pos and _words:
        _pos = _pos(input_text, _words)
        return list(recognizer.recognize(_words, _pos))
    elif not _words and not _pos:
        tmp_words = seg(input_text)
        tmp_pos = pos(input_text, tmp_words)
        return list(recognizer.recognize(tmp_words, tmp_pos))
    else:
        return list(recognizer.recognize(_words, _pos))

if __name__ == "__main__":
    # print(seg('今天上班给老人让坐，四十分鐘的車程')) 
    # print(pos('今天上班给老人让坐，四十分鐘的車程')) 
    print(ner('今天上班给老人让坐，四十分鐘的車程')) 