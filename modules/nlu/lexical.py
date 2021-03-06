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

def segment(input_text:str):
    words = segmentor.segment(input_text) 
    return list(words) 

def pos_tagging(input_text, words=None):
    if not words:
        words = segment(input_text)
        pos_tags = postagger.postag(words)  
        return list(pos_tags) 
    else:
        pos_tags = postagger.postag(words)  
        return list(pos_tags) 

def name_entity(input_text, words=None, pos=None):
    if not words and pos:
        words = segment(input_text)
        return list(recognizer.recognize(words, pos))
    elif not pos and words:
        pos = pos_tagging(input_text, words)
        return list(recognizer.recognize(words, pos))
    elif not words and not pos:
        tmp_words = segment(input_text)
        tmp_pos = pos_tagging(input_text, tmp_words)
        return list(recognizer.recognize(tmp_words, tmp_pos))
    else:
        return list(recognizer.recognize(words, pos))

def lexical_analysis(input_text):
    seg = segment(input_text)
    pos = pos_tagging(input_text, seg)
    ner = name_entity(input_text, pos)
    return seg, pos, ner

if __name__ == "__main__":
    # print(seg('今天上班给老人让坐，四十分鐘的車程')) 
    # print(pos('今天上班给老人让坐，四十分鐘的車程')) 
    print(ner('今天上班给老人让坐，四十分鐘的車程')) 