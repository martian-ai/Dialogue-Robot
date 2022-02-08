from typing import List
import ckpe    

ckpe_obj = ckpe.ckpe()

# def extract_phrase_hanlp(context:str, topK:int=5) -> List:
#     return HanLP.extractPhrase(context, topK)

def extract_phrase_ckpe(context:str, topK:int=5) -> List:
    key_phrases = ckpe_obj.extract_keyphrase(context)
    return key_phrases