'''
Description: 
Version: 
Author: Apollo
Date: 2021-12-15 18:33:29
LastEditors: sueRim
LastEditTime: 2021-12-15 18:33:30
Todo: 
'''
import os
import json
import numpy as np
import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM

DEVICE = 'cpu:0'
MODEL_PATH = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForMaskedLM.from_pretrained(MODEL_PATH)
model.eval()
model.to(DEVICE)
text_formatter = lambda x: "[CLS]{} [SEP]".format(x)

def sentence_preprocese(text):
    tokenized_text = np.array(tokenizer.tokenize(text))
    find_sep = np.argwhere(tokenized_text == '[SEP]')
    segments_ids = np.zeros(tokenized_text.shape, dtype=int)
    if find_sep.size == 1:
        start_point = 1
    else:
        start_point = find_sep[0, 0] + 1
        segments_ids[start_point:] = 1
    end_point = tokenized_text.size - 1
    tokenized_text = tokenizer.convert_tokens_to_ids(tokenized_text)
    masked_texts = []
    for masked_index in range(start_point, end_point):
        new_tokenized_text = np.array(tokenized_text, dtype=int)
        new_tokenized_text[masked_index] = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        masked_texts.append(new_tokenized_text)
    segments_ids = np.tile(segments_ids, (end_point - start_point, 1))
    return masked_texts, segments_ids, start_point, end_point, tokenized_text[start_point:end_point]

def line_perplexity(text):
    text = text_formatter(text)
    indexed_tokens, segments_ids, start_point, end_point, real_indexs = sentence_preprocese(text)
    tokens_tensor = torch.tensor(indexed_tokens)
    segments_tensors = torch.tensor(segments_ids)
    tokens_tensor = tokens_tensor.to(DEVICE)
    segments_tensors = segments_tensors.to(DEVICE)
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = torch.softmax(outputs[0], -1)
    total_perplexity = 0
    for i, step in enumerate(range(start_point, end_point)):
        total_perplexity += np.log(predictions[i, step, real_indexs[i]].item())
    total_perplexity = -total_perplexity / (end_point - start_point)
    return total_perplexity

if __name__ == '__main__':
    print(line_perplexity('华为手机的售后服务好吗'))