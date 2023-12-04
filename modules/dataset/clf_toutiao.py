import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from datasets import load_dataset
import torch


class Dataset_Toutiao(torch.utils.data.Dataset):
    def __init__(self, df, labels, tokenizer, max_len=128):
        self.labels = [labels[label] for label in df['label']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length = max_len,
                                truncation=True,
                                return_tensors="pt")
                      for text in df['text']]
 
    def classes(self):
        return self.labels
 
    def __len__(self):
        return len(self.labels)
 
    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])
 
    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]
 
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

def fetch_by_huggingface(name='fourteenBDr/toutiao'):
    """_summary_
    Args:
        name (str, optional): _description_. Defaults to 'fourteenBDr/toutiao'.

    Returns:
        _type_: _description_
    """
    dataset = load_dataset(name)
    return dataset

def convert_csv():
    """_summary_

    fourteenBDr/toutiao 
        source:
            [in use] https://huggingface.co/datasets/fourteenBDr/toutiao
            [other backup] https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset
        98% case length < 128
        all case numbert 380K
    
    Returns:
        _type_: _description_
    """
    data = fetch_by_huggingface()
    origin_list = data['train']['text']
    df = pd.DataFrame(columns=['label', 'text'])
    label_list, text_list = [], []
    for item in tqdm(origin_list):
        items = item.split('_!_')
        code = items[2]
        title = items[3]
        keyword = items[4]
        label = code
        text = title + keyword
        label_list.append(label)
        text_list.append(text)
    df['label'] = label_list
    df['text'] = text_list
    df.to_csv('toutiao_cat_data.csv', index=False, header=True)

if __name__ == '__main__':
    convert_csv()