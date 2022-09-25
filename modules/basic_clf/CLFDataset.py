from calendar import c
from torch.utils.data import Dataset
from tqdm import tqdm


class CLFDataset(Dataset):
    def __init__(self, fname, tokenizer, label_dict={}):
        """
        """
        with open(fname, "r", encoding='utf-8') as f:
            examples = f.readlines()
        all_data = []
        for entry in tqdm(examples):
            entry = entry.strip()
            try:
                label, content = entry.split('\t')
            except :
                continue
            content_indices = tokenizer.text_to_sequence(content, pad=True, add_special_tokens=True)
            data = {
                'indices':content_indices,
                'polarity' : label_dict[label]
            }
            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class CLFDatasetSingleLine(Dataset):
    def __init__(self, text, tokenizer, label_dict={}):
        """
        """
        examples = [text]
        all_data = []
        for entry in tqdm(examples):
            entry = entry.strip()
            try:
                label, content = entry.split('\t')
            except :
                label = 'NA'
                content = entry
            print(content)
            print(len(content))
            content_indices = tokenizer.text_to_sequence(content)
            data = {
                'indices':content_indices,
                'polarity' : label
            }
            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
