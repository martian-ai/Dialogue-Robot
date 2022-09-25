
import torch
# import HanziConv
import pandas as pd
from torch import Tensor, LongTensor
from tqdm import tqdm
from torch.utils.data import Dataset

"""
两种类型：
分类方式
排序方式
"""

SEP, PAD, CLS = '[SEP]', '[PAD]', '[CLS]'

class MatchDataset(Dataset):
    def __init__(self, fname, tokenizer, label_dict: dict, max_seq_len: int) -> None:
        """
        读入文件进行数据构建

        Args:
            fname (_type_): _description_
            tokenizer (_type_): _description_
            label_dict (dict): _description_
            max_seq_len (int): _description_
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seqs, self.seq_masks, self.seq_segments, self.labels = self.get_input(fname)

        all_data = []
        for seq, mask, seg, label in zip(self.seqs, self.seq_masks, self.seq_segments, self.labels):
            data = {
                'indices_merge': seq,
                'segment' : seg,
                'mask': mask,
                'label': label_dict[str(int(label))]
            }
            all_data.append(data)
        
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def trunate_and_pad(self, tokens_seq_1, tokens_seq_2):
        """
        1. 如果是单句序列，按照BERT中的序列处理方式，需要在输入序列头尾分别拼接特殊字符'CLS'与'SEP'，
            因此不包含两个特殊字符的序列长度应该小于等于max_seq_len-2，如果序列长度大于该值需要那么进行截断。
        2. 对输入的序列 最终形成['CLS',seq,'SEP']的序列，该序列的长度如果小于max_seq_len，那么使用0进行填充。
        入参: 
            seq_1       : 输入序列，在本处其为单个句子。
            seq_2       : 输入序列，在本处其为单个句子。
            max_seq_len : 拼接'CLS'与'SEP'这两个特殊字符后的序列长度
        
        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                            那么取值为1，否则为0。
            seq_segment : shape等于seq，单句，取值都为0 ，双句按照01切分
            
        """
        # 对超长序列进行截断
        if len(tokens_seq_1) > ((self.max_seq_len - 3)//2):
            tokens_seq_1 = tokens_seq_1[0:(self.max_seq_len - 3)//2]
        if len(tokens_seq_2) > ((self.max_seq_len - 3)//2):
            tokens_seq_2 = tokens_seq_2[0:(self.max_seq_len - 3)//2]
        # 分别在首尾拼接特殊符号
        seq = ['[CLS]'] + tokens_seq_1 + ['[SEP]'] + tokens_seq_2 + ['[SEP]']
        seq_segment = [0] * (len(tokens_seq_1) + 2) + [1] * (len(tokens_seq_2) + 1)
        # ID化
        seq = self.tokenizer.tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = seq_segment + padding
        # 对seq拼接填充序列
        seq += padding
        assert len(seq) == self.max_seq_len
        assert len(seq_mask) == self.max_seq_len
        assert len(seq_segment) == self.max_seq_len
        return seq, seq_mask, seq_segment

    # 获取文本与标签
    def get_input(self, file):
        """
        通对输入文本进行分词、ID化、截断、填充等流程得到最终的可用于模型输入的序列。
        入参:
            dataset     : pandas的dataframe格式，包含三列，第一,二列为文本，第三列为标签。标签取值为{0,1}，其中0表示负样本，1代表正样本。
            max_seq_len : 目标序列长度，该值需要预先对文本长度进行分别得到，可以设置为小于等于512（BERT的最长文本序列长度为512）的整数。
        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。
            labels      : 标签取值为{0,1}，其中0表示负样本，1代表正样本。
        """
        df = pd.read_csv(file, sep='\t')
        # sentences_1 = map(HanziConv.toSimplified, df['sentence1'].values)
        # sentences_2 = map(HanziConv.toSimplified, df['sentence2'].values)
        sentences_1 = df['q1'].values
        sentences_2 = df['q2'].values
        labels = df['label'].values
        # 切词
        tokens_seq_1 = list(map(self.tokenizer.tokenize, sentences_1))
        tokens_seq_2 = list(map(self.tokenizer.tokenize, sentences_2))
        # 获取定长序列及其mask
        result = list(map(self.trunate_and_pad, tokens_seq_1, tokens_seq_2))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.Tensor(seqs).type(torch.long), torch.Tensor(seq_masks).type(torch.long), torch.Tensor(seq_segments).type(torch.long), torch.Tensor(labels).type(torch.long)


class MatchDatasetSingleLine(Dataset):
    def __init__(self, text1, text2, tokenizer, pad_size=64, label_dict={}):
        """
        """
        self.max_seq_len = pad_size
        self.tokenizer = tokenizer
        self.seqs, self.seq_masks, self.seq_segments, self.labels = self.get_input([text1], [text2]) # TODO 一个batch 预测

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    # 获取文本与标签
    def get_input(self, sentences_1, sentences_2):
        """
        通对输入文本进行分词、ID化、截断、填充等流程得到最终的可用于模型输入的序列。
        入参:
            dataset     : pandas的dataframe格式，包含三列，第一,二列为文本，第三列为标签。标签取值为{0,1}，其中0表示负样本，1代表正样本。
            max_seq_len : 目标序列长度，该值需要预先对文本长度进行分别得到，可以设置为小于等于512（BERT的最长文本序列长度为512）的整数。
        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。
            labels      : 标签取值为{0,1}，其中0表示负样本，1代表正样本。
        """

        assert len(sentences_1) == len(sentences_2)
        labels = [0] * len(sentences_1)
        # 切词
        tokens_seq_1 = list(map(self.bert_tokenizer.tokenize, sentences_1))
        tokens_seq_2 = list(map(self.bert_tokenizer.tokenize, sentences_2))
        # 获取定长序列及其mask
        result = list(map(self.trunate_and_pad, tokens_seq_1, tokens_seq_2))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.Tensor(seqs).type(torch.long), torch.Tensor(seq_masks).type(torch.long), torch.Tensor(seq_segments).type(torch.long), torch.Tensor(labels).type(torch.long)