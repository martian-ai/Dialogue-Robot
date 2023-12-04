'''

Copyright (c) 2023 by Martain.AI, All Rights Reserved.

Description:  basic generateor data function
    1. conversation generate
    2. question generate
    3. answer generate
    4. poetry generate
    5. summary generate
    6. comment generate
    7. multimodal generate(todo)

    同时实现使用 paddle 和 torch 加载数据
    同时实现加载 本地数据 和 hugguingface 数据

Author: apollo2mars apollo2mars@gmail.com
Date: 2023-06-21 22:40:14
LastEditors: apollo2mars apollo2mars@gmail.com
LastEditTime: 2023-10-29 22:23:51
FilePath: \Dialogue-Robot\modules\basic_generator\data.py
'''

from torch.utils.data import Dataset
from hanziconv import HanziConv
import pandas as pd
import torch

import random
from functools import partial

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler, BatchSampler
from paddlenlp.data import Pad

class DataArugument(object):
    self.tokenizer = 
    self.max_seq_len = 
    self.max_target_len = 

    

def print_args(args):
    '''
    description: 
    return {*}
    '''
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def set_seed(seed):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(seed)
    np.random.seed(seed)
    # Maybe different op seeds(for dropout) for different procs is better.
    paddle.seed(seed + dist.get_rank())


def convert_example(example, tokenizer, max_seq_len=512, max_target_len=128, max_title_len=256, mode="train"):
    """Convert all examples into necessary features."""
    print(example)
    source = example["source"]
    title = None
    if "title" in example.keys():
        title = example["title"]

    if mode != "test":
        tokenized_example = tokenizer.gen_encode(
            source,
            title=title,
            target=example["target"],
            max_seq_len=max_seq_len,
            max_target_len=max_target_len,
            max_title_len=max_title_len,
            return_position_ids=True,
            return_length=True,
        )
        target_start = tokenized_example["input_ids"].index(tokenizer.cls_token_id, 1)
        target_end = tokenized_example["seq_len"]
        # Use to gather the logits corresponding to the labels during training
        tokenized_example["masked_positions"] = list(range(target_start, target_end - 1))
        tokenized_example["labels"] = tokenized_example["input_ids"][target_start + 1:target_end]

        return tokenized_example
    else:
        tokenized_example = tokenizer.gen_encode(
            source,
            title=title,
            max_seq_len=max_seq_len,
            max_title_len=max_title_len,
            add_start_token_for_decoding=True,
            return_position_ids=True,
        )

        if "target" in example and example["target"]:
            tokenized_example["target"] = example["target"]
        return tokenized_example


def convert_example_dureaderqg(example, tokenizer, max_seq_len=512, max_target_len=128, max_title_len=256, mode="train"):
    """Convert all examples into necessary features."""
    # print(example)
    source = example["context"]
    title = None
    if "answer" in example.keys():
        title = example["answer"]

    # if "title" in example.keys():
    #     title = example["title"]

    if mode != "test":
        tokenized_example = tokenizer.gen_encode(
            source,
            title=title,
            target=example["question"],  # target
            max_seq_len=max_seq_len,
            max_target_len=max_target_len,
            max_title_len=max_title_len,
            return_position_ids=True,
            return_length=True,
        )
        target_start = tokenized_example["input_ids"].index(tokenizer.cls_token_id, 1)
        target_end = tokenized_example["seq_len"]
        # Use to gather the logits corresponding to the labels during training
        tokenized_example["masked_positions"] = list(range(target_start, target_end - 1))
        tokenized_example["labels"] = tokenized_example["input_ids"][target_start + 1:target_end]

        return tokenized_example
    else:
        tokenized_example = tokenizer.gen_encode(
            source,
            title=title,
            max_seq_len=max_seq_len,
            max_title_len=max_title_len,
            add_start_token_for_decoding=True,
            return_position_ids=True,
        )

        # if "target" in example and example["target"]:
        #     tokenized_example["target"] = example["target"]
        if "question" in example and example["question"]:
            tokenized_example["question"] = example["question"]

        return tokenized_example


def batchify_fn(batch_examples, pad_val, mode):
    def pad_mask(batch_attention_mask):
        batch_size = len(batch_attention_mask)
        max_len = max(map(len, batch_attention_mask))
        attention_mask = np.ones((batch_size, max_len, max_len), dtype="float32") * -1e9
        for i, mask_data in enumerate(attention_mask):
            seq_len = len(batch_attention_mask[i])
            mask_data[-seq_len:, -seq_len:] = np.array(batch_attention_mask[i], dtype="float32")
        # In order to ensure the correct broadcasting mechanism, expand one
        # dimension to the second dimension (n_head of Transformer).
        attention_mask = np.expand_dims(attention_mask, axis=1)
        return attention_mask

    pad_func = Pad(pad_val=pad_val, pad_right=False, dtype="int64")

    input_ids = pad_func([example["input_ids"] for example in batch_examples])
    token_type_ids = pad_func([example["token_type_ids"] for example in batch_examples])
    position_ids = pad_func([example["position_ids"] for example in batch_examples])

    attention_mask = pad_mask([example["attention_mask"] for example in batch_examples])

    if mode != "test":
        max_len = max([example["seq_len"] for example in batch_examples])
        masked_positions = np.concatenate([
            np.array(example["masked_positions"]) + (max_len - example["seq_len"]) + i * max_len
            for i, example in enumerate(batch_examples)
        ])
        labels = np.concatenate([np.array(example["labels"], dtype="int64") for example in batch_examples])
        return input_ids, token_type_ids, position_ids, attention_mask, masked_positions, labels
    else:
        return input_ids, token_type_ids, position_ids, attention_mask


def create_data_loader(dataset, tokenizer, args, mode):
    trans_func = partial(
        convert_example_dureaderqg,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        max_target_len=args.max_target_len,
        max_title_len=args.max_title_len,
        mode=mode,
    )
    dataset = dataset.map(trans_func, lazy=True)
    # for item in dataset:
    #     print(item)
    if mode == "train":
        batch_sampler = DistributedBatchSampler(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        batch_sampler = BatchSampler(dataset, batch_size=args.batch_size // 2, shuffle=False)
    collate_fn = partial(batchify_fn, pad_val=tokenizer.pad_token_id, mode=mode)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, return_list=True)
    # for item in data_loader:
    #     print(item)
    return dataset, data_loader


def post_process_sum(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.mask_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    special_tokens = ["[UNK]"]
    tokens = [token for token in tokens if token not in special_tokens]
    return token_ids, tokens


def select_sum(ids, scores, tokenizer, max_dec_len=None, num_return_sequences=1):
    results = []
    group = []
    tmp = []
    if scores is not None:
        ids = ids.numpy()
        scores = scores.numpy()

        if len(ids) != len(scores) or (len(ids) % num_return_sequences) != 0:
            raise ValueError("the length of `ids` is {}, but the `num_return_sequences` is {}".format(
                len(ids), num_return_sequences))

        for pred, score in zip(ids, scores):
            pred_token_ids, pred_tokens = post_process_sum(pred, tokenizer)
            num_token = len(pred_token_ids)

            target = "".join(pred_tokens)

            # not ending
            if max_dec_len is not None and num_token >= max_dec_len:
                score -= 1e3

            tmp.append([target, score])
            if len(tmp) == num_return_sequences:
                group.append(tmp)
                tmp = []

        for preds in group:
            preds = sorted(preds, key=lambda x: -x[1])
            results.append(preds[0][0])
    else:
        ids = ids.numpy()

        for pred in ids:
            pred_token_ids, pred_tokens = post_process_sum(pred, tokenizer)
            num_token = len(pred_token_ids)
            response = "".join(pred_tokens)

            # TODO: Support return scores in FT.
            tmp.append([response])
            if len(tmp) == num_return_sequences:
                group.append(tmp)
                tmp = []

        for preds in group:
            results.append(preds[0][0])

    return results


class DataPrecessForSentence(Dataset):
    """对文本进行处理, 符合生成模型的要求."""
    def __init__(self, bert_tokenizer, LCQMC_file, max_char_len=32):
        """初始化函数.

        Args:
            bert_tokenizer (_type_): _description_
            LCQMC_file (_type_): _description_
            max_char_len (int, optional): _description_. Defaults to 32.
        """
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_char_len
        self.seqs, self.seq_masks, self.seq_segments, self.labels = self.get_input(LCQMC_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx], self.labels[idx]

    # 获取文本与标签
    def get_input(self, file):
        """_summary_

        Args:
            file (_type_): _description_

        Returns:
            _type_: _description_
        """
        train_ds = load_dataset(args.dataset_name, splits="train", data_files=args.train_file)
        dev_ds = load_dataset(args.dataset_name, splits="dev", data_files=args.predict_file)
        data = DuReaderQG().read()

        df = pd.read_csv(file, sep='\t')
        sentences_1 = map(HanziConv.toSimplified, df['text'].values)
        # sentences_2 = map(HanziConv.toSimplified, df['q2'].values)
        labels = df['label'].values
        # 切词
        tokens_seq_1 = list(map(self.bert_tokenizer.tokenize, sentences_1))
        # tokens_seq_2 = list(map(self.bert_tokenizer.tokenize, sentences_2))
        # 获取定长序列及其mask
        # result = list(map(self.trunate_and_pad, tokens_seq_1, tokens_seq_2))
        result = list(map(self.trunate_and_pad, tokens_seq_1))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.Tensor(seqs).type(torch.long), torch.Tensor(seq_masks).type(torch.long), torch.Tensor(seq_segments).type(
            torch.long), torch.Tensor(labels).type(torch.long)

    # def trunate_and_pad(self, tokens_seq_1, tokens_seq_2):
    def trunate_and_pad(self, tokens_seq_1):
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
        # if len(tokens_seq_1) > ((self.max_seq_len - 3)//2):
        #     tokens_seq_1 = tokens_seq_1[0:(self.max_seq_len - 3)//2]
        if len(tokens_seq_1) > (self.max_seq_len - 3):
            tokens_seq_1 = tokens_seq_1[0:(self.max_seq_len - 3)]
        # if len(tokens_seq_2) > ((self.max_seq_len - 3)//2):
        #     tokens_seq_2 = tokens_seq_2[0:(self.max_seq_len - 3)//2]
        # 分别在首尾拼接特殊符号
        # seq = ['[CLS]'] + tokens_seq_1 + ['[SEP]'] + tokens_seq_2 + ['[SEP]']
        # seq_segment = [0] * (len(tokens_seq_1) + 2) + [1] * (len(tokens_seq_2) + 1)

        seq = ['[CLS]'] + tokens_seq_1 + ['[SEP]']
        seq_segment = [0] * (len(tokens_seq_1) + 2)

        # ID化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
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

        # """
        # 通对输入文本进行分词、ID化、截断、填充等流程得到最终的可用于模型输入的序列。
        # 入参:
        #     dataset     : pandas的dataframe格式，包含三列，第一,二列为文本，第三列为标签。标签取值为{0,1}，其中0表示负样本，1代表正样本。
        #     max_seq_len : 目标序列长度，该值需要预先对文本长度进行分别得到，可以设置为小于等于512（BERT的最长文本序列长度为512）的整数。
        # 出参:
        #     seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
        #     seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
        #                   那么取值为1，否则为0。
        #     seq_segment : shape等于seq，因为是单句，所以取值都为0。
        #     labels      : 标签取值为{0,1}，其中0表示负样本，1代表正样本。
        # """