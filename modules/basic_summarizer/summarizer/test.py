import json
import utils
import argparse, random, numpy, os
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import jieba
import models
from utils.file_utils import get_dir_file_list
from utils.rouge_utils import rouge
from utils.Vocab import Vocab
from utils.Dataset import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

parser = argparse.ArgumentParser()
parser.add_argument('-output', type=str, default='output', required=False)
parser.add_argument('-input', type=str, default='input', required=False)
parser.add_argument('-project', type=str, default='.', required=False)
parser.add_argument('-gpu_id', type=str, default='0', required=False)
parser.add_argument('-model', type=str, default='CNN_RNN', required=False)
parser.add_argument('-batch_size', type=int, default=128, required=False)
parser.add_argument('-seed', type=int, default=1, required=False)
parser.add_argument('-device', type=int, required=False)
args, unknown = parser.parse_known_args()


def write_file_by_append(file_path, data, encoding='utf-8'):
    """
    如果文件存在，则通过从原文件结尾处添加新数据的方式写入
        + 打开一个文件用于追加。
        + 如果该文件已存在，文件指针将会放在文件的结尾。
        + 也就是说，新的内容将会被写入到已有内容之后。
        + 如果该文件不存在，创建新文件进行写入。
    :param file_path: 文件路径
    :param data: python基础数据类型的数据
    :param encoding: 字符集
    :return:
    """
    with open(file_path, 'a', encoding=encoding) as file:
        file.write(data)
        file.flush()  # 主动刷新文件内部缓冲，直接把内部缓冲区的数据立刻写入文件, 而不是被动的等待输出缓冲区写入。
        file.close()


def test():
    # use_gpu config
    use_gpu = args.device is not None
    if torch.cuda.is_available() and not use_gpu:
        print("WARNING: You have a CUDA device, should run with -device 0")
    if use_gpu:
        torch.cuda.set_device(args.device)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # build path
    embedding_file_path = os.path.join(args.project, "embedding.npz")
    vocab_file_path = os.path.join(args.project, "word2id.json")
    test_files_dir = os.path.join(args.input, "test_files")

    test_file_path_list = get_dir_file_list(test_files_dir)

    for item_test_file_path in test_file_path_list:
        print("文件[" + item_test_file_path + "]准备测试！")
        test_item_file(item_test_file_path, embedding_file_path, vocab_file_path, use_gpu)
        print("文件[" + item_test_file_path + "]测试结束！")

    print("智能摘要测试结束")


def test_item_file(end_test_file, embedding_file_path, vocab_file_path, use_gpu):
    embed = torch.Tensor(np.load(embedding_file_path)['arr_0'])
    with open(vocab_file_path) as f:
        word2id = json.load(f)
    vocab = Vocab(embed, word2id)
    #with open(end_test_file) as f:
    #    examples = [json.loads(line) for line in f]
    with open(end_test_file) as f:
        examples = list()
        for line in f:
            if line and not line.isspace():
                examples.append(json.loads(line))
    #print(examples[0])
    test_dataset = Dataset(examples)

    test_iter = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    load_dir = os.path.join(args.input,'model_files', 'CNN_RNN.pt')
    if use_gpu:
        checkpoint = torch.load(load_dir)
    else:
        checkpoint = torch.load(load_dir, map_location=lambda storage, loc: storage)
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(models, checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()
    doc_num = len(test_dataset)

    all_targets = []
    all_results = []
    all_probs = []
    all_acc = []
    all_p = []
    all_r = []
    all_f1 = []
    all_sum = []
    for batch in tqdm(test_iter):
        features, targets, summaries, doc_lens = vocab.make_features(batch)
        if use_gpu:
            probs = net(Variable(features).cuda(), doc_lens)
        else:
            probs = net(Variable(features), doc_lens)
        start = 0
        for doc_id, doc_len in enumerate(doc_lens):
            doc = batch['doc'][doc_id].split('\n')[:doc_len]
            stop = start + doc_len
            prob = probs[start:stop]
            hyp = []
            for _p, _d in zip(prob, doc):
                print(_p)
                print(_d)
                if _p > 0.5:
                    hyp.append(_d)
            if len(hyp) > 0:   
                print(hyp)
                all_sum.append("###".join(hyp))
            else:
                all_sum.append('')
            all_targets.append(targets[start:stop])
            all_probs.append(prob)
            start = stop 
    file_path_elems = end_test_file.split('/')
    file_name = 'TR-' + file_path_elems[len(file_path_elems) - 1]
    with open(os.path.join(args.output, file_name), mode='w', encoding='utf-8') as f:
        for text in all_sum:
            f.write(text.strip() + '\n')
    for item in all_probs:
        all_results.append( [1 if tmp > 0.5 else 0 for tmp in item.tolist()]) 
    print(len(all_results)) 
    print(len(all_targets)) 
    print(len(all_probs)) 
    for _1, _2, _3 in zip(all_results, all_targets, all_probs):
        _2 = _2.tolist() 
        _3 = _3.tolist()
        print("*"*3)
        print('probs : ', _3)
        print('results : ', _1)
        print('targets : ',  _2)
        tmp_acc = accuracy_score(_1, _2)
        tmp_p = precision_score(_1, _2)
        tmp_r = recall_score(_1, _2)
        tmp_f1 = f1_score(_1, _2)
        print('acc : ', tmp_acc)
        print('p : ', tmp_p)
        print('r : ', tmp_r)
        print('f1 : ', tmp_f1)
        all_acc.append(tmp_acc)
        all_p.append(tmp_p)
        all_r.append(tmp_r)
        all_f1.append(tmp_f1)
    print('all dataset acc : ', np.mean(all_acc))
    print('all dataset p : ', np.mean(all_p))
    print('all dataset r : ', np.mean(all_r))
    print('all dataset f1 : ', np.mean(all_f1))
    print('all results length : ', len(all_results))


if __name__ == '__main__':
    test()
