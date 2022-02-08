import json
import argparse, random, numpy, os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
from tqdm import tqdm
from time import time
import models
from utils.file_utils import merge_same_suf_text_file
from utils.Vocab import Vocab
from utils.Dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('-output', type=str, default='output', required=False)  
parser.add_argument('-input', type=str, default='input', required=False)  
parser.add_argument('-project', type=str, default='.', required=False)  
parser.add_argument('-gpu_id', type=str, default='0', required=False)  
parser.add_argument('-embed_dim', type=int, default=100, required=False)  
parser.add_argument('-embed_num', type=int, default=100, required=False)  
parser.add_argument('-pos_dim', type=int, default=50, required=False)  
parser.add_argument('-pos_num', type=int, default=100, required=False)  
parser.add_argument('-seg_num', type=int, default=20, required=False)  
parser.add_argument('-kernel_num', type=int, default=128, required=False)  # 100 # 卷积核数量
parser.add_argument('-kernel_sizes', type=str, default='2,3,4', required=False)  # 3,4,5 # 卷积核大小
parser.add_argument('-model', type=str, default='CNN_RNN', required=False)
parser.add_argument('-hidden_size', type=int, default=200, required=False)  # 200 # RNN 计算中的隐层维度
# train
parser.add_argument('-learning_rate', type=float, default=1e-4, required=False)  # lr
parser.add_argument('-batch_size', type=int, default=256, required=True)
parser.add_argument('-max_epoch', type=int, default=3, required=False)
parser.add_argument('-seed', type=int, default=1, required=False)
parser.add_argument('-seq_trunc', type=int, default=256, required=False)  # 50
parser.add_argument('-max_norm', type=float, default=1.0, required=False)
# device
parser.add_argument('-device', type=int, required=False)
args, unknown = parser.parse_known_args()


def train():
    print("*"*100)
    print("train begin")
    # use gpu
    use_gpu = args.device is not None
    if torch.cuda.is_available() and not use_gpu:
        print("WARNING: You have a CUDA device, should run with -device 0")
    if use_gpu:
        # set cuda device and seed
        torch.cuda.set_device(args.device)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # 路径准备
    embedding_file_path = os.path.join(args.project, "embedding.npz")
    vocab_file_path = os.path.join(args.project, "word2id.json")
    end_train_file = os.path.join(args.input, "train_files", "train.txt")
    train_files_dir = os.path.join(args.input, "train_files")

    # 合并同后缀文本文件
    merge_same_suf_text_file(train_files_dir, end_train_file, '.txt')

    print('Loading vocab,train and val dataset.Wait a second,please')
    embed = torch.Tensor(np.load(embedding_file_path)['arr_0'])  # embed = torch.Tensor(list(np.load(args.embedding)))
    with open(vocab_file_path) as f:
        word2id = json.load(f)
    vocab = Vocab(embed, word2id)
    with open(end_train_file) as f:
        examples = list()
        for line in tqdm(f):
            if line and not line.isspace():
                examples.append(json.loads(line))
    train_dataset = Dataset(examples)
    print(train_dataset[:1])

    args.embed_num = embed.size(0)  # 从embeding中读取维度
    args.embed_dim = embed.size(1)  #
    args.kernel_sizes = [int(ks) for ks in args.kernel_sizes.split(',')]
    net = getattr(models, args.model)(args, embed)
    if use_gpu:
        net.cuda()
    train_iter = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    criterion = nn.BCELoss()
    params = sum(p.numel() for p in list(net.parameters())) / 1e6
    print('#Params: %.1fM' % (params))

    min_loss = float('inf')
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    net.train()

    t1 = time()
    for epoch in range(1, args.max_epoch + 1):
        print("*"*10, 'epoch ', str(epoch), '*'*50)
        for i, batch in enumerate(train_iter):
            print("*"*10, 'batch', i, '*'*10)
            features, targets, _, doc_lens = vocab.make_features(batch, args.seq_trunc)
            features, targets = Variable(features), Variable(targets.float())
            if use_gpu:
                features = features.cuda()
                targets = targets.cuda()
            probs = net(features, doc_lens)
            loss = criterion(probs, targets)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(net.parameters(), args.max_norm)
            optimizer.step()
            net.save()
            print('Epoch: %2d Loss: %f' % (epoch, loss))
    t2 = time()
    print('Total Cost:%f h' % ((t2 - t1) / 3600))
    print("模型配置文件保存至输出文件夹")


if __name__ == '__main__':
    train()
