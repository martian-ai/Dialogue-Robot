import sys
sys.path.append(".")
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
import pandas as pd


from modules.alpha_nn.modeling_bert import BertClassifier
from modules.alpha_nn.modeling_base import MLP
from modules.dataset.clf_toutiao import Dataset_Toutiao

# 训练模型
def train(model, train_data, val_data, learning_rate, epochs, batch_size, labels, tokenizer): # TODO 参数估计要减少
    # 通过Dataset类获取训练和验证集
    train, val = Dataset_Toutiao(train_data, labels, tokenizer), Dataset_Toutiao(val_data, labels, tokenizer)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size)
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    # 开始进入训练循环
    for epoch_num in range(epochs):
        total_acc_train, total_loss_train = 0, 0
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            # 通过模型得到输出
            output = model(input_id, mask)
            # 计算损失
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        # ------ 验证模型 -----------
        # 定义两个变量，用于存储验证集的准确率和损失
        total_acc_val = 0
        total_loss_val = 0
        # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for val_input, val_label in val_dataloader:
                # 如果有GPU，则使用GPU，接下来的操作同训练
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                output = model(input_id, mask)
                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        print(
            f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')
 
# 评估模型
def evaluate(model, test_data, labels, tokenizer):
    test = Dataset_Toutiao(test_data, labels, tokenizer)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=16)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')


if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('./resources/embedding/bert-base-chinese')
    #  读取数据集
    labels = {'news_story':0,
            'news_culture':1,
            'news_entertainment':2,
            'news_sports':3,
            'news_finance':4,
            'news_house':5,
            'news_car':6,
            'news_edu':7,
            'news_tech':8,
            'news_military':9,
            'news_travel':10,
            'news_world':11,
            'stock':12,
            'news_agriculture':13,
            'news_game':14
            }

    df = pd.read_csv('modules/dataset/toutiao_cat_data.csv')
    df = df.head(1600)  #时间原因，我只取了1600条训练

    np.random.seed(112)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                        [int(.8*len(df)), int(.9*len(df))])  # 拆分为训练集、验证集和测试集，比例为 80:10:10。

    EPOCHS = 10  # 训练轮数
    # model = BertClassifier()  # 定义的模型
    model = MLP(input_n=128, output_n=15)
    LR = 1e-6  # 学习率
    Batch_Size = 4  # 看你的GPU，要合理取值
    train(model, df_train, df_val, LR, EPOCHS, Batch_Size, labels, tokenizer)
    torch.save(model.state_dict(), 'BERT-toutiao.pt')
    evaluate(model, df_test, labels, tokenizer)
