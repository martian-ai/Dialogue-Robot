# -*- coding: utf-8 -*-
"""
basic generator train function.
"""
import os
import torch
from data import DataPrecessForSentence, create_data_loader
from model import GenerateRNN, GenerateSeq2Seq
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertTokenizer, UNIMOTokenizer
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers.optimization import AdamW
from utils import load_vocab, train, validate


def main(train_file,
         dev_file,
         target_dir,
         epochs=10,
         batch_size=32,
         lr=2e-05,
         patience=3,
         max_grad_norm=10.0,
         checkpoint=None):
    """ 生成模型进行训练的主函数.
    Args:
        train_file (_type_): 
        dev_file (_type_): _description_
        target_dir (_type_): _description_
        epochs (int, optional): _description_. Defaults to 10.
        batch_size (int, optional): _description_. Defaults to 32.
        lr (_type_, optional): _description_. Defaults to 2e-05.
        patience (int, optional): _description_. Defaults to 3.
        max_grad_norm (float, optional): _description_. Defaults to 10.0.
        checkpoint (_type_, optional): _description_. Defaults to None.
    """
    # 使用bert 的tokenizer
    # bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',do_lower_case=True)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',
    #                                           do_lower_case=True)
    tokenizer = UNIMOTokenizer.from_pretrained('unimo-text-1.0')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # -------------------- Data loading ------------------- #

    # TODO 此部分替换

    # print("\t* Loading training data...")
    # train_data = DataPrecessForSentence(bert_tokenizer, train_file)
    # train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    # print("\t* Loading validation data...")
    # dev_data = DataPrecessForSentence(bert_tokenizer, dev_file)
    # dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)

    train_ds = load_dataset(
        '/Users/apollo/Documents/BotMVP-Mars/resources/dataset/DuReaderQG/',
        'DuReaderQG',
        splits="train",
        data_files='train.json')
    dev_ds = load_dataset(
        '/Users/apollo/Documents/BotMVP-Mars/resources/dataset/DuReaderQG/',
        'DuReaderQG',
        splits="validation",
        data_files='validation.json')

    # args = {}
    # args['max_seq_len'] = 512
    # args['max_target_len'] = 32
    # args['max_title_len'] = 32
    # args['batch_size'] = 16

    import argparse
    body = {
        'max_seq_len': 256,
        'max_target_len': 32,
        'max_title_len': 32,
        'batch_size': 16,
        "beam_size": 2,
        'enc_layers': 4,
        'enc_dropout': 0.1,
        'dec_layers': 4,
        'dec_hidden_size': 100,
        'dec_dropout': 0.1,
        'coverage': False,
        'copy_attn': False
    }
    args = argparse.Namespace(**body)

    train_ds, train_loader = create_data_loader(train_ds, tokenizer, args,
                                                "train")
    dev_ds, dev_loader = create_data_loader(dev_ds, tokenizer, args, "dev")

    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    # model = GenerateSeq2Seq().to(device)

    vocab = load_vocab("vocab.txt")
    model = GenerateRNN(args, device, vocab).to(device)

    # -------------------- Preparation for training  ------------------- #
    # 待优化的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  # todo 为何不优化
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        0.01
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.85,
                                                           patience=0)
    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    valid_losses = []
    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
    # Compute loss and accuracy before starting (or resuming) training.

    # TODO validate
    decode_output, summary, attn = validate(model, dev_loader)
    print(decode_output)
    print(summary)
    print(attn)
    # print(
    #     "\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}"
    #     .format(valid_loss, (valid_accuracy * 100), auc))
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training roberta model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))

        # TODO train result
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader,
                                                       optimizer, epoch,
                                                       max_grad_norm)
        train_losses.append(epoch_loss)
        # print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%".
        #       format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        # print("* Validation for epoch {}:".format(epoch))

        # TODO
        decode_output, summary, attn = validate(model, dev_loader)
        valid_losses.append(epoch_loss)
        # todo acc 改成bleu
        # print(
        #     "-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n"
        #     .format(epoch_time, epoch_loss, (epoch_accuracy * 100), epoch_auc))
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses
                }, os.path.join(target_dir, "best.pth.tar"))
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == "__main__":
    main("../../resources/dataset/clf_sentiemnt_car/train.tsv",
         "../../resources/dataset/clf_sentiemnt_car/dev.tsv",
         "models-seq2seq-test")
