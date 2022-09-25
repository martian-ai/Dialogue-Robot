#
# 基于bert 模型的文本分类代码构建
#

import logging
import os
import sys
from time import localtime, strftime

import torch
import torch.nn as nn
from pyexpat import model
from torch.utils.data import DataLoader, random_split

sys.path.append('..') 
# from modules.learner.nn_with_bert.function.modeling import BertConfig
# from modules.utils_tokenizer.bert import Tokenizer4Bert 
# from modules.utils_dataset.CLFDataset import CLFDataset

from modules.alpha_nn.modeling_bert import BertConfig
from modules.modules_args import opt
from modules.utils.dataset.CLFDataset import CLFDataset
from modules.utils.tokenizer.bert import Tokenizer4Bert

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if not os.path.exists(self.opt.log_dir):  # TODO
            os.makedirs(self.opt.log_dir)
        log_file = self.opt.log_dir + '/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
        logger.addHandler(logging.FileHandler(log_file))

        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.tokenizer_type)

        config = BertConfig(vocab_size_or_config_json_file=opt.config_file)
        self.model = opt.model_class(config, opt).to(opt.device)

        self.trainset = CLFDataset(opt.dataset_file['train'], tokenizer, label_dict=opt.label_dict)
        assert 0 <= opt.valset_ratio < 1

        self.testset = self.trainset
        self.valset = self.testset

        if opt.valset_ratio > 0:
            logger.info(' testset == valset, trainset and valset split by valset_ratio ')
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
            self.testset = self.valset
        else:
            logger.info(' trainset == testset == valset')
            self.testset = self.trainset
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    # def _reset_params(self):
    #     """
    #     作用
    #     """
    #     for child in self.model.children():
    #         if type(child) != BertModel:  # skip bert params
    #             for p in child.parameters():
    #                 if p.requires_grad:
    #                     if len(p.shape) > 1:
    #                         self.opt.initializer(p)
    #                     else:
    #                         stdv = 1. / math.sqrt(p.shape[0])
    #                         torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        """
        具有early stopping 能力

        """
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        path = None
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = batch['polarity'].to(self.opt.device)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc = self._evaluate_clf_acc(val_data_loader)
            logger.info('> val_acc: {:.4f}'.format(val_acc))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                if not os.path.exists('state_dict'):  # TODO sava path 需要制定
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc_{2}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break

        return path

    def _evaluate_clf_acc(self, data_loader):
        """
        进行分类能力的评估
        """
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['polarity'].to(self.opt.device) 
                t_outputs = self.model(t_inputs)
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)
                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        return acc

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        test_acc = self._evaluate_clf_acc(test_data_loader)
        logger.info('>> test_acc: {:.4f}'.format(test_acc))


def main():
    
    print("*"*100)
    print(opt)
    print("*"*100)
    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    print(torch.cuda.is_available())
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
    main()
