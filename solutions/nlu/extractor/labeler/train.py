#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019-09-03 16:35
# @Author  : apollo2mars
# @File    : train.py
# @Contact : apollo2mars@gmail.com
# @Desc    :


import os,sys,time,argparse,logging
import tensorflow as tf
import numpy as np
from pathlib import Path
from os import path
sys.path.append(path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.Tokenizer import build_tokenizer
from utils.Dataset_NER import Dataset_NER
from solutions.lexical_analysis.models.BIRNN_CRF import BIRNN_CRF
from solutions.lexical_analysis.evals.evaluate import get_results_by_line

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        """

        :param opt:
        """
        self.opt = opt
        logger.info("parameters for programming :  {}".format(self.opt))
        """
        parameters
        """
        self.max_seq_len = self.opt.max_seq_len
        self.epochs = opt.epochs
        self.label_list = self.opt.label_list
        self.dataset_file = opt.dataset_file 
        self.batch_size = opt.batch_size
        self.output_dir = opt.output_dir
        self.result_path = opt.result_path
        self.batch_size = opt.batch_size
        self.dataset_name = opt.dataset_name
        self.model_name = opt.model_name
        self.model_class = opt.model_class
        self.do_train = opt.do_train
        self.do_test = opt.do_test
        self.do_predict = opt.do_predict
        self.es = opt.es

        """
        build tokenizer
        """
        tokenizer = build_tokenizer(corpus_files=[opt.dataset_file['train'], opt.dataset_file['test']],corpus_type=opt.dataset_name,
                                    task_type='NER', embedding_type='tencent')
        self.tokenizer = tokenizer
        """
        build model and session
        """
        self.model = self.model_class(self.opt, tokenizer)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        session.run(tf.global_variables_initializer())
        self.session = session

        """
        set saver and max_to_keep 
        """
        self.saver = tf.train.Saver(max_to_keep=1)
        """
        dataset
        """
        self._set_dataset()

    def _set_dataset(self):
        # train
        self.trainset = Dataset_NER(self.dataset_file['train'], self.tokenizer, self.max_seq_len, 'entity', self.label_list)
        self.train_data_loader = tf.data.Dataset.from_tensor_slices({'text': self.trainset.text_list, 'label': self.trainset.label_list}).batch(self.batch_size).shuffle(10000)
        # test
        self.testset = Dataset_NER(self.dataset_file['test'], self.tokenizer, self.max_seq_len, 'entity', self.label_list)
        self.test_data_loader = tf.data.Dataset.from_tensor_slices({'text': self.testset.text_list, 'label': self.testset.label_list}).batch(self.opt.batch_size)

        # eval
        self.eval_data_loader = self.test_data_loader

         # predict
        if self.opt.do_predict is True:
            self.predictset = Dataset_NER(self.dataset_file['predict'], self.tokenizer, self.max_seq_len, 'entity', self.label_list)
            self.predict_data_loader = tf.data.Dataset.from_tensor_slices({'text': self.predictset.text_list, 'label': self.predictset.label_list}).batch(self.batch_size)
        
        # print(self.tokenizer.word2idx)
        print(self.trainset.text_list[0:10])
        print(self.trainset.label2idx)
        print(self.trainset.idx2label)

        logger.info('>> load data done')


    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):

        max_f1 = 0
        path = None

        for _epoch in range(self.epochs):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(_epoch))

            iterator = train_data_loader.make_one_shot_iterator()
            one_element = iterator.get_next()

            while True:
                try:
                    sample_batched = self.session.run(one_element)
                    inputs = sample_batched['text']
                    print("inputs >>> ", inputs)
                    labels = sample_batched['label']
                    print("inputs >>> ", labels)

                    model = self.model
                    _ = self.session.run(model.trainer, feed_dict={model.input_x: inputs, model.input_y: labels, model.global_step: _epoch, model.keep_prob: 1.0})
                    self.model = model

                except tf.errors.OutOfRangeError:
                    break
            
            val_p, val_r, val_f1 = self._evaluate_metric(val_data_loader)
            logger.info('>>>>>> val_p: {:.4f}, val_r:{:.4f}, val_f1: {:.4f}'.format(val_p, val_r, val_f1))

            if val_f1 > max_f1:
                logger.info(">> val f1 > max_f1, enter save model part")
                """
                update max_f1
                """
                max_f1 = val_f1
                """
                output path for pb and ckpt model
                """
                if not os.path.exists(self.output_dir):
                    os.mkdir(self.output_dir)
                ckpt_path = os.path.join(self.output_dir, '{0}_{1}'.format(self.model_name, self.dataset_name), '{0}'.format(round(val_f1, 4)))
                """
                flag for early stopping
                """
                last_improved = _epoch
                """
                save ckpt model
                """
                self.saver.save(sess=self.session, save_path=ckpt_path)
                logger.info('>> ckpt model saved in : {}'.format(ckpt_path))
                """
                save pb model
                """
                
                pb_dir = os.path.join(self.output_dir,'{0}_{1}'.format(self.model_name,  self.dataset_name))

                from tensorflow.python.framework import graph_util
                trained_graph = graph_util.convert_variables_to_constants(self.session,
                                                          self.session.graph_def,
                                                          output_node_names=['outputs'])
                tf.train.write_graph(trained_graph, pb_dir, "model.pb", as_text=False)
                logger.info('>> pb model saved in : {}'.format(pb_dir))

            if abs(last_improved - _epoch) > self.es:
                logging.info(">> too many epochs not imporve, break")

            if abs(last_improved - _epoch) > self.es:
                logging.info(">> too many epochs not imporve, break")
                break

        return ckpt_path

    def _test(self):
        ckpt = tf.train.get_checkpoint_state(self.opt.output_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            test_p, test_r, test_f1 = self._evaluate_metric(self.test_data_loader)
            logger.info('>> test_p: {:.4f}, test_r:{:.4f}, test_f1: {:.4f}'.format(test_p, test_r, test_f1))
        else:
            logger.info('@@@ Error:load ckpt error')

    def _predict(self, data_loader):
        ckpt = tf.train.get_checkpoint_state(os.path.join(self.output_dir, '{0}_{1}'.format(self.model_name, self.dataset_name)))

        if ckpt and ckpt.model_checkpoint_path:
            logger.info('>>> load ckpt model path for predict'.format(ckpt.model_checkpoint_path))
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            self._output_result(data_loader)
            logger.info('>> predict done')
        else:
            logger.info('@@@ Error:load ckpt error')

    def _output_result(self, data_loader):

        t_texts_all, t_targets_all, t_outputs_all = [], [], []
        iterator = data_loader.make_one_shot_iterator()
        one_element = iterator.get_next()

        print(self.trainset.label2idx)

        def convert_text(encode_list):
            return [self.tokenizer.idx2word[item] for item in encode_list if item not in [self.tokenizer.word2idx["<PAD>"]]]

        def convert_label(encode_list):
            return [self.trainset.idx2label[item] for item in encode_list if item not in [self.trainset.label2idx["<PAD>"]]]

        while True:
            try:
                sample_batched = self.session.run(one_element)
                inputs = sample_batched['text']
                targets = sample_batched['label']

                model = self.model
                outputs = self.session.run(model.outputs, feed_dict={model.input_x: inputs, model.input_y: targets,
                                                                     model.global_step: 1, model.keep_prob: 1.0})

                inputs = list(map(convert_text, inputs))
                targets = list(map(convert_label, targets))
                outputs = list(map(convert_label, outputs))

                t_texts_all.extend(inputs)
                t_targets_all.extend(targets)
                t_outputs_all.extend(outputs)

            except tf.errors.OutOfRangeError:
                if self.opt.do_predict is True or self.opt.do_test is True and self.opt.do_train is False:
                    with open(os.path.join(self.output_dir, '{0}_{1}'.format(self.model_name, self.dataset_name), 'result.log'), mode='w', encoding='utf-8') as f:
                        for item in t_outputs_all:
                            f.write(str(item) + '\n')

                break

    def _evaluate_metric(self, data_loader):

        t_texts_all, t_targets_all, t_outputs_all = [], [], []
        iterator = data_loader.make_one_shot_iterator()
        one_element = iterator.get_next()

        def convert_text(encode_list):
            return [self.tokenizer.idx2word[item] for item in encode_list if item not in [self.tokenizer.word2idx["<PAD>"] ]]

        def convert_label(encode_list):
            return [self.trainset.idx2label[item] for item in encode_list if item not in [self.trainset.label2idx["<PAD>"]]]
            # return [self.trainset.idx2label[item] for item in encode_list if item not in [self.trainset.label2idx["<PAD>"]] ]

        while True:
            try:
                sample_batched = self.session.run(one_element)
                inputs = sample_batched['text']
                targets = sample_batched['label']

                model = self.model
                outputs = self.session.run(model.outputs, feed_dict={model.input_x: inputs, model.input_y: targets, model.global_step: 1, model.keep_prob: 1.0})
                
                inputs = list(map(convert_text, inputs))
                targets = list(map(convert_label, targets))
                outputs = list(map(convert_label, outputs))

                t_texts_all.extend(inputs)
                t_targets_all.extend(targets)
                t_outputs_all.extend(outputs)

            except tf.errors.OutOfRangeError:
                if self.opt.do_test is True and self.opt.do_train is False:
                    with open(os.path.join(self.output_dir, '{0}_{1}'.format(self.model_name, self.dataset_name), 'result.log'), mode='w', encoding='utf-8') as f:
                        for item in t_outputs_all:
                            f.write(str(item) + '\n')

                break

        #print(t_texts_all)
        #print(t_targets_all)
        #print(t_outputs_all)

        p, r, f1 = get_results_by_line(t_texts_all, t_targets_all, t_outputs_all)

        return p, r, f1

    def run(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.opt.learning_rate)

        if self.opt.do_train is True and self.opt.do_test is True:
            best_model_path = self._train(None, optimizer, self.train_data_loader, self.test_data_loader)
            self.saver.restore(self.session, best_model_path)
            test_p, test_r, test_f1 = self._evaluate_metric(self.test_data_loader)
            logger.info('>> test_p: {:.4f}, test_r:{:.4f}, test_f1: {:.4f}'.format(test_p, test_r, test_f1)) 
        elif self.do_train is False and self.do_test is True: 
            self._test() 
        elif self.do_predict is True:
            self._predict(self.predict_data_loader) 
        else: 
            logger.info("@@@ Not Include This Situation") 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='promotion', help='air-purifier, refrigerator, shaver, promotion')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--result_path', type=str)

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_seq_len', type=str, default=64)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=500, help='hidden dim of dense')
    parser.add_argument('--es', type=int, default=5, help='early stopping epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--model_name', type=str, default='birnn_crf')
    parser.add_argument('--inputs_cols', type=str, default='text')
    parser.add_argument('--initializer', type=str, default='???')
    parser.add_argument('--optimizer', type=str, default='adam')

    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--do_test', action='store_true', default=False)
    parser.add_argument('--do_predict', action='store_true', default=False)

    args = parser.parse_args()

    prefix_path = '/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/'
    prefix_path_1 = '/export/home/sunhongchao1/Workspace-of-NLU/corpus/nlu/'
    train_path = '/slot/train.txt'
    test_path = '/slot/test.txt'
    predict_path = '/slot/predict.txt'

    dataset_files = {
        'promotion':{
            'train': prefix_path_1 + args.dataset_name + train_path,
            'eval': prefix_path_1 + args.dataset_name + test_path,
            'test': prefix_path_1 + args.dataset_name + test_path,
            'predict': prefix_path_1 + args.dataset_name + predict_path},
        'frying-pan': {
            'train': prefix_path + args.dataset_name + train_path,
            'eval': prefix_path + args.dataset_name + test_path,
            'test': prefix_path + args.dataset_name + test_path,
            'predict': prefix_path + args.dataset_name + predict_path},
        'vacuum-cleaner': {
            'train': prefix_path + args.dataset_name + train_path,
            'eval': prefix_path + args.dataset_name + test_path,
            'test': prefix_path + args.dataset_name + test_path,
            'predict': prefix_path + args.dataset_name + predict_path},
        'air-purifier': {
            'train': prefix_path + args.dataset_name + train_path,
            'eval': prefix_path + args.dataset_name + test_path,
            'test': prefix_path + args.dataset_name + test_path,
            'predict': prefix_path + args.dataset_name + predict_path},
        'shaver': {
            'train': prefix_path + args.dataset_name + train_path,
            'eval': prefix_path + args.dataset_name + test_path,
            'test': prefix_path + args.dataset_name + test_path,
            'predict': prefix_path + args.dataset_name + predict_path},
        'electric-toothbrush': {
            'train': prefix_path + args.dataset_name + train_path,
            'eval': prefix_path + args.dataset_name + test_path,
            'test': prefix_path + args.dataset_name + test_path,
            'predict': prefix_path + args.dataset_name + predict_path},
    }

    prefix_list = ['B', 'I', 'E', 'S']
    # prefix_list = ['B', 'I', 'E', 'S']

    promotion_type = ['DATE', 'PRODUCT', 'BRAND', 'SHOP', 'COLOR', 'PRICE',
                      'AMOUT', 'ATTRIBUTE']

    promotion_list_1 = [ item_prefix + '-' + item_promotion for item_prefix in
                      prefix_list for item_promotion in promotion_type]
    promotion_list = []
    promotion_list.append("<PAD>")
    promotion_list.extend(promotion_list_1)
    promotion_list.append("O")

    comment_list = ['<PAD>', 'O', 'B-3', 'I-3']

    label_lists = {
        'promotion':promotion_list,
        'shaver':comment_list,
        'vacuum-cleaner':"'entity'",
        'air-purifier':"'entity'",
        'electric-toothbrush':"'entity'",
        'frying-pan':"'entity'",
    }

    model_classes = {
        'birnn_crf': BIRNN_CRF,
        # 'bert_cnn': BERT_BIRNN_CRF
    }

    inputs_cols = {
        'bert_birnn_crf': ['text'],
        'birnn_crf': ['text']
    }

    initializers = {
        'random_normal': tf.random_normal_initializer,  # 符号标准正太分布的tensor
        'truncted_normal': tf.truncated_normal_initializer,  # 截断正太分布
        'random_uniform': tf.random_uniform_initializer,  # 均匀分布
        # tf.orthogonal_initializer() 初始化为正交矩阵的随机数，形状最少需要是二维的
        # tf.glorot_uniform_initializer() 初始化为与输入输出节点数相关的均匀分布随机数
        # tf.glorot_normal_initializer（） 初始化为与输入输出节点数相关的截断正太分布随机数
        # tf.variance_scaling_initializer() 初始化为变尺度正太、均匀分布
    }

    optimizers = {
        'adadelta': tf.train.AdadeltaOptimizer,  # default lr=1.0
        'adagrad': tf.train.AdagradOptimizer,  # default lr=0.01
        'adam': tf.train.AdamOptimizer,  # default lr=0.001
        'adamax': '',  # default lr=0.002
        'asgd': '',  # default lr=0.01
        'rmsprop': '',  # default lr=0.01
        'sgd': '',
    }
    args.model_class = model_classes[args.model_name]
    args.dataset_file = dataset_files[args.dataset_name]
    args.inputs_cols = inputs_cols[args.model_name]
    args.label_list = label_lists[args.dataset_name]
    args.optimizer = optimizers[args.optimizer]
    log_dir = Path('outputs/logs')
    if not log_dir.exists():
        Path.mkdir(log_dir, parents=True)
    log_file = log_dir/'{}-{}-{}.log'.format(args.model_name, args.dataset_name, time.strftime("%y%m%d-%H%M", time.localtime(time.time())))
    logger.addHandler(logging.FileHandler(log_file))
    ins = Instructor(args)
    ins.run()


if __name__ == "__main__":
    main()
