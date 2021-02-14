# https://www.iteye.com/blog/cherishlc-2348962

import os, sys, time, argparse, logging
import tensorflow as tf
import numpy as np
from os import path
sys.path.append(path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn import metrics

from pathlib import Path
from utils.Dataset_CLF import Dataset_CLF, preprocess_with_label, preprocess_without_label
from utils.Tokenizer import build_tokenizer
from solutions.classification.models.TextCNN import TextCNN
from solutions.classification.models.BERT_CNN import BERTCNN

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, args):
        """
        :param opt: parameters for build model
        """
        self.args = args
        logger.info("parameters for programming :  {}".format(self.args))

        """
        parameter
        """
        self.optimizer = args.optimizer
        self.initializer = args.initializer

        self.tag_list = args.tag_list
        self.output_dir = args.output_dir
        self.result_file = args.result_file
        self.batch_size = args.batch_size
        self.dataset_file = args.dataset_file
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.model_class = args.model_class

        self.max_seq_len = args.max_seq_len
        self.lr = args.lr
        self.gpu_list = [int(item) for item in args.gpu_list.split(",")]
        self.epochs = args.epochs
        self.es = args.es

        self.do_train = args.do_train
        self.do_test = args.do_test
        self.do_predict_batch = args.do_predict_batch
        self.do_predict_single = args.do_predict_single

        """
        build tokenizer
        """
        tokenizer = build_tokenizer(corpus_files=[self.dataset_file['train'], self.dataset_file['test']], corpus_type=self.dataset_name, task_type='CLF', embedding_type='tencent')
        self.tokenizer = tokenizer

        """
        set session
        """

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        self.session = session

        """
        set model
        """

        # model = self.model_class(self.args, self.tokenizer)
        
        """
        set model list
        """
        #model_list = []
       
        #tower_grads = []

        #for gpu_id in self.gpu_list:
        #    with tf.device('/gpu:%d' % gpu_id):
        #        print('tower:%d...' % gpu_id)
        #        with tf.name_scope('tower_%d' % gpu_id) as scope:
        #            model_list.append(model)

        #self.model_list = model_list


        """
        dataset
        """
        self._set_dataset()

    def _set_dataset(self):
        """
        set dataset for train, dev, test, predict
        :return:
        """
        """
        dataset build
        """
        dataset_clf = Dataset_CLF(tokenizer=self.tokenizer, max_seq_len=self.max_seq_len, data_type='normal', tag_list=self.tag_list)
        train_text_list, train_label_list = preprocess_with_label(dataset_clf, self.dataset_file['train'])
        test_text_list, test_label_list = preprocess_with_label(dataset_clf, self.dataset_file['test'])
        
        if self.do_predict_single or self.do_predict_batch:
            predict_text_list = preprocess_without_label(dataset_clf, self.dataset_file['predict'])

        """
        imbalance
        """
        from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

        # ros = RandomOverSampler(random_state=0)
        # x_resampled, y_resampled = ros.fit_sample(self.trainset.text_list, self.trainset.label_list)

        # x_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(self.trainset.text_list, self.trainset.label_list)
        # print(">>> y_resampled", y_resampled[:4])
        # x_resampled = self.trainset.text_list
        # y_resampled = self.trainset.label_list

        """
        train set
        """
        self.train_data_loader = tf.data.Dataset.from_tensor_slices({'text': train_text_list,
                                                                     'label':train_label_list}).batch(self.batch_size).shuffle(100000)
        
        """
        test and dev set
        """
        self.test_data_loader = tf.data.Dataset.from_tensor_slices({'text': test_text_list, 'label': test_label_list}).batch(self.batch_size)
        self.val_data_loader = self.test_data_loader

        """
        predict set
        """
        if self.do_predict_batch is True or self.do_predict_single is True:
            self.predict_data_loader = tf.data.Dataset.from_tensor_slices({'text': predict_text_list, 'label': self.predict_label_list}).batch(self.batch_size)

        logger.info('>> load data done')


    def _test(self,data_loader):
        """
        load model and evaluate test data
        output metric on test data
        last checkpoint is best model
        :return:
        """
        ckpt = tf.train.get_checkpoint_state(os.path.join(self.output_dir, '{0}_{1}'.format(self.model_name, self.dataset_name)))
        if ckpt and ckpt.model_checkpoint_path:
            logger.info('>>> load ckpt model path for test', ckpt.model_checkpoint_path)
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            test_p, test_r, test_f1 = self._evaluate_metric(data_loader)
            logger.info('>> test_p: {:.4f}, test_r:{:.4f}, test_f1: {:.4f}'.format(test_p, test_r, test_f1))
            logger.info('>> test done')
        else:
            logger.info('@@@ Error:load ckpt error')

    def _predict(self, data_loader):
        """
        load model and predict predict data
        output predict results, not metric
        no label or default label
        last checkpoint is best model
        :return:
        """
        ckpt = tf.train.get_checkpoint_state(os.path.join(self.output_dir, '{0}_{1}'.format(self.model_name, self.dataset_name)))

        if ckpt and ckpt.model_checkpoint_path:
            logger.info('>>> load ckpt model path for predict', ckpt.model_checkpoint_path)
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            self._output_result(data_loader)
            logger.info('>> predict done')
        else:
            logger.info('@@@ Error:load ckpt error')


    def _output_result(self, data_loader):
        """
        output result for predict
        """
        t_targets_all, t_outputs_all = [], []
        iterator = data_loader.make_one_shot_iterator()
        one_element = iterator.get_next()

        while True:
            try:
                sample_batched = self.session.run(one_element)
                inputs = sample_batched['text']
                model = self.model
                outputs = self.session.run(model.output_softmax, feed_dict={model.input_x: inputs, model.global_step: 1, model.keep_prob: 1.0})
                t_targets_all.extend(targets)
                t_outputs_all.extend(outputs)

            except tf.errors.OutOfRangeError:
                with open(os.path.join(self.output_dir, '{0}_{1}'.format(self.model_name, self.dataset_name), 'result.log'), mode='w', encoding='utf-8') as f:
                    for item in t_outputs_all:
                        f.write(str(item) + '\n')

                break

        else:
            logger.info('@@@ Error:load ckpt error')

    def _evaluate_metric(self, data_loader):
        """
        use model calculate metric
        """
        t_targets_all, t_outputs_all = [], []
        iterator = data_loader.make_one_shot_iterator()
        one_element = iterator.get_next()

        model = self.model_list[0]

        while True:
            try:
                sample_batched = self.session.run(one_element)    
                inputs = sample_batched['text']
                labels = sample_batched['label']

                outputs = self.session.run(model.output_onehot, feed_dict={model.input_x: inputs, model.input_y: labels, model.global_step: 1, model.keep_prob: 1.0})
                t_targets_all.extend(labels)
                t_outputs_all.extend(outputs)

            except tf.errors.OutOfRangeError:
                if self.do_test is True and self.do_train is False:
                    with open(self.result_file,  mode='w', encoding='utf-8') as f:
                        for item in t_outputs_all:
                            f.write(str(item) + '\n')

                break

        flag = 'weighted'

        t_targets_all = np.asarray(t_targets_all)
        t_outputs_all = np.asarray(t_outputs_all)

        p = metrics.precision_score(t_targets_all, t_outputs_all,  average=flag)
        r = metrics.recall_score(t_targets_all, t_outputs_all,  average=flag)
        f1 = metrics.f1_score(t_targets_all, t_outputs_all,  average=flag)

        t_targets_all = [np.argmax(item) for item in t_targets_all]
        t_outputs_all = [np.argmax(item) for item in t_outputs_all]

        logger.info(metrics.classification_report(t_targets_all, t_outputs_all, labels=list(range(len(self.tag_list))), target_names=list(self.tag_list)))
        logger.info(metrics.confusion_matrix(t_targets_all, t_outputs_all))        
        
        return p, r, f1

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = [g for g, _ in grad_and_vars]
            # Average over the 'tower' dimension.
            grad = tf.stack(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
            return average_grads


    def _train_multi_gpu(self, gpu_list, optimizer, train_data_loader, val_data_loader):
        """
        :param criterion: no use, select loss function
        :param optimizer: no use, select optimizer
        :param train_data_loader: ..
        :param val_data_loader: ..
        :return: best model ckpt path
        """
        max_f1, path, model_list, tower_grads = 0, None, [], []

        # for gpu_id in gpu_list:
        for gpu_id in gpu_list:
            with tf.device('/gpu:%d' % gpu_id):
                print('tower:%d...' % gpu_id)
                with tf.name_scope('tower_%d' % gpu_id) as scope:
                    model = self.model_class(self.args, self.tokenizer)
                    model_list.append(model)
                    grads = optimizer.compute_gradients(model.loss)
                    tower_grads.append(grads)

                    tf.get_variable_scope().reuse_variables()  # we reuse variables here after we initiate one model
                    
        with tf.variable_scope("tower_gradient", reuse=tf.AUTO_REUSE):
            apply_gradient_op = optimizer.apply_gradients(self.average_gradients(tower_grads))

        self.saver = tf.train.Saver(max_to_keep=1)

        self.session.run(tf.global_variables_initializer())

        logger.info("$" * 50)
        logger.info(" >>>>>> train begin")
        for _epoch in range(self.epochs):
            logger.info('>' * 50)
            logger.info(' >>>>>> epoch: {}'.format(_epoch))

            iterator = train_data_loader.make_one_shot_iterator()
            one_element = iterator.get_next()

            while True:
                try:
                    feed_dict = {}
                    for idx, gpu_id in enumerate(gpu_list):
                        print(gpu_id)
                        sample_batched = self.session.run(one_element)
                        inputs = sample_batched['text']
                        labels = sample_batched['label']
                        feed_dict[model_list[idx].input_x] = inputs
                        feed_dict[model_list[idx].input_y] = labels
                        feed_dict[model_list[idx].global_step] = 1
                        feed_dict[model_list[idx].keep_prob] = 1.0
                        
                    # gradient, loss = self.session.run([apply_gradient_op, loss], feed_dict=feed_dict)
                    gradient = self.session.run(apply_gradient_op, feed_dict=feed_dict)
                    # TODO loss

                except tf.errors.OutOfRangeError:
                    break
            
            print("$" * 50)
            print("length of model list", len(model_list))
            print("$" * 50)
            self.model_list = model_list

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
                pb_dir = os.path.join(self.output_dir, '{0}_{1}'.format(self.model_name, self.dataset_name))
                
                print("*"*50)
                print("graph node")
                print([n.name for n in self.session.graph_def.node])

                from tensorflow.python.framework import graph_util
                trained_graph = graph_util.convert_variables_to_constants(self.session,
                                                          self.session.graph_def,
                                                          output_node_names=['tower_0/logits/output_argmax'])
                # 此处可以修改为 多卡汇总的 node
                tf.train.write_graph(trained_graph, pb_dir, "model.pb", as_text=False)
                logger.info('>> pb model saved in : {}'.format(pb_dir))

            if abs(last_improved - _epoch) > self.es:
                logging.info(">> too many epochs not imporve, break")
                break
        if ckpt_path is None:
            logging.warning(">> return path is None")

        return ckpt_path

    def run(self):
        optimizer = self.optimizer(learning_rate=self.lr)

        best_model_path = ""
        if self.do_train is True:
            best_model_path = self._train_multi_gpu(self.gpu_list, optimizer, self.train_data_loader, self.test_data_loader)

        if self.do_test in True:
            # best_model_path = self._train(None, optimizer, self.train_data_loader, self.test_data_loader)
            self.saver.restore(self.session, best_model_path)
            test_p, test_r, test_f1 = self._evaluate_metric(self.test_data_loader)
            logger.info('>> test_p: {:.4f}, test_r:{:.4f}, test_f1: {:.4f}'.format(test_p, test_r, test_f1))

        elif self.do_predict_batch is True:
            self._predict(self.predict_data_loader)
        elif self.do_predict_single is True:
            self._predict_single(self.single_text)
        else:
            logger.info("@@@ Not Include This Situation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='promotion', help='air-purifier, refrigerator, shaver')

    parser.add_argument('--emb_file', type=str, default='embedding.text')
    parser.add_argument('--vocab_file', type=str, default='vacab.txt')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--result_file', type=str, default='results.txt')
    parser.add_argument('--tag_list', type=str)

    parser.add_argument('--model_name', type=str, default='text_cnn')
    parser.add_argument('--inputs_cols', type=str, default='text')
    parser.add_argument('--initializer', type=str, default='random_normal')
    parser.add_argument('--optimizer', type=str, default='adam')

    parser.add_argument('--emb_dim', type=int, default='200')
    parser.add_argument('--gpu_list', type=str, default='0,1,2')
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--epochs', type=int, default=100, help='epochs for trianing')
    parser.add_argument('--es', type=int, default=3, help='early stopping epochs')

    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dim of dense')
    parser.add_argument('--filters_num', type=int, default=256, help='number of filters')
    parser.add_argument('--filters_size', type=int, default=[4,3,2], help='size of filters')

    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_test', action='store_true', default=False)
    parser.add_argument('--do_predict_batch', action='store_true',
                        default=False)
    parser.add_argument('--do_predict_single', action='store_true',
                        default=False)
     
    args = parser.parse_args()
    
    model_classes = {
        'text_cnn': TextCNN,
        'bert_cnn': BERTCNN
    }

    prefix_path = '/export/home/sunhongchao1/Workspace-of-NLU/corpus/nlu'

    dataset_files = {
        'promotion': {
            'train': os.path.join(prefix_path, args.dataset_name,'clf/train.txt'),
            'dev': os.path.join(prefix_path, args.dataset_name, 'clf/dev.txt'),
            'test': os.path.join(prefix_path, args.dataset_name, 'clf/test.txt'),
            'predict': os.path.join(prefix_path, args.dataset_name, 'clf/predict.txt')},
    }

    tag_lists ={
        'promotion': ['A', 'B', 'C', "D"],
    }

    inputs_cols = {
        'text_cnn': ['text'],
        'bert_cnn': ['text']
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
    args.tag_list = tag_lists[args.dataset_name]
    args.initializer = initializers[args.initializer]
    args.optimizer = optimizers[args.optimizer]

    log_dir = Path('outputs/logs')
    if not log_dir.exists():
        Path.mkdir(log_dir, parents=True)
    log_file = log_dir / '{}-{}-{}.log'.format(args.model_name, args.dataset_name, time.strftime("%y%m%d-%H%M", time.localtime(time.time())))
    logger.addHandler(logging.FileHandler(log_file))
    ins = Instructor(args)
    ins.run()


if __name__ == "__main__":
    main()
