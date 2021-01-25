import pickle
import keras
import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from collections import OrderedDict
from solutions.utils.optimizer.pipeline import create_optimizer
from solutions.search.rank.selector.trainer import Trainer
from solutions.search.rank.selector.base_model import BaseModel

class SMN(BaseModel):
    def __init__(self, vocab, sequence_len, hidden_unit):
        # TODO
        self.max_turn = 1 # 上下文最大轮数
        # self.negative_samples = 4 # 负样本个数
        self.max_sentence_len = sequence_len # 文本最大长度
        self.word_embedding_size = 200 # TODO tencent embedding
        self.rnn_units = hidden_unit
        self.total_words = vocab.get_vocab_size() # TODO

        self.vocab = vocab
        sess_conf = tf.ConfigProto()
        self.session = tf.Session(config=sess_conf)

        self.initialized = False
        self._build_graph()
        # self._build_graph()

    def _build_graph(self):

        self.history = tf.placeholder(tf.int32, shape=(None, self.max_turn, self.max_sentence_len))
        self.history_len = tf.placeholder(tf.int32, shape=(None, self.max_turn))
        self.response = tf.placeholder(tf.int32, shape=(None, self.max_sentence_len))
        self.response_len = tf.placeholder(tf.int32, shape=(None,))
        self.y_true = tf.placeholder(tf.int32, shape=(None,))
        self.embedding_ph = tf.placeholder(tf.float32, shape=(self.total_words, self.word_embedding_size))

        word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.total_words, self.word_embedding_size), dtype=tf.float32, trainable=False) # TODO trainable
        self.embedding_init = word_embeddings.assign(self.embedding_ph) # 将placeholder 赋值给 value，run 的时候给placeholder 赋值时 同事更新 value
        all_utterance_embeddings = tf.nn.embedding_lookup(word_embeddings, self.history) # 使用lookup 将 utterance_ph 进行 embedding 表示
        response_embeddings = tf.nn.embedding_lookup(word_embeddings, self.response) # 使用lookup 将 response_ph 进行 embedding 表示
        sentence_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer()) # 初始化GRU 单元， 用于句子编码
        all_utterance_embeddings = tf.unstack(all_utterance_embeddings, num=self.max_turn, axis=1) # 改变信息表示方式
        all_utterance_len = tf.unstack(self.history_len, num=self.max_turn, axis=1) # 改变信息表示方式
        A_matrix = tf.get_variable('A_matrix_v', shape=(self.rnn_units, self.rnn_units), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        final_GRU = tf.nn.rnn_cell.GRUCell(self.rnn_units, kernel_initializer=tf.orthogonal_initializer()) # 最终GRU 单元，用于结果输出
        reuse = None

        response_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, response_embeddings, sequence_length=self.response_len, dtype=tf.float32, scope='sentence_GRU') # 返回值 (outputs， states)， 获得response的GRU表示
        self.response_embedding_save = response_GRU_embeddings
        response_embeddings = tf.transpose(response_embeddings, perm=[0, 2, 1])
        response_GRU_embeddings = tf.transpose(response_GRU_embeddings, perm=[0, 2, 1])
        matching_vectors = []
        for utterance_embeddings, utterance_len in zip(all_utterance_embeddings, all_utterance_len):
            matrix1 = tf.matmul(utterance_embeddings, response_embeddings) # 计算utterance 和 response 的匹配矩阵
            utterance_GRU_embeddings, _ = tf.nn.dynamic_rnn(sentence_GRU, utterance_embeddings, sequence_length=utterance_len, dtype=tf.float32, scope='sentence_GRU') # 获得每个utterance 的GRU 表示
            matrix2 = tf.einsum('aij,jk->aik', utterance_GRU_embeddings, A_matrix)  # TODO:check this
            matrix2 = tf.matmul(matrix2, response_GRU_embeddings)
            matrix = tf.stack([matrix1, matrix2], axis=3, name='matrix_stack')
            conv_layer = tf.layers.conv2d(matrix, filters=8, kernel_size=(3, 3), padding='VALID', kernel_initializer=tf.contrib.keras.initializers.he_normal(), activation=tf.nn.relu, reuse=reuse, name='conv')  # TODO: check other params
            pooling_layer = tf.layers.max_pooling2d(conv_layer, (3, 3), strides=(3, 3), padding='VALID', name='max_pooling')  # TODO: check other params
            matching_vector = tf.layers.dense(tf.contrib.layers.flatten(pooling_layer), 50, kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.tanh, reuse=reuse, name='matching_v')  # TODO: check wthether this is correct
            if not reuse:
                reuse = True
            matching_vectors.append(matching_vector)
        _, last_hidden = tf.nn.dynamic_rnn(final_GRU, tf.stack(matching_vectors, axis=0, name='matching_stack'), dtype=tf.float32,
                                           time_major=True, scope='final_GRU')  # TODO: check time_major

        with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):  
            self.logits = tf.layers.dense(last_hidden, 2, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='final_v')
            self.y_pred = tf.nn.softmax(self.logits, name = 'preds')
            self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_true,
                                                                      logits=self.logits))

        self.input_placeholder_dict = OrderedDict({
            "history": self.history,
            "history_len": self.history_len,
            "response": self.response,
            "response_len": self.response_len,
            "y_true": self.y_true
        })

        # Train Metric
        with tf.variable_scope("train_metrics"):
            self.train_metrics = {'loss': tf.metrics.mean(self.total_loss)}
        self.train_update_metrics = tf.group(*[op for _, op in self.train_metrics.values()])
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="train_metrics")
        self.train_metric_init_op = tf.variables_initializer(metric_variables)
        # Eval Metric
        with tf.variable_scope("eval_metrics"):
            self.eval_metrics = {'loss': tf.metrics.mean(self.total_loss)} # TODO
        self.eval_update_metrics = tf.group(*[op for _, op in self.eval_metrics.values()])
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="eval_metrics")
        self.eval_metric_init_op = tf.variables_initializer(metric_variables)

        tf.summary.scalar('loss', self.total_loss)
        self.summary_op = tf.summary.merge_all()

    def compile(self, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False):
        self.train_op = create_optimizer(self.total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

    def train_and_evaluate(self, train_generator, eval_generator, evaluator, embeddings, epochs=1, eposides=1,
                           save_dir=None, summary_dir=None, save_summary_steps=10):
        if not self.initialized: # TODO 每次都是 从头训练， 后续可根据情况选择是否要从头训练
            # self.bert_embedding.init_bert()
            self.session.run(self.embedding_init, feed_dict={self.embedding_ph: embeddings})
            self.session.run(tf.global_variables_initializer())

        Trainer._train_and_evaluate(self, train_generator, eval_generator, evaluator, epochs=epochs, eposides=eposides,
                                    save_dir=save_dir, summary_dir=summary_dir, save_summary_steps=save_summary_steps)
    
    def evaluate(self, batch_generator, evaluator):
        Trainer._evaluate(self, batch_generator, evaluator)
    
    def inference(self, batch_generator):
        return Trainer._single_inference(self, batch_generator)
