# !/usr/bin/python
#  -*- coding: utf-8 -*-
# author : Apollo2Mars@gmail.com
# Problems : inputs and terms

import tensorflow as tf


class TextCNN(object):
    def __init__(self, args, tokenizer):
        self.vocab_size = len(tokenizer.word2idx) + 2
        self.seq_len = args.max_seq_len
        self.emb_dim = args.emb_dim
        self.hidden_dim = args.hidden_dim
        self.filters_num = args.filters_num
        self.filters_size = args.filters_size
        self.class_num = len(args.tag_list)
        self.learning_rate = args.lr

        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_len], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.float64, shape=[None, self.class_num], name='input_y')
        self.global_step = tf.placeholder(shape=(), dtype=tf.int32, name='global_step')
        self.keep_prob = tf.placeholder(tf.float64, name='keep_prob')

        self.embedding_matrix = tokenizer.embedding_matrix
        self.cnn()

    def focal_loss(self, pred, y, alpha=0.25, gamma=2):
        r"""Compute focal loss for predictions.
            Multi-labels Focal loss formula:
                FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                     ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
        Args:
         pred: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
         y: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
         alpha: A scalar tensor for focal loss alpha hyper-parameter
         gamma: A scalar tensor for focal loss gamma hyper-parameter
        Returns:
            loss: A (scalar) tensor representing the value of the loss function
        """
        zeros = tf.zeros_like(pred, dtype=pred.dtype)

        # For positive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so positive coefficient = z - p.
        pos_p_sub = tf.where(y > zeros, y - pred, zeros)  # positive sample 寻找正样本，并进行填充

        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = tf.where(y > zeros, zeros, pred)  # negative sample 寻找负样本，并进行填充
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))

        return tf.reduce_sum(per_entry_cross_ent)


    def focal_loss(self, logits, labels, gamma):
        '''
        :param logits:  [batch_size, n_class]
        :param labels: [batch_size]
        :return: -(1-y)^r * log(y)
        '''
        softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
        labels = tf.range(0, logits.shape[0]) * logits.shape[1] + labels
        prob = tf.gather(softmax, labels)
        weight = tf.pow(tf.subtract(1., prob), gamma)
        loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))
        return loss

    def cnn(self):
        with tf.device('/cpu:0'):
            inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x)

        with tf.name_scope('conv'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.filters_size):
                with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=tf.AUTO_REUSE):
                    conv = tf.layers.conv1d(inputs, self.filters_num, filter_size, name='conv1d', padding='same')
                    pooled = tf.reduce_max(conv, axis=[1], name='gmp')
                    pooled_outputs.append(pooled)
            outputs = tf.concat(pooled_outputs, 1)

        with tf.variable_scope("fully-connect", reuse=tf.AUTO_REUSE):
            fc = tf.layers.dense(outputs, self.hidden_dim, name='fc1')
            fc = tf.nn.relu(fc)
            fc = tf.nn.dropout(fc, self.keep_prob)

        with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
            logits = tf.layers.dense(fc, self.class_num, name='fc2')
            self.output_softmax = tf.nn.softmax(logits, name="output_softmax",)
            self.output_argmax = tf.argmax(self.output_softmax, 1, name='output_argmax')
            self.output_onehot = tf.one_hot(tf.argmax(self.output_softmax, 1, name='output_onehot'), self.class_num)

        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
            # loss =  tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.input_y)
            # tf.nn.sigmoid_cross_entropy_with_logits

            class_weights = tf.constant([1.0, 10.0, 15.0, 1.0])
            self.loss = tf.nn.weighted_cross_entropy_with_logits(logits=tf.cast(logits, tf.float64), targets=tf.cast(self.input_y, tf.float64), pos_weight=tf.cast(class_weights, tf.float64))
            loss = tf.reduce_mean(self.loss)

        with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
            self.learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                            global_step=self.global_step,
                                                            decay_steps=2,
                                                            decay_rate=0.95,
                                                            staircase=True)
            self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

            tf.summary.scalar('loss', loss)

