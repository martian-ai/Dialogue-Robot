import tensorflow as tf


class BIRNN_CRF(object):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        self.batch_size = args.batch_size
        self.initializer = tf.random_uniform_initializer
        self.is_training = False
        self.is_attention = False
        self.is_crf = True
        self.bidirectional = True
        self.dropout_rate = 0.1
        self.num_layers = 1

        self.rnn_outputs = None
        self.att_outputs = None

        self.seq_len = args.max_seq_len
        self.emb_dim = 200  # ???
        self.hidden_dim = args.hidden_dim
        self.attention_dim = 256
        self.class_num = len(args.label_list)
        self.learning_rate = args.learning_rate

        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_len], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, self.seq_len], name='input_y')
        self.global_step = tf.placeholder(shape=(), dtype=tf.int32, name='global_step')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.embedding_matrix = tokenizer.embedding_matrix

        self.birnn_crf()

    def char_embedding(self):
        pass

    def birnn_crf(self):

        with tf.device('/cpu:0'):

            mask = tf.reduce_sum(tf.sign(self.input_x), reduction_indices=1)
            mask = tf.cast(mask, tf.int32)

            inputs_emb = tf.nn.embedding_lookup(self.embedding_matrix, self.input_x)  # 维度增加

        if self.bidirectional:
            with tf.variable_scope("bi-lstm"):

                if self.is_training:
                     cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=(1 - self.dropout_rate))
                     cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=(1 - self.dropout_rate))
                else:
                    cell_fw = tf.nn.rnn_cell.GRUCell(self.hidden_dim)
                    cell_bw = tf.nn.rnn_cell.GRUCell(self.hidden_dim)

                (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                                    cell_bw=cell_bw,
                                                                                    inputs=inputs_emb,
                                                                                    sequence_length=mask,
                                                                                    dtype=tf.float64)
                outputs = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
                outputs = tf.cast(outputs, dtype=tf.float32)

                if self.is_training:
                    self.outputs = tf.nn.dropout(outputs, 0.1)
                else:
                    self.outputs = outputs

#        if self.is_attention:
#            H1 = tf.reshape(self.rnn_outputs, [-1, self.hidden_dim * 2])
#            W_a1 = tf.get_variable("W_a1", shape=[self.hidden_dim * 2, self.attention_dim],
#                                   initializer=self.initializer, trainable=True)
#            u1 = tf.matmul(H1, W_a1)
#
#            H2 = tf.reshape(tf.identity(self.rnn_outputs), [-1, self.hidden_dim * 2])
#            W_a2 = tf.get_variable("W_a2", shape=[self.hidden_dim * 2, self.attention_dim],
#                                   initializer=self.initializer, trainable=True)
#            u2 = tf.matmul(H2, W_a2)
#
#            u1 = tf.reshape(u1, [self.batch_size, self.seq_len, self.hidden_dim * 2])
#            u2 = tf.reshape(u2, [self.batch_size, self.seq_len, self.hidden_dim * 2])
#            u = tf.matmul(u1, u2, transpose_b=True)

            # Array of weights for each time step
            # A = tf.nn.softmax(u, name="attention")
            # self.outputs = tf.matmul(A, tf.reshape(tf.identity(self.rnn_outputs), [-1, self.seq_len, self.hidden_dim * 2]))

        # linear
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.class_num], initializer=self.initializer, dtype=tf.float32)
        self.softmax_b = tf.get_variable("softmax_b", [self.class_num], initializer=self.initializer, dtype=tf.float32)

        self.logits = tf.matmul(tf.reshape(self.outputs, [-1, 2*self.hidden_dim]), self.softmax_w) + self.softmax_b
        self.logits = tf.reshape(self.logits, [-1, self.seq_len, self.class_num])

        if not self.is_crf:
            pass
            # softmax
            #softmax_out = tf.nn.softmax(self.logits, axis=-1)
            #self.batch_pred_sequence = tf.cast(tf.argmax(softmax_out, -1), tf.int32)
            #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            #mask = tf.sequence_mask(self.seq_len)
            #self.losses = tf.boolean_mask(losses, mask)
            #self.loss = tf.reduce_mean(losses)
        else:
            # crf
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.input_y, mask)
            self.outputs, self.batch_viterbi_score = tf.contrib.crf.crf_decode(self.logits, transition_params, mask)
            self.outputs_ = tf.identity(self.outputs, name='outputs')
            self.loss = tf.reduce_mean(-log_likelihood)

        # summary
        self.train_summary = tf.summary.scalar("loss", self.loss)
        self.dev_summary = tf.summary.scalar("loss", self.loss)

        # optimize
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

# lstm_cell = self.cell
# if self.is_training:
#     lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=(1 - self.dropout_rate))
# lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
#
# outputs, _ = tf.contrib.rnn.static_rnn(
#     lstm_cell,
#     inputs_emb,
#     dtype=tf.float32,
#     sequence_length=self.seq_len * self.args.batch_size
# )
# # outputs: list_steps[batch, 2*dim]
# outputs = tf.concat(outputs, 1)
# outputs = tf.reshape(outputs, [self.batch_size, self.max_time_steps, self.hidden_dim * 2])
