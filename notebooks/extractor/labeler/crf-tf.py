import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
Timestep = 15#输入的总长度，可以理解为15个rnn cell
Batchsize = 1#一次就输入一个
Inputsize = 1
LR = 0.5
num_tags = 2
#定义batch输出
def get_batch():
    xs = np.array([[2, 3, 4, 5, 5, 5, 1, 5, 3, 2, 5, 5, 5, 3, 5]])
    res = np.array([[0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]])
    return [xs[:, :, np.newaxis], res]
# xs, res = get_batch()
# print(xs)
# xs变成三维的 res还是二维的
class crf:
    def __init__(self, time_steps, input_size, num_tags, batch_size):
        self.time_steps = time_steps
        self.input_size = input_size
        self.num_tags = num_tags
        self.batch_size = batch_size
        self.xs = tf.placeholder(tf.float32, [None, self.time_steps, self.input_size], name='xs')
        self.res = tf.placeholder(tf.int32, [self.batch_size, self.time_steps], name='res')#为什么和xs的定义模式不一样
        weights = tf.get_variable('weights', [self.input_size, self.num_tags])
        matricized_xs = tf.reshape(self.xs, [-1, self.input_size])
        matricized_unary_scores = tf.matmul(matricized_xs, weights)
        unary_scores = tf.reshape(matricized_unary_scores, [self.batch_size, self.time_steps, self.num_tags])
        sequence_len = np.full(self.batch_size, self.time_steps, dtype=np.int32)
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores, self.res, sequence_len)
        self.pred, viterbiscore = tf.contrib.crf.crf_decode(unary_scores, transition_params, sequence_len)
        self.loss = tf.reduce_mean(-log_likelihood)
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.loss)


if __name__ == '__main__':
    model = crf(Timestep, Inputsize, num_tags, Batchsize)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    plt.ion()#动态曲线
    plt.show()
    for i in range(150):
        xs, res = get_batch()
        feed_dict = {model.xs: xs,
                     model.res: res}
        _, cost, pred = sess.run([model.train_op, model.loss, model.pred],
                                 feed_dict=feed_dict)#只有placeholder才可以feed
        x = xs.reshape(-1, 1)
        r = res.reshape(-1, 1)
        p = pred.reshape(-1, 1)
        x = range(len(x))
        plt.clf()
        plt.plot(x, r, 'r', x, p, 'g')
        plt.ylim(-1.2, 1.2)
        plt.draw()
        plt.pause(0.3)
        if i % 20 == 0:
            print('cost:', round(cost, 4))