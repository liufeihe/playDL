# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import math


class Config(object):
    input_size = 33
    output_size = 33
    hidden_size = 100
    batch_size = 32
    learning_rate = 0.01
    epochs = 100
    data_path = './data/ssq'


class LotteryNN(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_train = self.build_datasets()
        self.build_graph()

    def train(self):
        saver = tf.train.Saver()
        config = self.cfg
        with tf.Session() as sess:
            iteration = 1
            loss = 0
            valid_x, valid_y = next(self.get_batches('valid'))
            sess.run(tf.global_variables_initializer())
            epochs = config.epochs
            for e in range(1, epochs + 1):
                batches = self.get_batches()
                start = time.time()
                for x, y in batches:
                    train_loss, y_,  _ = sess.run([self.loss, self.y, self.optimizer],
                                                  feed_dict={self.inputs: x, self.targets: y})
                    loss += train_loss
                    if iteration % 100 == 0:
                        end = time.time()
                        validation_loss = sess.run([self.loss],
                                                   feed_dict={self.inputs: valid_x, self.targets: valid_y})
                        print "Epoch {}/{}, " \
                              "Iteration: {}, " \
                              "Avg. Training loss: {:.4f}, " \
                              "Valid loss: {:.4f}, " \
                              "{:.4f} sec/batch".format(e, epochs, iteration, loss / 100, validation_loss[0], (end - start)/100)
                        loss = 0
                        start = time.time()

                    iteration += 1

                    # if e == epochs:
                    #     print y_[0]
                    #     print len(y_)
                    #     print len(y_[0])
                    #     print y[0]

    def build_graph(self):
        cfg = self.cfg
        input_size = cfg.input_size
        output_size = cfg.output_size
        hidden_size = cfg.hidden_size
        batch_size = cfg.batch_size

        with tf.variable_scope('input'):
            self.inputs = tf.placeholder(tf.float32, [batch_size, input_size], name='inputs')
            self.targets = tf.placeholder(tf.float32, [batch_size, output_size], name='targets')
        with tf.variable_scope('output'):
            # w = tf.Variable(tf.zeros([input_size, output_size]))
            w = tf.Variable(tf.random_uniform((input_size, output_size),-1,1))
            b = tf.Variable(tf.zeros([output_size]))
            y = tf.sigmoid(tf.matmul(self.inputs, w) + b)
            self.y = y
        with tf.variable_scope('loss'):
            cross_entropy = -tf.reduce_sum(self.targets * tf.log(y))
            self.loss = cross_entropy
            optimizer = tf.train.GradientDescentOptimizer(cfg.learning_rate).minimize(self.loss)
            self.optimizer = optimizer

    def get_batches(self, data_type='train'):
        data = self.data_train
        data_len = len(data)
        valid_len = int(0.2*data_len)
        if data_type == 'valid':
            data2 = data[0: valid_len]
        else:
            data2 = data[valid_len:]

        batch_size = self.cfg.batch_size
        n_batches = (len(data2)-1)//batch_size
        data_x = data2[1:n_batches*batch_size+1]
        data_y = data2[:n_batches*batch_size]
        for idx in range(0, len(data_x), batch_size):
            batch_x = data_x[idx:idx+batch_size]
            batch_y = data_y[idx:idx+batch_size]
            yield batch_x, batch_y
            # print batch_x
            # print len(batch_x), len(batch_x[0])
            # print batch_y
            # print len(batch_y), len(batch_y[0])
            # break

    def build_datasets(self):
        input_size = self.cfg.input_size
        data = self.load_data()
        data_in_day = [day for day in data.split('\n')]
        data_day_prefix = [day.split(':')[1] for day in data_in_day if day]
        data_day_prefix = [prefix.split(',')[:-1] for prefix in data_day_prefix if prefix]
        data_train = [[1 if str(idx+1) in prefix else 0 for idx in range(input_size)] for prefix in data_day_prefix]
        print 'building data set ok.'
        # print data_train[0], len(data_train[0])
        return data_train

    def load_data(self):
        with open(self.cfg.data_path, 'r') as f:
            data = f.read()
        return data


def main():
    cfg = Config()
    lotteryNN = LotteryNN(cfg)
    lotteryNN.train()


if __name__ == '__main__':
    main()
