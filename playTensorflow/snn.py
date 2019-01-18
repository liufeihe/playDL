# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from numpy.random import RandomState

batch_size = 8


def predict_sales_num_with_shallow_nn():
    x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

    w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))
    y = tf.matmul(x, w1)

    loss_more = 1
    loss_less = 10
    # loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*loss_more, (y_-y)*loss_less))
    loss = tf.reduce_sum(tf.square(y-y_))
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

    rdm = RandomState(1)
    dataset_size =128
    X = rdm.rand(dataset_size, 2)
    Y = [[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        steps = 5000
        for i in range(steps):
            start = (i*batch_size)%dataset_size
            end = min(start+batch_size, dataset_size)
            sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
        print sess.run(w1)


def classify_2_with_shallow_nn():
    w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

    x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    rdm = RandomState(1)
    dataset_size = 128
    X = rdm.rand(dataset_size, 2)
    Y = [[int(x1+x2)<1] for (x1, x2) in X]

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        print sess.run(w1)
        print sess.run(w2)

        steps = 100000
        for i in range(steps):
            start = (i+batch_size)%dataset_size
            end = min(start+batch_size, dataset_size)

            sess.run(train_step, feed_dict={x:X[start:end],y_:Y[start:end]})
            if i%1000 == 0:
                total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
                print "After %d training steps, cross entropy on all data is %g"%(i, total_cross_entropy)

        print sess.run(w1)
        print sess.run(w2)


def main():
    predict_sales_num_with_shallow_nn()
    # classify_2_with_shallow_nn()


if __name__ == '__main__':
    main()
