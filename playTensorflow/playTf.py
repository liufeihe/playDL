# -*- coding: utf-8 -*-

import tensorflow as tf


def vars():
    n_features = 120
    n_labels = 5
    weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print weights
        print sess.run(weights)


def hello():
    hello_constant = tf.constant('hello world')

    with tf.Session() as sess:
        output = sess.run(hello_constant)
        print output


def play():
    # hello()
    vars()


if __name__ == '__main__':
    play()