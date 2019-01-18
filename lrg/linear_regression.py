# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib
import os
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt


MODEL_SAVE_DIR = '/home/odark/Ai/codes/playWithTf/lrg/model/'
MODEL_SAVE_NAME = 'lrg'


def generate_dataset():
    x_batch = np.linspace(-1, 1, 101)
    y_batch = 2 * x_batch + np.random.randn(*x_batch.shape)*0.3
    return x_batch, y_batch


def linear_regression():
    x = tf.placeholder(tf.float32, (None,), name='x')
    y = tf.placeholder(tf.float32, (None,), name='y')

    with tf.variable_scope('lreg') as scope:
        w = tf.Variable(np.random.normal(), name='W')
        y_pred = tf.multiply(w, x)
        loss = tf.reduce_mean(tf.square(y_pred-y))
    return x, y, y_pred, loss, w


def train_model():
    x_batch, y_batch = generate_dataset()
    x, y, y_pred, loss, w = linear_regression()
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=4)
    tf.add_to_collection('cw', w)
    with tf.Session() as session:
        session.run(init)

        feed_dict = {x:x_batch,y:y_batch}
        for step in range(30):
            loss_val, train_w,  _ = session.run([loss, w, optimizer], feed_dict=feed_dict)
            print "[%d] loss val:%f, w is :%f" %(step, loss_val.mean(), train_w)
            if step%5 == 0:
                saver.save(session, os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME), global_step=step)
        y_pred_batch = session.run(y_pred, feed_dict={x:x_batch})

        # writer = tf.summary.FileWriter('./log/', session.graph)

    # plt.figure(1)
    # plt.scatter(x_batch, y_batch)
    # plt.plot(x_batch, y_pred_batch)
    # plt.savefig("lreg_plot.png")
    # plt.show()


def inference():
    print 'loading and inference...'
    input_x = [2.0]
    # load meta first
    saver = tf.train.import_meta_graph(os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME+'-25.meta'))
    with tf.Session() as sess:
        # load variables
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_SAVE_DIR))
        graph = tf.get_default_graph()
        print tf.global_variables()
        print graph.get_operations()
        # print 'x:'
        # print sess.run([graph.get_operation_by_name('x'),], feed_dict={'x:0': input_x})
        cw = tf.get_collection('cw')
        print cw
        print sess.run(cw)
        print 'W:'
        print sess.run([graph.get_tensor_by_name('lreg/W:0')])
        print 'y_pred:'
        print sess.run(graph.get_tensor_by_name('lreg/Mul:0'), feed_dict={'x:0': input_x})
        # print sess.run([graph.get_tensor_by_name('y_pred:0')])


def main():
    # train_model()
    inference()


if __name__ == '__main__':
    main()