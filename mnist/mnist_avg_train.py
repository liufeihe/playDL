# -*- coding:utf-8 -*-

import os
import input_data
import tensorflow as tf
import numpy as np
import mnist_avg_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'model.ckpt'

USE_CNN = True


def train(mnist):
    if USE_CNN:
        x = tf.placeholder(tf.float32, [BATCH_SIZE,
                                        mnist_avg_inference.IMAGE_SIZE,
                                        mnist_avg_inference.IMAGE_SIZE,
                                        mnist_avg_inference.NUM_CHANNELS], name='x-input')
    else:
        x = tf.placeholder(tf.float32, [None, mnist_avg_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_avg_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    if USE_CNN:
        y = mnist_avg_inference.inference_with_cnn(x, True, regularizer)
    else:
        y = mnist_avg_inference.inference(x, regularizer)

    global_step = tf.Variable(0, trainable=False)
    variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variables_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            if USE_CNN:
                xs = np.reshape(xs, (BATCH_SIZE,
                                     mnist_avg_inference.IMAGE_SIZE,
                                     mnist_avg_inference.IMAGE_SIZE,
                                     mnist_avg_inference.NUM_CHANNELS))
            _, loss_value, step, acc= sess.run([train_op, loss, global_step, accuracy], feed_dict={x: xs, y_: ys})
            if i%1000 == 0:
                print 'after %d training steps, loss on training batch is %g, ' \
                      'accuracy on the training batch is %g.'%(i, loss_value, acc)
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()