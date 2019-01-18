# -*- coding:utf-8 -*-

import time
import input_data
import tensorflow as tf
import mnist_avg_inference
import mnist_avg_train

EVAL_INTERNAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_avg_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_avg_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        y = mnist_avg_inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variables_averages = tf.train.ExponentialMovingAverage(mnist_avg_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variables_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_avg_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print 'after %s training steps, validation accuracy = %g'%(global_step, accuracy_score)
                else:
                    print 'No checkpoint file found'
                    return
                time.sleep(EVAL_INTERNAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()

