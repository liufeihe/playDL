# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


def rnn_with_tf():
    # just demonstrate how to use rnn with tensorflow
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
    state = lstm.zero_state(batch_size, tf.float32)
    loss = 0.0
    for i in range(num_steps):
        if i>0:
            tf.get_variable_scope().reuse_variables()
        lstm_output, state = lstm(current_input, state)
        final_output = fully_connected(lstm_output)
        loss += calc_loss(final_output, expected_output)


def mrnn_with_tf():
    # just demonstrate how to use rnn with tensorflow
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)
    # dropout will occur between layers , not in the layer
    dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=0.5)
    # stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm]*number_of_layers)
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([dropout_lstm] * number_of_layers)
    state = stacked_lstm.zero_state(batch_size, tf.float32)
    loss = 0.0
    for i in range(num_steps):
        if i > 0:
            tf.get_variable_scope().reuse_variables()
        stacked_lstm_output, state = lstm(current_input, state)
        final_output = fully_connected(stacked_lstm_output)
        loss += calc_loss(final_output, expected_output)


def rnn_simple_simulate():
    X = [1, 2]
    state = [0.0, 0.0]

    w_cell_state = np.asarray([[0.1,0.2],[0.3,0.4]])
    w_cell_input = np.asarray([0.5,0.6])
    b_cell = np.asarray([0.1,-0.1])

    w_output = np.asarray([[1.0],[2.0]])
    b_output = 0.1
    for i in range(len(X)):
        before_activation = np.dot(state, w_cell_state)+X[i]*w_cell_input+b_cell
        state = np.tanh(before_activation)

        final_output = np.dot(state, w_output)+b_output

        print 'before activation: ', before_activation
        print 'state: ', state
        print 'output: ', final_output


def main():
    rnn_simple_simulate()


if __name__ == '__main__':
    main()

