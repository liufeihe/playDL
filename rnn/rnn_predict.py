# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


LSTM_HIDDEN_SIZE = 10


def predict_even_num():
    lstm = tf.nn.rnn_cell.BasicLSTMCell(LSTM_HIDDEN_SIZE)


def main():
    predict_even_num()


if __name__ == '__main__':
    main()