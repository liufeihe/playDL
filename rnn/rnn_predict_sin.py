# -*- coding: utf-8 -*-
from sklearn import cross_validation
from sklearn import datasets
from sklearn import metrics
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

learn = tf.contrib.learn

HIDDEN_SIZE = 30
NUM_LAYERS = 2
TIMESTEPS = 10
TRAINING_STEPS = 10000
BATCH_SIZE =32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01


def generate_data(seq):
    x = []
    y = []
    for i in range(len(seq)-TIMESTEPS-1):
        x.append([seq[i:i+TIMESTEPS]])
        y.append([seq[i+TIMESTEPS]])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(X, y):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*NUM_LAYERS)
    x_ = tf.unstack(X, axis=1)
    output, _ =  tf.nn.rnn(cell, x_, dtype=tf.float32)
    output = output[-1]
    prediction, loss = learn.models.linear_regression(output, y)
    train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
                                               optimizer='Adagrad', learning_rate=0.1)
    return prediction, loss, train_op


def sin_predict():
    regressor = learn.Estimator(model_fn=lstm_model)
    test_start = TRAINING_EXAMPLES * SAMPLE_GAP
    test_end = (TRAINING_EXAMPLES+TESTING_EXAMPLES) * SAMPLE_GAP
    train_x, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
    test_x, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))
    regressor.fit(train_x, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
    predicted = [[pred] for pred in regressor.predict(test_x)]
    rmse = np.sqrt(((predicted-test_y)**2).mean(axis=0))
    print 'mean square error is: %f', rmse[0]

    fig = plt.figure()
    plot_predicted = plt.plot(predicted, label='predicted')
    plot_test = plt.plot(test_y, label='real_sin')
    plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])
    fig.savefig('sin.png')




def my_model(features, target):
    # 将预测的目标转换为one-hot形式
    target = tf.one_hot(target, 3, 1, 0)
    logits, loss = learn.models.logistic_regression(features, target)
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer = 'Adagrad',
        learning_rate = 0.1
    )
    return tf.argmax(logits, 1), loss, train_op


def siri_classify():
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=0)
    classifier = learn.Estimator(model_fn = my_model)
    classifier.fit(x_train, y_train, steps=100)
    y_predicted = classifier.predict(x_test)
    score = metrics.accuracy_score(y_test, y_predicted)
    print 'Accuracy: %.2f%', (score*100)


def main():
    # siri_classify()
    sin_predict()


if __name__ == '__main__':
    main()
