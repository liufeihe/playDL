# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


def simulate_linear_func():
    #x_data = np.random.rand(100).astype("float32")
    x_data = tf.placeholder(tf.float32)
    y_data = x_data*0.1 + 0.3
    tf.constant()

    #define variables to find W and b
    W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b

    #minize the MSE
    loss = tf.reduce_mean(tf.square(y-y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    #before starting, init variables first
    init = tf.global_variables_initializer()

    #launch the graph
    sess = tf.Session()
    sess.run(init)

    for step in xrange(2001):
        sess.run(train, feed_dict={x_data: np.random.rand(100).astype("float32")})
        if step%20 == 0:
            print step,sess.run(W), sess.run(b)
    sess.close()


def get_min_of_func():
    coefficients = np.array([[1],[-20],[25]])
    w = tf.Variable([0], dtype=tf.float32)
    x = tf.placeholder(tf.float32, [3,1])
    cost = x[0][0]*w**2+x[1][0]*w+x[2][0]
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        print session.run(w)

        for i in range(1000):
            session.run(train, feed_dict={x: coefficients})
        print session.run(w)


def test_two_graph():
    g1 = tf.Graph()
    with g1.as_default():
        v = tf.get_variable('v', initializer=tf.zeros_initializer(shape=[1]))
    a = tf.constant(1, name='a')
    b = tf.constant(2, name='b')
    with g1.device('/gpu:0'):
        result = a+b

    g2 = tf.Graph()
    with g2.as_default():
        v = tf.get_variable('v', initializer=tf.ones_initializer(shape=[1]))

    with tf.Session(graph=g1) as sess:
        tf.initialize_all_variables.run()
        with tf.variable_scope('', reuse=True):
            print sess.run(tf.get_variable('v'))
            print sess.run(result)

    with tf.Session(graph=g2) as sess:
        tf.initialize_all_variables.run()
        with tf.variable_scope('', reuse=True):
            print sess.run(tf.get_variable('v'))


def test_session():
    a = tf.constant([1], name='a')
    b = tf.constant([2], name='b')
    d = b/2
    d2 = d/3
    with tf.variable_scope('test_cc'):
        c = a + b
    c2 = a + d2
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run([d, c])
        # print c.eval(session=sess)
        graph = tf.get_default_graph()
        print graph.get_operations()

    # InteractiveSession will be set to the default session
    # sess = tf.InteractiveSession()
    # print c.eval()
    # sess.close()

    # sess = tf.Session()
    # with sess.as_default():
    #     print c.eval()
    # sess.close()
    #
    # sess = tf.Session()
    # print c.eval(session=sess)
    # print sess.run(c)
    # sess.close()

    # config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    # sess1 = tf.InteractiveSession(config=config)
    # sess2 = tf.Session(config=config)


def test_variable_reuse():
    with tf.variable_scope('foo'):
        # under namespace foo, create variable v
        v = tf.get_variable('v', [1], initializer=tf.constant_initializer(1.0))
    with tf.variable_scope('foo'):
        # v is already existed under foo, it's a error to create again
        # if u just want to get the v , u should set reuse true
        v2 = tf.get_variable('v', [1])
    with tf.variable_scope('foo', reuse=True):
        v3 = tf.get_variable('v', [1])
        print v == v3
    with tf.variable_scope('bar', reuse=True):
        # reuse is true, get_variable can only get the v ,but it's not existed under bar ,so it's a error
        v4 = tf.get_variable('v', [1])


def argmax_testx():
    print tf.argmax(range(50), 0)
    print 'hello world!'


def embedding_testx():
    embedding = tf.get_variable('embedding', [10, 7])
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print sess.run(embedding)


def tf_test():
    # a = [[1,2,3],[4,5,6]]
    # b = tf.reshape(a, [-1,6])
    # b = tf.ones([3*2])
    # b = tf.reduce_sum([3,2,1,2])

    a = [[[1,11],[2,22],[3,33],[7,77]],[[4,44],[5,55],[6,66],[8,88]],[[4,44],[5,55],[6,66],[9,99]]]
    # a = [[1],[2]]a
    # a = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
    b = tf.concat(a, 1)
    # c1 = [[1,2,3],[4,5,6]]
    # c2 = [[7,8,9],[10,11,12]]
    # c = tf.concat([c1,c2],1)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print np.array(a).shape
        print sess.run(b)
        print b.shape
        # print 'c:\n'
        # print sess.run(c)
        # print c.shape

vocab_size = 1000
word_vector_size = 10


def get_cosine(v1, v2):
    a1 = tf.reduce_sum(tf.multiply(v1, v2))
    b1 = tf.sqrt(tf.reduce_sum(tf.multiply(v1, v1)))
    b2 = tf.sqrt(tf.reduce_sum(tf.multiply(v2, v2)))
    return a1/(b1*b2)


def get_max_cosine(vocab, word):
    cosines = []
    for i in range(vocab_size):
        cosines.append(get_cosine(vocab[i].reshape(1, word_vector_size), word))
        # cosines.append(tf.losses.cosine_distance(vocab[i].reshape(1, word_vector_size), word, dim=0))
    return cosines, tf.argmax(cosines)


def get_max_cosine2(vocab, word):
    a1 = tf.matmul(vocab, word.reshape(word_vector_size, 1))  # (vocab_size,1)
    b1 = tf.sqrt(tf.reduce_sum(tf.multiply(vocab, vocab), 1, keep_dims=True))  # (vocab_size, 1)
    b2 = tf.sqrt(tf.reduce_sum(tf.multiply(word, word)))
    cosines = tf.div(a1, tf.multiply(b1, b2))
    return cosines, tf.argmax(cosines)


def cosine_max_test():
    vocab = tf.Variable(initial_value=tf.random_uniform([vocab_size, word_vector_size], 0, 1))
    # word = tf.Variable(initial_value=tf.random_uniform([1, word_vector_size], 0, 1))
    word = tf.ones(shape=(1, word_vector_size), dtype=tf.float32)
    # cos_dist = tf.losses.cosine_distance(vocab, word, dim=0)
    # cos_dist2 = tf.losses.cosine_distance(word, word, dim=0)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        v, w = sess.run([vocab, word])
        cosines, max_index = sess.run(get_max_cosine(v, w))
        cosines, max_index2 = sess.run(get_max_cosine2(v, w))
        # print cosines
        print max_index
        print max_index2
        # print w
        # print v
        # k = [i[1]/i[0] for i in v]
        # print k


def main():
    # argmax_testx()
    # test_session()
    # test_two_graph()
    embedding_testx()
    # tf_test()
    # cosine_max_test()

if __name__ == '__main__':
    #simulate_linear_func()
    #get_min_of_func()
    main()