# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import reader
import os

PTB_PATH = '/home/odark/Ai/codes/playWithTf/rnn/ptb/'
DATA_PATH = PTB_PATH+'/PTB_data/'
VOCAB_SIZE = 10000

NUM_LAYERS = 2
HIDDEN_SIZE = 200
LEARNING_RATE = 1.0
KEEP_PROB = 0.5
MAX_GRAD_NORM = 5

NUM_EPOCH = 2
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35
EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1

MODEL_SAVE_DIR = PTB_PATH+'/model/'
MODEL_SAVE_NAME = 'ptb'


def get_cosine(v1, v2):
    a1 = tf.reduce_sum(tf.multiply(v1, v2))
    b1 = tf.sqrt(tf.reduce_sum(tf.multiply(v1, v1)))
    b2 = tf.sqrt(tf.reduce_sum(tf.multiply(v2, v2)))    # element-wise
    return a1/(b1*b2)


def get_max_cosine(vocab, word):
    cosines = []
    for i in range(VOCAB_SIZE):
        cosines.append(get_cosine(vocab[i].reshape(1, HIDDEN_SIZE), word))
    return tf.argmax(cosines)


def get_max_cosine2(vocab, word):
    a1 = tf.matmul(vocab, word.reshape(HIDDEN_SIZE, 1))  # (vocab_size,1)
    b1 = tf.sqrt(tf.reduce_sum(tf.multiply(vocab, vocab), 1, keep_dims=True))  # (vocab_size, 1)
    b2 = tf.sqrt(tf.reduce_sum(tf.multiply(word, word)))
    cosines = tf.div(a1, tf.multiply(b1, b2))
    return tf.argmax(cosines)


class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*NUM_LAYERS)

        self.initial_state = cell.zero_state(batch_size, tf.float32)
        embedding = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE])
        self.embedding = embedding
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        # outputs = []
        # state = self.initial_state
        # with tf.variable_scope('RNN'):
        #     for time_step in range(num_steps):
        #         if time_step > 0:
        #             tf.get_variable_scope().reuse_variables()
        #         cell_output, state = cell(inputs[:, time_step, :], state)
        #         outputs.append(cell_output)
        # output = tf.concat(outputs, 0)  # output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])


        lstm_state_as_tensor_shape = [NUM_LAYERS, num_steps, batch_size, HIDDEN_SIZE]
        initial_state = tf.zeros(lstm_state_as_tensor_shape)
        unstack_state = tf.unstack(initial_state, axis=0)
        tuple_state = tuple([tf.contrib.rnn.LSTMStateTuple(unstack_state[idx][0], unstack_state[idx][1]) for idx in range(NUM_LAYERS)])
        # [batch_size, num_steps, hidden_size] -> [num_steps, batch_size, hidden_size]
        inputs = tf.unstack(inputs, num=num_steps, axis=1)
        outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=tuple_state, scope='RNN')
        # [num_steps, batch_size, hidden_size] -> [num_steps * batch_size, hidden_size]
        output = tf.concat(outputs, 0)

        self.output = output
        weight = tf.get_variable('weight', [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable('bias', [VOCAB_SIZE])
        logits = tf.matmul(output, weight)+bias

        logits = tf.reshape(logits, [self.batch_size, self.num_steps, VOCAB_SIZE])
        loss = tf.contrib.seq2seq.sequence_loss(logits,
                                                self.targets,
                                                tf.ones([batch_size, num_steps], dtype=tf.float32))
        self.cost = loss
        # loss = tf.contrib.seq2seq.sequence_loss(logits,
        #                                         self.targets,
        #                                         tf.ones([batch_size, num_steps],dtype=tf.float32),
        #                                         average_across_timesteps=False,
        #                                         average_across_batch=False)
        # self.cost = tf.reduce_sum(loss)/batch_size
        # loss = tf.contrib.seq2seq.sequence_loss_by_example([logits],
        #                                               [tf.reshape(self.targets, [-1])],
        #                                               [tf.ones([batch_size*num_steps],dtype=tf.float32)])
        # self.cost = tf.reduce_sum(loss)/batch_size
        self.final_state = state
        # self.final_state = tf.divide(state, 1.0, 'm_final_state')

        tf.add_to_collection('m_cost', self.cost)
        # tf.add_to_collection('m_final_state', state)
        tf.add_to_collection('m_embedding', embedding)
        tf.add_to_collection('m_output', output)
        tf.add_to_collection('m_input_data', self.input_data)
        tf.add_to_collection('m_targets', self.targets)
        # tf.add_to_collection('m_initial_state', self.initial_state)

        if not is_training:
            return
        training_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, training_variables), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        self.train_op = optimizer.apply_gradients(zip(grads, training_variables), name='train_op')
        tf.add_to_collection('m_train_op', self.train_op)


def run_epoch(session, model, data, train_op, output_log):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps)):
        cost, state, _, embedding, output = session.run([model.cost, model.final_state,
                                                         train_op, model.embedding, model.output],
                                     {model.input_data: x,
                                      model.targets: y,
                                      model.initial_state: state})
        total_costs += cost
        iters += model.num_steps

        if output_log and step % 100 == 0:
            print 'After %d steps, perplexity is %.3f' % (step, np.exp(total_costs / iters))
        if not output_log and step % 5000 == 0:
            # index = session.run(get_max_cosine(embedding, output))
            print output
            index = session.run(get_max_cosine2(embedding, output))
            index = index[0]
            print 'step:%d, x: %d, predicted output index: %d' % (step, x[0][0], index)

    return np.exp(total_costs/iters)


def train_and_inference():
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    with tf.variable_scope('language_model', reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
    with tf.variable_scope('language_model', reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        for i in range(NUM_EPOCH):
            print 'In iteration: %d' % (i + 1)
            run_epoch(session, train_model, train_data, train_model.train_op, True)
            valid_perplexity = run_epoch(session, eval_model, valid_data, tf.no_op(), False)
            print 'Epoch: %d Validation Perplexity: %.3f' % (i + 1, valid_perplexity)

        test_perplexity = run_epoch(session, eval_model, test_data, tf.no_op(), False)
        print 'Test Perplexity: %.3f' % test_perplexity


def train_model():
    print 'training ...'
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    with tf.variable_scope('language_model', reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    saver = tf.train.Saver(max_to_keep=4)
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print 'In iteration: %d' % (i + 1)

            total_costs = 0.0
            iters = 0
            state = session.run(train_model.initial_state)
            for step, (x, y) in enumerate(reader.ptb_iterator(train_data,
                                                              train_model.batch_size,
                                                              train_model.num_steps)):
                cost, state, _, embedding, output = session.run([train_model.cost,
                                                                 train_model.final_state,
                                                                 train_model.train_op,
                                                                 train_model.embedding,
                                                                 train_model.output],
                                                                {train_model.input_data: x,
                                                                 train_model.targets: y,
                                                                 train_model.initial_state: state})
                # print train_model.initial_state
                # print 'final:'
                # print train_model.final_state
                # print train_model.train_op
                total_costs += cost
                iters += train_model.num_steps
                if step % 100 == 0:
                    print 'After %d steps, perplexity is %.3f' % (step, np.exp(cost))  # total_costs/iters
                    saver.save(session, os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME), global_step=step)


def train_model_from_save():
    print 'loading and inference ...'
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
    data = train_data
    saver = tf.train.import_meta_graph(os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME+'-1300.meta'))
    with tf.Session() as session:
        saver.restore(session, tf.train.latest_checkpoint(MODEL_SAVE_DIR))
        graph = tf.get_default_graph()

        m_cost = tf.get_collection('m_cost')[0]
        m_final_state = graph.get_tensor_by_name('language_model/RNN/RNN/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_137:0')
        m_train_op = tf.get_collection('m_train_op')[0]
        m_embedding = tf.get_collection('m_embedding')[0]
        m_output = tf.get_collection('m_output')[0]
        m_input_data = tf.get_collection('m_input_data')[0]
        m_targets = tf.get_collection('m_targets')[0]
        m_initial_state = graph.get_tensor_by_name('language_model/MultiRNNCellZeroState/DropoutWrapperZeroState/BasicLSTMCellZeroState/zeros:0')

        for i in range(NUM_EPOCH):
            print 'In iteration: %d' % (i + 1)
            total_costs = 0.0
            iters = 0
            state = session.run(m_initial_state)
            for step, (x, y) in enumerate(reader.ptb_iterator(data,
                                                              TRAIN_BATCH_SIZE,
                                                              TRAIN_NUM_STEP)):
                cost, state, _, embedding, output = session.run([m_cost, m_final_state, m_train_op, m_embedding, m_output],
                                                                {m_input_data: x,
                                                                 m_targets: y,
                                                                 m_initial_state: state})
                total_costs += cost
                iters += TRAIN_NUM_STEP
                if step % 100 == 0:
                    print 'After %d steps, perplexity is %.3f' % (step, np.exp(total_costs / iters))

        # print cost
        # print final_state
        # print train_op
        # print embedding
        # print output
        # print input_data
        # print targets
        # print initial_state
        # print tf.trainable_variables()
        # print 'variables:'
        # var_str = tf.global_variables()
        # print 'operations:'
        # ops_str = graph.get_operations()
        # with open(os.path.join(PTB_PATH, 'var_ops.txt'), 'w+') as f:
        #     f.write('vars:\n')
        #     f.write(str(var_str))
        #     f.write('\n')
        #     f.write('\n')
        #     f.write('ops:\n')
        #     f.write(str(ops_str))


def handle_data():
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
    print len(train_data)
    print train_data[:100]

    x, y = reader.ptb_producer(train_data, 4, 5)
    print "X: ",x
    print "Y: ",y


def main(_):
    # handle_data()
    # train_and_inference()
    train_model()
    # train_model_from_save()

if __name__ == '__main__':
    tf.app.run()
