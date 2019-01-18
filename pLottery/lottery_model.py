# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.platform import gfile
import numpy as np
import os


# 特殊标记，用来填充标记对话
PAD = "__PAD__"
GO = "__GO__"
EOS = "__EOS__"  # 对话结束
UNK = "__UNK__"  # 标记未出现在词汇表中的字符
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


class Config(object):
    epochs = 3  # 10
    batch_size = 64
    rnn_size = 256
    rnn_layers = 3
    attention_num_units = 10
    encoding_embedding_size = 15
    decoding_embedding_size = 15
    learning_rate = 0.01

    display_step = 10
    max_train_data_size = 10000

    log_dir = './logs/'
    checkpointDir = './lotteryModel/'
    dataDir = './data/'
    # checkpoint = './lotteryModel/lottery_model.ckpt'


class LotteryModel(object):
    def __init__(self, mode, l_type, config):
        self.mode = mode
        self.l_type = l_type
        self.config = config
        # load data
        source_int_to_letter, source_letter_to_int, \
        target_int_to_letter, target_letter_to_int = self.load_data_vocab()
        self.source_int_to_letter = source_int_to_letter
        self.source_letter_to_int = source_letter_to_int
        self.target_int_to_letter = target_int_to_letter
        self.target_letter_to_int = target_letter_to_int
        # build graph
        self.build_graph()

    def train(self, sess, saver):
        config = self.config
        batch_size = config.batch_size
        self.handle_data()
        train_source, train_target = self.read_data('train')
        valid_source, valid_target = self.read_data('valid')

        (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = \
            next(self.get_batches(valid_target, valid_source, batch_size,
                                  self.source_letter_to_int[PAD],
                                  self.target_letter_to_int[PAD]))
        display_step = config.display_step
        checkpoint = os.path.join(os.path.dirname(__file__), config.checkpoint)
        log_dir = os.path.join(os.path.dirname(__file__), config.log_dir)
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        merged = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer())
        for epoch_i in range(1, config.epochs + 1):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    self.get_batches(train_target, train_source, batch_size,
                                     self.source_letter_to_int[PAD],
                                     self.target_letter_to_int[PAD])
            ):
                summary, _, loss = sess.run([merged, self.train_op, self.loss], {
                    self.inputs: sources_batch,
                    self.targets: targets_batch,
                    self.learning_rate: config.learning_rate,
                    self.target_sequence_length: targets_lengths,
                    self.source_sequence_length: sources_lengths
                })
                if batch_i % display_step == 0:
                    writer.add_summary(summary, batch_i)
                    validation_loss = sess.run([self.loss], {
                        self.inputs: valid_sources_batch,
                        self.targets: valid_targets_batch,
                        self.learning_rate: config.learning_rate,
                        self.target_sequence_length: valid_targets_lengths,
                        self.source_sequence_length: valid_sources_lengths
                    })
                    print 'After %d steps, perplexity is %.3f, valid perplexity is %.3f' % (
                    batch_i, np.exp(loss), np.exp(validation_loss))
                    print 'Epoch {:>3}/{} Batch {:>4}/{} - Trainging loss: {:>6.3f} - Validation loss: {:>6.3f}' \
                        .format(epoch_i, self.config.epochs, batch_i, len(train_source) // batch_size, loss,
                                validation_loss[0])
            saver.save(sess, checkpoint, global_step=epoch_i)
            print 'model trained and saved'

        writer.close()

    def predict(self):
        pass

    def get_batches(self):
        pass

    def data_to_vector(self):
        print 'data to vector......'
        src_path = self.config.dataDir+self.l_type
        vocab_path = self.config.dataDir+self.l_type+'_vocab'
        dest_path = self.config.dataDir+self.l_type+'_ids'

        tmp_vocab = []
        # 读取字典文件的数据，生成一个dict，也就是键值对的字典
        with open(vocab_path, 'r') as f:
            tmp_vocab.extend(f.readlines())
        tmp_vocab = [line.strip() for line in tmp_vocab]
        # 将vocabulary_file中的键值对互换，因为在字典文件里是按照{123：好}这种格式存储的，我们需要换成{好：123}格式
        vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])

        output_f = open(dest_path, 'w')
        with open(src_path, 'r') as f:
            for line in f:
                line_vec = []
                for words in line.split():
                    line_vec.append(vocab.get(words, UNK_ID))
                # 将input_file里的中文字符通过查字典的方式，替换成对应的key，并保存在output_file
                output_f.write(" ".join([str(num) for num in line_vec]) + '\n')
        output_f.close()

    def read_data(self, data_type='train'):

        pass

    def handle_data(self):
        self.data_to_vector()

    def build_graph(self):
        with tf.variable_scope('input'):
            self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
            self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            self.target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
            self.max_target_sequence_length = tf.reduce_max(self.target_sequence_length, name='max_target_length')
            self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

        with tf.variable_scope('encoder'):
            encoder_embed_input = tf.contrib.layers.embed_sequence(self.inputs,
                                                                   len(self.source_letter_to_int),
                                                                   self.config.encoding_embedding_size)
            encoder_cell = tf.contrib.rnn.MultiRNNCell(
                [self.get_lstm_cell(self.config.rnn_size) for _ in range(self.config.rnn_layers)])
            encoder_output, encoder_state = tf.nn.dynamic_rnn(encoder_cell,
                                                              encoder_embed_input,
                                                              sequence_length=self.source_sequence_length,
                                                              dtype=tf.float32)

        with tf.variable_scope('decoder'):
            # 1. embedding
            decoder_input = self.process_decoder_input(self.targets,
                                                       self.target_letter_to_int,
                                                       self.config.batch_size)
            target_vocab_size = len(self.target_letter_to_int)
            decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size,
                                                                self.config.decoding_embedding_size]))
            decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)
            # decoder_embed_input = tf.contrib.layers.embed_sequence(decoder_input, target_vocab_size, self.config.decoding_embedding_size)

            # 2. construct the rnn
            num_units = self.config.rnn_size
            attention_states = encoder_output  # tf.transpose(encoder_output, [1, 0, 2])
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units,
                                                                    attention_states,
                                                                    memory_sequence_length=self.source_sequence_length)
            # cells = []
            # for i in range(self.config.rnn_layers):
            #     cell = self.get_lstm_cell(self.config.rnn_size)
            #     cell = tf.contrib.seq2seq.AttentionWrapper(cell,
            #                                                attention_mechanism,
            #                                                attention_layer_size=num_units)
            #     cells.append(cell)
            # decoder_cell = tf.contrib.rnn.MultiRNNCell(cells)

            decoder_cell = tf.contrib.rnn.MultiRNNCell(
                [self.get_lstm_cell(self.config.rnn_size) for _ in range(self.config.rnn_layers)])
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                               attention_mechanism,
                                                               attention_layer_size=num_units)
            attention_zero = decoder_cell.zero_state(self.config.batch_size, dtype=tf.float32)
            initial_state = attention_zero.clone(cell_state=encoder_state)

            # 3. output fully connected
            output_layer = Dense(target_vocab_size,
                                 kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            if self.mode == 'train':
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                                    sequence_length=self.target_sequence_length,
                                                                    time_major=False)
                training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, training_helper, initial_state,
                                                                   output_layer)
                decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                         impute_finished=True,
                                                                         maximum_iterations=self.max_target_sequence_length)
            else:
                start_tokens = tf.tile(tf.constant([self.target_letter_to_int[GO]], dtype=tf.int32),
                                       [self.config.batch_size],
                                       name='start_tokens')
                predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                             start_tokens,
                                                                             self.target_letter_to_int[EOS])
                predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                                     predicting_helper,
                                                                     initial_state,
                                                                     output_layer)
                decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                         impute_finished=True,
                                                                         maximum_iterations=self.max_target_sequence_length)

        with tf.variable_scope('loss'):
            training_logits = tf.identity(decoder_output.rnn_output, 'logits')
            predicting_logits = tf.identity(decoder_output.sample_id, name='predictions')
            masks = tf.sequence_mask(self.target_sequence_length, self.max_target_sequence_length, dtype=tf.float32,
                                     name='masks')
            self.loss = tf.contrib.seq2seq.sequence_loss(training_logits, self.targets, masks)
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope('optimize'):
            # optimizer = tf.train.AdamOptimizer(lr)
            # gradients = optimizer.compute_gradients(cost)
            # capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            # train_op = optimizer.apply_gradients(capped_gradients)
            training_variables = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, training_variables), 5)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, training_variables), name='train_op')

    def get_lstm_cell(self, rnn_size):
        return tf.contrib.rnn.LSTMCell(rnn_size,
                                       initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))

    def process_decoder_input(self, data, vocab_to_int, batch_size):
        print vocab_to_int[GO]
        ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int[GO]), ending], 1)
        return decoder_input

    def load_data_vocab(self):
        # 构造映射表
        source_int_to_letter = {}
        source_letter_to_int = {}
        target_int_to_letter = {}
        target_letter_to_int = {}

        if self.l_type == 'ssq':
            source_int_to_letter, source_letter_to_int = self.initialize_vocabulary()
            target_int_to_letter, target_letter_to_int = self.initialize_vocabulary()

        return source_int_to_letter, source_letter_to_int, target_int_to_letter, target_letter_to_int

    def initialize_vocabulary(self):
        vocabulary_path = self.config.dataDir + self.l_type+'_vocab'
        # 初始化字典，这里的操作与上面的48行的的作用是一样的，是对调字典中的key-value
        if gfile.Exists(vocabulary_path):
            rev_vocab = []
            with open(vocabulary_path, 'r') as f:
                rev_vocab.extend(f.readlines())
            rev_vocab = [line.strip().decode('utf-8') for line in rev_vocab]  # int -> letter
            vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
            return rev_vocab, vocab
        else:
            raise ValueError('Vocabulary file %s is not found' % vocabulary_path)


def main():
    config = Config()

    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('mode', '', 'model mode')
    tf.app.flags.DEFINE_string('type', 'ssq', 'lottery type, default ssq')
    l_type = FLAGS.type
    mode = FLAGS.mode
    model = LotteryModel(mode, l_type, config)

    saver = tf.train.Saver(max_to_keep=4)
    with tf.Session() as sess:
        if mode == 'train':
            model.train(sess, saver)
        else:
            model.predict(sess, saver)


if __name__ == '__main__':
    main()