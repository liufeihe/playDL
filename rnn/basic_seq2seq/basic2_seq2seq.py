# -*- coding:utf-8 -*_
# python basic2_seq2seq.py --mode train
# python basic2_seq2seq.py --mode predict

import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
import os


class Config(object):
    epochs = 60
    batch_size = 128
    rnn_size = 50
    rnn_layers = 2
    encoding_embedding_size = 15
    decoding_embedding_size = 15
    learning_rate = 0.001

    checkpoint = './model2/train_model.ckpt'
    display_step = 50


class LSTM(object):
    def __init__(self, mode, config):
        self.mode = mode
        self.config = config
        # load data
        source_int_to_letter, source_letter_to_int, \
        target_int_to_letter, target_letter_to_int, \
        source_int, target_int = self.load_data()
        self.source_int_to_letter = source_int_to_letter
        self.source_letter_to_int = source_letter_to_int
        self.target_int_to_letter = target_int_to_letter
        self.target_letter_to_int = target_letter_to_int
        self.source_int = source_int
        self.target_int = target_int
        # build graph
        self.build_graph()

    def train(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            batch_size = self.config.batch_size
            train_source = self.source_int[batch_size:]
            train_target = self.target_int[batch_size:]
            valid_source = self.source_int[:batch_size]
            valid_target = self.target_int[:batch_size]
            (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = \
                next(self.get_batches(valid_target, valid_source, batch_size,
                                      self.source_letter_to_int['<PAD>'],
                                      self.target_letter_to_int['<PAD>']))
            display_step = self.config.display_step
            checkpoint = self.config.checkpoint

            sess.run(tf.global_variables_initializer())
            for epoch_i in range(1, self.config.epochs + 1):
                for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                        self.get_batches(train_target, train_source, batch_size,
                                         self.source_letter_to_int['<PAD>'],
                                         self.target_letter_to_int['<PAD>'])
                ):
                    _, loss = sess.run([self.train_op, self.loss], {
                        self.inputs: sources_batch,
                        self.targets: targets_batch,
                        self.learning_rate: self.config.learning_rate,
                        self.target_sequence_length: targets_lengths,
                        self.source_sequence_length: sources_lengths
                    })
                    if batch_i % display_step == 0:
                        validation_loss = sess.run([self.loss], {
                            self.inputs: valid_sources_batch,
                            self.targets: valid_targets_batch,
                            self.learning_rate: self.config.learning_rate,
                            self.target_sequence_length: valid_targets_lengths,
                            self.source_sequence_length: valid_sources_lengths
                        })
                        print 'Epoch {:>3}/{} Batch {:>4}/{} - Trainging loss: {:>6.3f} - Validation loss: {:>6.3f}' \
                            .format(epoch_i, self.config.epochs, batch_i, len(train_source) // batch_size, loss, validation_loss[0])
                saver.save(sess, checkpoint)
                print 'model trained and saved'

    def predict(self):
        batch_size = self.config.batch_size
        checkpoint = self.config.checkpoint

        loader = tf.train.Saver()
        loaded_graph = tf.get_default_graph()
        # loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            # loader = tf.train.import_meta_graph(checkpoint + '.meta')
            print checkpoint
            loader.restore(sess, checkpoint)

            input_word = 'common'
            text = self.source_to_seq(input_word)
            input_data = loaded_graph.get_tensor_by_name('input/inputs:0')
            logits = loaded_graph.get_tensor_by_name('loss/predictions:0')
            source_sequence_length = loaded_graph.get_tensor_by_name('input/source_sequence_length:0')
            target_sequence_length = loaded_graph.get_tensor_by_name('input/target_sequence_length:0')
            answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                              target_sequence_length: [len(input_word)] * batch_size,
                                              source_sequence_length: [len(input_word)] * batch_size
                                              })[0]
            pad = self.source_letter_to_int['<PAD>']
            print 'input:' + input_word
            print '\n Source'
            print '   Word 编号： {}'.format([i for i in text])
            print '   Input Words: {}'.format(' '.join([self.source_int_to_letter[i] for i in text]))
            print '\n Target'
            print '   Word 编号： {}'.format([i for i in answer_logits if i != pad])
            print '   Response Words: {}'.format(' '.join([self.target_int_to_letter[i] for i in answer_logits if i != pad]))

    def source_to_seq(self, text):
        sequence_length = 7
        source_letter_to_int = self.source_letter_to_int
        return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text] + \
               [source_letter_to_int['<PAD>']] * (sequence_length - len(text))

    def get_batches(self, targets, sources, batch_size, source_pad_int, target_pad_int):
        for batch_i in range(0, len(sources) // batch_size):
            start_i = batch_i * batch_size
            sources_batch = sources[start_i: start_i + batch_size]
            targets_batch = targets[start_i: start_i + batch_size]
            pad_sources_batch = np.array(self.pad_sentence_batch(sources_batch, source_pad_int))
            pad_targets_batch = np.array(self.pad_sentence_batch(targets_batch, target_pad_int))
            targets_lengths = []
            for target in targets_batch:
                targets_lengths.append(len(target))
            source_lengths = []
            for source in targets_batch:
                source_lengths.append(len(source))

            yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths

    def pad_sentence_batch(self, sentence_batch, pad_int):
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

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
            encoder_cell = tf.contrib.rnn.MultiRNNCell([self.get_lstm_cell(self.config.rnn_size) for _ in range(self.config.rnn_layers)])
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
            decoder_cell = tf.contrib.rnn.MultiRNNCell([self.get_lstm_cell(self.config.rnn_size) for _ in range(self.config.rnn_layers)])
            # 3. output fully connected
            output_layer = Dense(target_vocab_size,
                                 kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            if self.mode == 'train':
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                                    sequence_length=self.target_sequence_length,
                                                                    time_major=False)
                training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, training_helper, encoder_state, output_layer)
                decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                                  impute_finished=True,
                                                                                  maximum_iterations=self.max_target_sequence_length)
            else:
                start_tokens = tf.tile(tf.constant([self.target_letter_to_int['<GO>']], dtype=tf.int32),
                                       [self.config.batch_size],
                                       name='start_tokens')
                predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                             start_tokens,
                                                                             self.target_letter_to_int['<EOS>'])
                predicting_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, predicting_helper,
                                                                     encoder_state, output_layer)
                decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                                    impute_finished=True,
                                                                                    maximum_iterations=self.max_target_sequence_length)

        with tf.variable_scope('loss'):
            training_logits = tf.identity(decoder_output.rnn_output, 'logits')
            predicting_logits = tf.identity(decoder_output.sample_id, name='predictions')  # used for predict
            masks = tf.sequence_mask(self.target_sequence_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')
            self.loss = tf.contrib.seq2seq.sequence_loss(training_logits, self.targets, masks)

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
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    def process_decoder_input(self, data, vocab_to_int, batch_size):
        ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
        return decoder_input

    def load_data(self):
        with open('data/letters_source.txt', 'r') as f:
            source_data = f.read()
        with open('data/letters_target.txt', 'r') as f:
            target_data = f.read()

        # 构造映射表
        source_int_to_letter, source_letter_to_int = self.extract_character_vocab(source_data)
        target_int_to_letter, target_letter_to_int = self.extract_character_vocab(target_data)

        # 对字母进行转换
        source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>'])
                       for letter in line] for line in source_data.split('\n')]
        target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
                       for letter in line] + [target_letter_to_int['<EOS>']] for line in target_data.split('\n')]
        return source_int_to_letter, source_letter_to_int, target_int_to_letter, target_letter_to_int, source_int, target_int

    def extract_character_vocab(self, data):  # data -> letter -> int
        special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
        set_words = list(set([character for line in data.split('\n') for character in line]))
        int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
        return int_to_vocab, vocab_to_int


def main():
    config = Config()
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('mode', '', 'mode')
    mode = FLAGS.mode  # train or predict
    lstm = LSTM(mode, config)

    if mode == 'train':
        lstm.train()
    else:
        lstm.predict()


if __name__ == '__main__':
    main()
