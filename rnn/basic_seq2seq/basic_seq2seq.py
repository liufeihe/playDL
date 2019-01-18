# -*- coding:utf-8 -*_

import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
import time

epochs = 60
batch_size = 128
rnn_size = 50
rnn_layers = 2
encoding_embedding_size = 15
decoding_embedding_size = 15
learning_rate = 0.001

source_int_to_letter = None
target_int_to_letter = None
source_letter_to_int = None
target_letter_to_int = None
source_int = None
target_int = None
checkpoint = './model/train_model.ckpt'


def predict():
    input_word = 'common'
    text = source_to_seq(input_word)
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(checkpoint+'.meta')
        print checkpoint
        loader.restore(sess, checkpoint)
        input_data = loaded_graph.get_tensor_by_name('inputs:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        answer_logits = sess.run(logits, {input_data: [text]*batch_size,
                                          target_sequence_length: [len(input_word)]*batch_size,
                                          source_sequence_length: [len(input_word)]*batch_size
                                          })[0]
    pad = source_letter_to_int['<PAD>']
    print 'input:'+input_word
    print '\n Source'
    print '   Word 编号： {}'.format([i for i in text])
    print '   Input Words: {}'.format(' '.join([source_int_to_letter[i] for i in text]))
    print '\n Target'
    print '   Word 编号： {}'.format([i for i in answer_logits if i!= pad])
    print '   Response Words: {}'.format(' '.join([target_int_to_letter[i] for i in answer_logits if i != pad]))


def source_to_seq(text):
    sequence_length = 7
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text] + \
           [source_letter_to_int['<PAD>']]*(sequence_length-len(text))


def train():
    train_source = source_int[batch_size:]
    train_target = target_int[batch_size:]
    valid_source = source_int[:batch_size]
    valid_target = target_int[:batch_size]
    (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = \
        next(get_batches(valid_target, valid_source, batch_size,
                         source_letter_to_int['<PAD>'],
                         target_letter_to_int['<PAD>']))
    display_step = 50
    train_graph, cost, train_op, input_data, targets, lr, target_sequence_length, source_sequence_length = \
        build_model()
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(1, epochs+1):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                get_batches(train_target, train_source, batch_size, source_letter_to_int['<PAD>'],target_letter_to_int['<PAD>'])
            ):
                _, loss = sess.run([train_op, cost], {
                                       input_data: sources_batch,
                                       targets: targets_batch,
                                       lr: learning_rate,
                                       target_sequence_length: targets_lengths,
                                       source_sequence_length: sources_lengths
                                   })
                if batch_i % display_step == 0:
                    validation_loss = sess.run([cost],{
                                       input_data: valid_sources_batch,
                                       targets: valid_targets_batch,
                                       lr: learning_rate,
                                       target_sequence_length: valid_targets_lengths,
                                       source_sequence_length: valid_sources_lengths
                                   })
                    print 'Epoch {:>3}/{} Batch {:>4}/{} - Trainging loss: {:>6.3f} - Validation loss: {:>6.3f}'\
                        .format(epoch_i, epochs, batch_i, len(train_source)//batch_size, loss, validation_loss[0])
            saver = tf.train.Saver()
            saver.save(sess, checkpoint)
            print 'model trained and saved'


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    for batch_i in range(0,len(sources)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i: start_i+batch_size]
        targets_batch = targets[start_i: start_i+batch_size]
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))
        source_lengths = []
        for source in targets_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths


def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch ])
    return [sentence+[pad_int]*(max_sentence-len(sentence)) for sentence in sentence_batch]


def build_model():
    train_graph = tf.Graph()
    with train_graph.as_default():
        input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()
        training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
                                                                           targets,
                                                                           lr,
                                                                           target_sequence_length,
                                                                           max_target_sequence_length,
                                                                           source_sequence_length,
                                                                           len(source_letter_to_int),
                                                                           len(target_letter_to_int),
                                                                           encoding_embedding_size,
                                                                           decoding_embedding_size,
                                                                           rnn_size,
                                                                           rnn_layers)
        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
        with tf.name_scope('optimization'):
            cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
            # optimizer = tf.train.AdamOptimizer(lr)
            # gradients = optimizer.compute_gradients(cost)
            # capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            # train_op = optimizer.apply_gradients(capped_gradients)
            training_variables = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, training_variables), 5)
            optimizer = tf.train.AdamOptimizer(lr)
            train_op = optimizer.apply_gradients(zip(grads, training_variables), name='train_op')
    return train_graph, cost, train_op, input_data, targets, lr, target_sequence_length, source_sequence_length


def seq2seq_model(input_data, targets, lr, target_sequence_length, max_target_sequence_length,
                  source_sequence_length, source_vocab_size, target_vocab_size,
                  encoder_embedding_size, decoder_embedding_size, rnn_size, rnn_layers):
    global target_letter_to_int
    _, encoder_state = get_encoder_layer(input_data,  rnn_size, rnn_layers,
                                         source_sequence_length, source_vocab_size,
                                         encoder_embedding_size)
    decoder_input = process_decoder_input(targets, target_letter_to_int, batch_size)
    training_decoder_output, predicting_decoder_output = decoding_layer(target_letter_to_int,
                                                                        decoder_embedding_size,
                                                                        rnn_layers,
                                                                        rnn_size,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        encoder_state,
                                                                        decoder_input)
    return training_decoder_output, predicting_decoder_output


def decoding_layer(target_letter_to_int, decoder_embedding_size, rnn_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, encoder_state, decoder_input):
    # 1. embedding
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoder_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)
    # decoder_embed_input = tf.contrib.layers.embed_sequence(decoder_input, target_vocab_size, decoder_embedding_size)

    # 2. construct the rnn
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1,0.1,seed=2))
        return decoder_cell
    cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(rnn_layers)])

    # 3. output fully connected
    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    # 4. training decoder
    with tf.variable_scope('decode'):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell, training_helper, encoder_state, output_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations=max_target_sequence_length)

    # 5. predicting decoder
    with tf.variable_scope('decode', reuse=True):
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']],dtype=tf.int32),
                               [batch_size], name='start_tokens')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, start_tokens, target_letter_to_int['<EOS>'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell, predicting_helper,
                                                             encoder_state, output_layer)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                         impute_finished=True,
                                                                         maximum_iterations=max_target_sequence_length)
    return training_decoder_output, predicting_decoder_output


def process_decoder_input(data, vocab_to_int, batch_size):
    ending = tf.strided_slice(data, [0,0], [batch_size, -1], [1,1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
    return decoder_input


def get_encoder_layer(input_data, rnn_size, rnn_cell_num,
                      source_sequence_length, source_vocab_size, encoder_embedding_size):
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoder_embedding_size)

    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1,0.1, seed=2))
        return lstm_cell
    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(rnn_cell_num)])

    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input, sequence_length=source_sequence_length, dtype=tf.float32)
    return encoder_output, encoder_state


def get_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_length')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length


def get_data():
    with open('data/letters_source.txt', 'r') as f:
        source_data = f.read()

    with open('data/letters_target.txt', 'r') as f:
        target_data = f.read()

    # 构造映射表
    source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
    target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)

    # 对字母进行转换
    source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>'])
                   for letter in line] for line in source_data.split('\n')]
    target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
                   for letter in line] + [target_letter_to_int['<EOS>']] for line in target_data.split('\n')]
    return source_int_to_letter, source_letter_to_int, target_int_to_letter, target_letter_to_int, source_int, target_int


def extract_character_vocab(data):  # data -> letter -> int
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    set_words = list(set([character for line in data.split('\n') for character in line]))
    int_to_vocab = {idx: word for idx, word in enumerate(special_words+set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int


def main():
    global source_int_to_letter, source_letter_to_int, \
        target_int_to_letter, target_letter_to_int, \
        source_int, target_int
    source_int_to_letter, source_letter_to_int, target_int_to_letter, target_letter_to_int, source_int, target_int = get_data()
    # train()
    predict()

if __name__ == '__main__':
    main()
