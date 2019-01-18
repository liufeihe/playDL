# -*- coding: utf-8 -*-

import collections


import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


def load_data(path):
    sentences = []
    if not path:
        return sentences

    with open(path) as f:
        sentences.extend(f.readlines())

    return sentences


def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    tk = Tokenizer()
    tk.fit_on_texts(x)
    return tk.texts_to_sequences(x), tk


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    if length is None:
        length = max([len(sentence) for sentence in x])

    return pad_sequences(x, maxlen=length, padding="post")



def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*(preprocess_y.shape+(1,)))

    return preprocess_x, preprocess_y, x_tk, y_tk


def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Build the layers
    learning_rate = 1e-3

    input_seq = Input(input_shape[1:])
    rnn = GRU(64, return_sequences=True)(input_seq)
    logits = TimeDistributed(Dense(french_vocab_size))(rnn)

    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model


def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a RNN model using word embedding on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Implement
    learning_rate = 1e-3

    input_seq = Input(input_shape[1:])
    embed_seq = Embedding(english_vocab_size, 64)(input_seq)
    rnn = GRU(64, return_sequences=True)(embed_seq)
    logits = TimeDistributed(Dense(french_vocab_size))(rnn)

    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model


def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a bidirectional RNN model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Implement
    learning_rate = 1e-3

    input_seq = Input(input_shape[1:])
    bi_rnn = Bidirectional(GRU(64, return_sequences=True))(input_seq)
    logits = TimeDistributed(Dense(french_vocab_size))(bi_rnn)

    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model


def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train an encoder-decoder model on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # OPTIONAL: Implement
    learning_rate = 1e-3

    input_seq = Input(shape=input_shape[1:])
    rnn = GRU(64, return_sequences=False)(input_seq)

    c_rnn = RepeatVector(output_sequence_length)(rnn)
    d_rnn = GRU(64, return_sequences=True)(c_rnn)
    logits = TimeDistributed(Dense(french_vocab_size))(d_rnn)

    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model


def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a model that incorporates embedding, encoder-decoder, and bidirectional RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Implement
    learning_rate = 1e-3

    input_seq = Input(input_shape[1:])
    embed_seq = Embedding(english_vocab_size, 64, input_length=output_sequence_length)(input_seq)
    bi_rnn = Bidirectional(GRU(64, return_sequences=False))(embed_seq)

    c_rnn = RepeatVector(output_sequence_length)(bi_rnn)
    d_rnn = Bidirectional(GRU(64, return_sequences=True))(c_rnn)
    logits = TimeDistributed(Dense(french_vocab_size))(d_rnn)

    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model


def playMt():
    english_sentences = load_data('data/small_vocab_en')
    print english_sentences[:10]
    french_sentences = load_data('data/small_vocab_fr')
    print french_sentences[:10]

    preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = \
        preprocess(english_sentences, french_sentences)
    print preproc_english_sentences.shape
    print preproc_french_sentences.shape

    max_english_sequence_length = preproc_english_sentences.shape[1]
    max_french_sequence_length = preproc_french_sentences.shape[1]
    english_vocab_size = len(english_tokenizer.word_index)
    french_vocab_size = len(french_tokenizer.word_index)

    # Reshaping the input to work with a basic RNN
    # tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
    tmp_x = pad(preproc_english_sentences, max_english_sequence_length) # enddec_model

    # tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))
    # tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2])) # embed_model
    tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-1], 1)) # encdec_model

    # Train the neural network
    rnn_model = encdec_model(
        tmp_x.shape,
        max_french_sequence_length,
        english_vocab_size+1,
        french_vocab_size+1)
    rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

    # Print prediction(s)
    print(logits_to_text(rnn_model.predict(tmp_x[:1])[0], french_tokenizer))


def main():
    playMt()


if __name__ == '__main__':
    main()
