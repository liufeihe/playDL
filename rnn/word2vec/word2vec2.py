# -*- coding: utf-8 -*-
import collections
import numpy as np
import math
import time
import os
import random
import zipfile
import urllib
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Config(object):
    url = 'http://mattmahoney.net/dc/'
    epochs = 1000
    batch_size = 128
    window_size = 10
    n_embedding = 128
    n_sampled = 100
    learning_rate = 1.0
    vocabulary_size = 50000

    valid_size = 16
    valid_window = 100


class Word2Vec(object):
    def __init__(self, cfg):
        self.config = cfg
        self.train_words = self.build_datasets()
        self.build_graph()

    def train(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            iteration = 1
            loss = 0
            sess.run(tf.global_variables_initializer())
            epochs = self.config.epochs
            for e in range(1, epochs+1):
                batches = self.get_batches()
                start = time.time()
                for x, y in batches:
                    feed = {self.inputs: x, self.labels:np.array(y)[:, None]}
                    train_loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed)
                    loss += train_loss
                    if iteration%100 == 0:
                        end = time.time()
                        print "Epoch {}/{}, Iteration: {}, Avg. Training loss: {:.4f}, {:.4f} sec/batch".format(e, epochs, iteration, loss/100, (end-start)/100)
                        loss = 0
                        start = time.time()
                    if iteration%1000 == 0:
                        sim = self.similarity.eval()
                        for i in range(self.config.valid_size):
                            valid_word = self.int_to_vocab[self.valid_examples[i]]
                            top_k = 8
                            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                            log_str = 'nearest to %s:' % valid_word
                            for k in range(top_k):
                                close_word = self.int_to_vocab[nearest[k]]
                                log_str = '%s %s,' % (log_str, close_word)
                            print log_str
                    iteration += 1

    def predict(self):
        pass

    def build_graph(self):
        n_vocab = self.n_vocab
        n_embedding = self.config.n_embedding
        n_sampled = self.config.n_sampled
        self.valid_examples = np.random.choice(self.config.valid_window, self.config.valid_size, replace=False)

        with tf.variable_scope('input'):
            self.inputs = tf.placeholder(tf.int32, [None], name='inputs')
            self.labels = tf.placeholder(tf.int32, [None, None], name='labels')
            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

        with tf.variable_scope('embedding'):
            self.embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding),-1,1))
            embed = tf.nn.embedding_lookup(self.embedding, self.inputs)

        with tf.variable_scope('loss'):
            # softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding), stddev=1.0))
            # softmax_b = tf.Variable(tf.zeros(n_vocab))
            # # 负采样
            # self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_w, softmax_b, self.labels, embed, n_sampled, n_vocab))
            nce_weights = tf.Variable(tf.truncated_normal([n_vocab, n_embedding],
                                                          stddev=1.0 / math.sqrt(n_embedding)))
            nce_biases = tf.Variable(tf.zeros([n_vocab]))
            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             labels=self.labels,
                                             inputs=embed,
                                             num_sampled=n_sampled,
                                             num_classes=n_vocab))


        with tf.variable_scope('optimize'):
            # optimizer = tf.train.AdamOptimizer(1.0).minimize(self.loss)
            optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss)
            self.optimizer = optimizer

            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding), 1, keep_dims=True))
            normalized_embeddings = self.embedding/norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
            self.similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    def get_batches(self):
        batch_size = self.config.batch_size
        words = self.train_words
        n_batches = len(words)//batch_size
        words = words[:n_batches*batch_size]
        for idx in range(0, len(words), batch_size):
            batch = words[idx: idx+batch_size]
            x, y = [], []
            for ii in range(len(batch)):
                batch_x = batch[ii]
                batch_y = self.get_target(batch, ii)
                y.extend(batch_y)
                x.extend([batch_x]*len(batch_y))
            yield x,y

    def get_target(self, batch, idx):
        window_size = self.config.window_size
        R = np.random.randint(1, window_size+1)
        start = idx-R if idx-R>0 else 0
        end = idx+R
        targets = set(batch[start:idx]+batch[idx+1:end+1])
        return list(targets)

    def build_datasets(self):
        # get words
        print 'begin loading data...'
        words = self.load_data()
        int_to_vocab, vocab_to_int = self.create_lookup_table(words)
        int_words = [vocab_to_int[word] for word in words]
        self.n_vocab = len(int_to_vocab)
        self.int_to_vocab = int_to_vocab

        # 子采样，舍弃高频单词的过程
        word_counts = collections.Counter(int_words)
        total_count = len(int_words)
        word_freqs = {word: count*1.0/total_count for word, count in word_counts.items()}
        threshold = 1e-5
        word_drop = {word: (1-np.sqrt(threshold/word_freqs[word])) for word in word_counts}
        train_words = [word for word in int_words if random.random()<(1-word_drop[word])]
        print 'building data sets ok.'
        return train_words

    def create_lookup_table(self, words):
        word_counts = collections.Counter(words)
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        int_to_vocab = {idx: word for idx, word in enumerate(sorted_vocab)}
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
        return int_to_vocab, vocab_to_int

    def load_data(self):
        filename = self.maybe_download('./data/text8.zip', 31344016)
        with zipfile.ZipFile(filename) as f:
            words = tf.compat.as_str(f.read(f.namelist()[0])).split()
            if self.config.vocabulary_size:
                word_counts_list = collections.Counter(words).most_common(self.config.vocabulary_size)
                trimmed_words = [word_count[0] for word_count in word_counts_list]
            else:
                # Remove all words with  5 or fewer occurences
                word_counts = collections.Counter(words)
                trimmed_words = [word for word in words if word_counts[word] > 5]
        return trimmed_words

    def maybe_download(self, filename, expected_bytes):
        if not os.path.exists(filename):
            filename, _ = urllib.urlretrieve(self.config.url + filename, filename)
        status_info = os.stat(filename)

        if status_info.st_size == expected_bytes:
            print 'Found and verified %s' % filename
        else:
            print status_info.st_size
            raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
        return filename


def main():
    cfg = Config()
    word2vec = Word2Vec(cfg)
    word2vec.train()


if __name__ == '__main__':
    main()