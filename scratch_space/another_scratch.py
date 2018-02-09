import tensorflow as tf
import numpy as np


class self_coding_network(object):
    def __init__(self, training_epochs, n_hidden_1, n_hidden_2):
        self.training_epochs = training_epochs
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.learning_rate = 0.01
        self.batch_size = 128
        self.display_step = 1
        self.n_input = 676
        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1], dtype=tf.float64)),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2], dtype=tf.float64)),
            'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_1], dtype=tf.float64)),
            'decoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_input], dtype=tf.float64))
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1], dtype=tf.float64)),
            'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2], dtype=tf.float64)),
            'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1], dtype=tf.float64)),
            'decoder_b2': tf.Variable(tf.random_normal([self.n_input], dtype=tf.float64))
        }

        self.X = tf.placeholder("float64", [None, self.n_input], name='input')

        # define model
        self.encoder_op = self.encoder(self.X)
        self.decoder_op = self.decoder(self.encoder_op)

        # predict
        self.y_pred = self.decoder_op
        self.y_true = self.X

        # cost function and optimizer
        self.cost = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

    def encoder(self, x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']), name="output")
        return layer_2

    def decoder(self, x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                       self.biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                       self.biases['decoder_b2']))
        return layer_2


if __name__ == '__main__':
    # init auto-encoder
    aa = self_coding_network(40, 60, 2)
    sess = tf.Session()
    tf.train.write_graph(sess.graph_def,
                         logdir="/Users/susaneraly/SKYMIND/dl4j-test-resources/src/main/resources/tf_graphs/examples",
                         name="unfrozen_simple_ae.pb", as_text=False)
    tf.train.write_graph(sess.graph_def,
                         logdir="/Users/susaneraly/SKYMIND/dl4j-test-resources/src/main/resources/tf_graphs/examples",
                         name="unfrozen_simple_ae.pb.txt")
