from __future__ import print_function

import tensorflow as tf

from graphs.mlp.mlp_00 import get_input, num_input, n_hidden_1, num_classes, save_dir
from helper import load_save_utils

my_feed_dict = {}
X = tf.placeholder("float", [None, num_input], name="input")
my_feed_dict[X] = get_input("input")

weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1]), name="l0W", dtype=tf.float32),
    'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes]), name="l1W", dtype=tf.float32)
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name="l0B"),
    'out': tf.Variable(tf.random_normal([num_classes]), name="l1B")
}


def neural_net(x):
    # We don't have auto broadcasting
    # layer_1 = tf.matmul(x, weights['h1']) + biases['b1']
    layer_1 = tf.nn.bias_add(tf.matmul(x, weights['h1']), biases['b1'], name="l0Preout")
    layer_1_post_actv = tf.abs(layer_1, name="l0Out")
    out_layer = tf.nn.bias_add(tf.matmul(layer_1_post_actv, weights['out']), biases['out'], name="l1PreOut")
    return out_layer


# Construct model
logits = neural_net(X)
output = tf.nn.softmax(logits, name='output')

init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    prediction = sess.run(output, feed_dict=my_feed_dict)
    load_save_utils.save_graph(sess, all_saver, save_dir)
    load_save_utils.save_prediction(save_dir, prediction)