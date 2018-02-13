import tensorflow as tf
from tests.rnn.gru_dynamic_simple import featuresize, timesteps, minibatch, get_input, get_output, save_dir

from tfoptests import load_save_utils

num_hidden = 3
num_layers = 5
learning_rate = 0.01
training_steps = 1000
display_step = 100

# tf Graph input
X = tf.placeholder("float", [None, featuresize, timesteps - 1], name="input")
Y = tf.placeholder("float", [None, featuresize])
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, featuresize]))
}
biases = {
    'out': tf.Variable(tf.random_normal([featuresize]))
}


def RNN(data, target):
    cells = []
    for _ in range(num_layers):
        cell = tf.contrib.rnn.GRUCell(num_hidden)  # Or LSTMCell(num_units)
        cells.append(cell)
    network = tf.contrib.rnn.MultiRNNCell(cells)
    output, _ = tf.nn.dynamic_rnn(network, data, dtype=tf.float32)
    output = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output, int(output.get_shape()[0]) - 1)
    # Softmax layer.
    weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])], stddev=0.01))
    bias = tf.Variable(tf.constant(0.1, shape=[int(target.get_shape()[1])]))
    aprediction = tf.nn.softmax(tf.matmul(last, weight) + bias, name="output")
    return aprediction


preout = RNN(X, Y)
output = tf.identity(preout, name="output")

# Define loss and optimizer
loss_op = tf.reduce_sum(tf.pow(output - Y, 2)) / (2 * minibatch)  # MSE
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
accuracy = tf.pow(output - Y, 2)

all_saver = tf.train.Saver()
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for step in range(1, training_steps + 1):
        sess.run(train_op, feed_dict={X: get_input("input"), Y: get_output("output")})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: get_input("input"),
                                                                 Y: get_output("output")})
            print("Step " + str(step) + ", Loss= " + \
                  "{:.4f}".format(loss))
            # print(acc)
    print("Optimization Finished!")

    prediction = sess.run(output, feed_dict={X: get_input("input")})
    print(get_input("input"))
    print(get_input("input").shape)
    print("====")
    print(get_output("output"))
    print("====")
    print(prediction)
    load_save_utils.save_prediction(save_dir, prediction)
    load_save_utils.save_graph(sess, all_saver, save_dir)