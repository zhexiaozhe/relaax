import tensorflow as tf
import numpy as np

HIDDEN_SIZE = 2
EPOCHS = 10**3

# define input / output placeholders to infer forward pass & compute loss
x_ = tf.placeholder(name="input", shape=[None, 2], dtype=tf.float32)
y_ = tf.placeholder(name="output", shape=[None, 1], dtype=tf.float32)

# 1st layer: weight matrix variable + bias vector & activation
w1 = tf.Variable(tf.random_uniform(shape=[2, HIDDEN_SIZE]))
b1 = tf.Variable(tf.constant(value=0.0, shape=[HIDDEN_SIZE], dtype=tf.float32))
layer1 = tf.nn.relu(tf.add(tf.matmul(x_, w1), b1))

# 2nd layer: weight matrix variable + bias vector & activation
w2 = tf.Variable(tf.random_uniform(shape=[HIDDEN_SIZE, 1]))
b2 = tf.Variable(tf.constant(value=0.0, shape=[1], dtype=tf.float32))
nn_output = tf.nn.relu(tf.add(tf.matmul(layer1, w2), b2))

# define optimizer & diff loss & train_op
adam = tf.train.AdamOptimizer(1e-2)
loss = tf.reduce_mean(tf.square(nn_output - y_))
train_op = adam.minimize(loss)

# generate xor table and its ground truth
x = np.array([[0, 0],
              [1, 0],
              [0, 1],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

# try to train
with tf.Session() as sess:
    sess.run(tf.variables_initializer(tf.global_variables()))
    # batched all table within number of epochs
    for _ in range(EPOCHS):
        sess.run(train_op, feed_dict={x_: x, y_: y})
    # check how well we trained
    print(sess.run(nn_output, feed_dict={x_: x}))
