from __future__ import print_function

import tensorflow as tf
import numpy as np

HIDDEN_SIZE = 4  # should increase to compensate the biases
EPOCHS = 10**3
OUTPUT = 'sigmoid'  # 'relu' | 'sigmoid'

# define input / output placeholders to infer forward pass & compute loss
x_ = tf.placeholder(name="input", shape=[None, 2], dtype=tf.float32)
y_ = tf.placeholder(name="output", shape=[None, 1], dtype=tf.float32)

# 1st layer: weight matrix variable & activation
w1 = tf.Variable(tf.random_uniform(shape=[2, HIDDEN_SIZE]))
layer1 = tf.nn.relu(tf.matmul(x_, w1))

# 2nd layer: weight matrix variable & activation
w2 = tf.Variable(tf.random_uniform(shape=[HIDDEN_SIZE, 1]))
if OUTPUT == 'relu':
    nn_output = tf.nn.relu(tf.matmul(layer1, w2))
    # relu more straightforward in this way -> since works better than sigmoid
elif OUTPUT == 'sigmoid':
    nn_output = tf.nn.sigmoid(tf.matmul(layer1, w2))
else:
    nn_output = tf.matmul(layer1, w2)
    print('You can pass output activations as: "relu" or "sigmoid"')
print('Output activation:', OUTPUT)

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
