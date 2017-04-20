import tensorflow as tf
import numpy as np

from config import cfg


class _A3CNetwork(object):
    def __init__(self):
        self.global_t = tf.Variable(0, tf.int64)
        self.increment_global_t = tf.assign_add(self.global_t, 1)

        self.W_conv1 = _conv_weight_variable([8, 8, 4, 16])  # stride=4
        self.b_conv1 = _conv_bias_variable([16], 8, 8, 4)

        self.W_conv2 = _conv_weight_variable([4, 4, 16, 32])  # stride=2
        self.b_conv2 = _conv_bias_variable([32], 4, 4, 16)

        self.W_fc1 = _fc_weight_variable([2592, 256])
        self.b_fc1 = _fc_bias_variable([256], 2592)

        # weight for policy output layer
        self.W_fc2 = _fc_weight_variable([256, cfg.action_size])
        self.b_fc2 = _fc_bias_variable([cfg.action_size], 256)

        # weight for value output layer
        self.W_fc3 = _fc_weight_variable([256, 1])
        self.b_fc3 = _fc_bias_variable([1], 256)

        self.values = [
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3
        ]


def _conv_weight_variable(shape):
    w = shape[0]
    h = shape[1]
    input_channels = shape[2]
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


def _conv_bias_variable(shape, w, h, input_channels):
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


# weight initialization
def _fc_weight_variable(shape):
    input_channels = shape[0]
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


def _fc_bias_variable(shape, input_channels):
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


def _conv2d(x, w, stride):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding="VALID")
