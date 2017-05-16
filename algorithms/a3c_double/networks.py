from __future__ import print_function
import tensorflow as tf
import numpy as np

from relaax.algorithm_lib.lstm import CustomBasicLSTMCell
from config import cfg


class _Perception(object):
    def __init__(self):
        self.W_conv1 = _conv_weight_variable([8, 8, 3, 16])   # stride=4
        self.b_conv1 = _conv_bias_variable([16], 8, 8, 3)

        self.W_conv2 = _conv_weight_variable([4, 4, 16, 32])  # stride=2
        self.b_conv2 = _conv_bias_variable([32], 4, 4, 16)


class _A3CNetwork(_Perception):
    def __init__(self, thread_index):
        super(_A3CNetwork, self).__init__()
        W_fc1 = _fc_weight_variable([2592, 256])
        b_fc1 = _fc_bias_variable([256], 2592)

        # lstm
        self.lstm = CustomBasicLSTMCell(256)

        if (thread_index % 2) == 0:
            self.action_size = cfg.action_size0
        else:
            self.action_size = cfg.action_size1

        # weight for policy output layer
        W_fc2 = _fc_weight_variable([256, self.action_size])
        b_fc2 = _fc_bias_variable([self.action_size], 256)

        # weight for value output layer
        W_fc3 = _fc_weight_variable([256, 1])
        b_fc3 = _fc_bias_variable([1], 256)

        # state (input)
        self.s = tf.placeholder("float", [None] + cfg.state_size)  # (?, 84, 84, 3)

        h_conv1 = tf.nn.relu(_conv2d(self.s, self.W_conv1, 4) + self.b_conv1)
        # h_conv1 (?, 20, 20, 16)
        h_conv2 = tf.nn.relu(_conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)
        # h_conv2 (?, 9, 9, 32)

        h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
        # h_conv2_flat(?, 2592)
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
        # h_fc1 (?, 256)
        h_fc1_reshaped = tf.reshape(h_fc1, [1, -1, 256])
        # h_fc_reshaped (1, ?, 256)

        # placeholders for LSTM unrolling time step size & initial_lstm_state
        self.step_size = tf.placeholder(tf.float32, [1])
        self.initial_lstm_state = tf.placeholder(tf.float32, [1, self.lstm.state_size])

        scope = "net_" + str(thread_index)
        lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                          h_fc1_reshaped,
                                                          initial_state=self.initial_lstm_state,
                                                          sequence_length=self.step_size,
                                                          time_major=False,
                                                          scope=scope)
        # lstm_outputs (1, ?, 256)
        self.weights = [
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            W_fc1, b_fc1,
            self.lstm.matrix, self.lstm.bias,
            W_fc2, b_fc2,
            W_fc3, b_fc3
        ]

        lstm_outputs = tf.reshape(lstm_outputs, [-1, 256])
        # lstm_outputs (?, 256)

        # policy (output)
        self.pi = tf.nn.softmax(tf.matmul(lstm_outputs, W_fc2) + b_fc2)
        # self.pi(?, action_size)

        # value (output)
        v_ = tf.matmul(lstm_outputs, W_fc3) + b_fc3
        # v_(?, 1)
        self.v = tf.reshape(v_, [-1])
        # self.v (?,)

        self.lstm_state_out = np.zeros([1, self.lstm.state_size])


class _A3CDoubleNetwork(_Perception):
    def __init__(self, thread_index):
        super(_A3CDoubleNetwork, self).__init__()
        W_fc01 = _fc_weight_variable([2592, 256])
        b_fc01 = _fc_bias_variable([256], 2592)

        W_fc11 = _fc_weight_variable([2592, 256])
        b_fc11 = _fc_bias_variable([256], 2592)

        # lstm
        self.lstm0 = CustomBasicLSTMCell(256)
        self.lstm1 = CustomBasicLSTMCell(256)

        # weight for policy output layer
        W_fc02 = _fc_weight_variable([256, cfg.action_size0])
        b_fc02 = _fc_bias_variable([cfg.action_size0], 256)

        W_fc12 = _fc_weight_variable([256, cfg.action_size1])
        b_fc12 = _fc_bias_variable([cfg.action_size1], 256)

        # weight for value output layer
        W_fc03 = _fc_weight_variable([256, 1])
        b_fc03 = _fc_bias_variable([1], 256)

        W_fc13 = _fc_weight_variable([256, 1])
        b_fc13 = _fc_bias_variable([1], 256)

        # state (input)
        self.s0 = tf.placeholder("float", [None] + cfg.state_size)  # (?, 84, 84, 3)
        self.s1 = tf.placeholder("float", [None] + cfg.state_size)  # (?, 84, 84, 3)

        h_conv01 = tf.nn.relu(_conv2d(self.s0, self.W_conv1, 4) + self.b_conv1)
        h_conv11 = tf.nn.relu(_conv2d(self.s1, self.W_conv1, 4) + self.b_conv1)
        # h_conv1 (?, 20, 20, 16)
        h_conv02 = tf.nn.relu(_conv2d(h_conv01, self.W_conv2, 2) + self.b_conv2)
        h_conv12 = tf.nn.relu(_conv2d(h_conv11, self.W_conv2, 2) + self.b_conv2)
        # h_conv2 (?, 9, 9, 32)

        h_conv02_flat = tf.reshape(h_conv02, [-1, 2592])
        h_conv12_flat = tf.reshape(h_conv12, [-1, 2592])
        # h_conv2_flat(?, 2592)
        h_fc01 = tf.nn.relu(tf.matmul(h_conv02_flat, W_fc01) + b_fc01)
        h_fc11 = tf.nn.relu(tf.matmul(h_conv12_flat, W_fc11) + b_fc11)
        # h_fc1 (?, 256)
        h_fc01_reshaped = tf.reshape(h_fc01, [1, -1, 256])
        h_fc11_reshaped = tf.reshape(h_fc11, [1, -1, 256])
        # h_fc_reshaped (1, ?, 256)

        # placeholders for LSTM unrolling time step size & initial_lstm_state
        self.step_size0 = tf.placeholder(tf.float32, [1])
        self.step_size1 = tf.placeholder(tf.float32, [1])
        self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, self.lstm0.state_size])
        self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, self.lstm1.state_size])

        scope0 = "net0_" + str(thread_index)
        lstm_outputs0, self.lstm_state0 =\
            tf.nn.dynamic_rnn(self.lstm0,
                              h_fc01_reshaped,
                              initial_state=self.initial_lstm_state0,
                              sequence_length=self.step_size0,
                              time_major=False,
                              scope=scope0)
        scope1 = "net1_" + str(thread_index)
        lstm_outputs1, self.lstm_state1 =\
            tf.nn.dynamic_rnn(self.lstm1,
                              h_fc11_reshaped,
                              initial_state=self.initial_lstm_state1,
                              sequence_length=self.step_size1,
                              time_major=False,
                              scope=scope1)
        # lstm_outputs (1, ?, 256)
        self.weights0 = [
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            W_fc01, b_fc01,
            self.lstm0.matrix, self.lstm0.bias,
            W_fc02, b_fc02,
            W_fc03, b_fc03
        ]
        self.weights1 = [
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            W_fc11, b_fc11,
            self.lstm1.matrix, self.lstm1.bias,
            W_fc12, b_fc12,
            W_fc13, b_fc13
        ]


class A3CGlobalNetwork(_A3CDoubleNetwork):
    def __init__(self, thread_index=-1):
        super(A3CGlobalNetwork, self).__init__(thread_index)
        self.learning_rate_input0 = tf.placeholder(tf.float32, [], name="lr0")
        self.learning_rate_input1 = tf.placeholder(tf.float32, [], name="lr1")

        self.optimizer0 = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate_input0,
            decay=cfg.RMSP_ALPHA,
            momentum=0.0,
            epsilon=cfg.RMSP_EPSILON
        )
        self.optimizer1 = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate_input1,
            decay=cfg.RMSP_ALPHA,
            momentum=0.0,
            epsilon=cfg.RMSP_EPSILON
        )


class A3CLocalNetwork(_A3CNetwork):
    def __init__(self, thread_index):
        super(A3CLocalNetwork, self).__init__(thread_index)
        # taken action (input for policy)
        self.a = tf.placeholder("float", [None, self.action_size])

        # temporary difference (R-V) (input for policy)
        self.td = tf.placeholder("float", [None])

        # avoid NaN with getting the maximum with small value
        log_pi = tf.log(tf.maximum(self.pi, 1e-20))

        # policy entropy
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)

        # policy loss (output)  (Adding minus, because the original paper's
        # objective function is for gradient ascent, but we use gradient descent optimizer)
        policy_loss = -tf.reduce_sum(
            tf.reduce_sum(tf.multiply(log_pi, self.a), axis=1) * self.td + entropy * cfg.entropy_beta)

        # R (input for value)
        self.r = tf.placeholder("float", [None])

        # value loss (output)
        # (Learning rate for Critic is half of Actor's, it's l2 without dividing by 0.5)
        value_loss = tf.reduce_sum(tf.square(self.r - self.v))

        # gradient of policy and value are summed up
        self.total_loss = policy_loss + value_loss

    def sync_from(self, src_weights):
        sync_ops = []
        for (src_var, dst_var) in zip(src_weights, self.weights):
            sync_op = tf.assign(dst_var, src_var)
            sync_ops.append(sync_op)

        return tf.group(*sync_ops)

    def reset_state(self):
        self.lstm_state_out = np.zeros([1, self.lstm.state_size])

    def run_policy_and_value(self, sess, s_t):
        pi_out, v_out, self.lstm_state_out = sess.run([self.pi, self.v, self.lstm_state],
                                                      feed_dict={self.s: [s_t],
                                                                 self.initial_lstm_state: self.lstm_state_out,
                                                                 self.step_size: [1]})
        # pi_out.shape(1, action_size), v_out.shape(1, 1)-> reshaped to (1,)
        return pi_out[0], v_out[0]

    def run_policy(self, sess, s_t):
        pi_out, self.lstm_state_out = sess.run([self.pi, self.lstm_state],
                                               feed_dict={self.s: [s_t],
                                                          self.initial_lstm_state: self.lstm_state_out,
                                                          self.step_size: [1]})
        # pi_out.shape(1, action_size)
        return pi_out[0]

    def run_value(self, sess, s_t):
        prev_lstm_state_out = self.lstm_state_out
        v_out, _ = sess.run([self.v, self.lstm_state],
                            feed_dict={self.s: [s_t],
                                       self.initial_lstm_state: self.lstm_state_out,
                                       self.step_size: [1]})
        # roll back lstm state
        self.lstm_state_out = prev_lstm_state_out
        # v_out.shape(1, 1)-> reshaped to (1,)
        return v_out[0]


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
