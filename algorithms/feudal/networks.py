from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

from relaax.algorithm_lib.lstm import CustomBasicLSTMCell
from relaax.algorithm_lib.lstm import DilatedLSTMCell
from config import cfg


class _Perception(object):
    def __init__(self):
        self.W_conv1 = _conv_weight_variable([8, 8, 3, 16])   # stride=4
        self.b_conv1 = _conv_bias_variable([16], 8, 8, 3)

        self.W_conv2 = _conv_weight_variable([4, 4, 16, 32])  # stride=2
        self.b_conv2 = _conv_bias_variable([32], 4, 4, 16)

        self.W_fc1 = _fc_weight_variable([2592, cfg.d])
        self.b_fc1 = _fc_bias_variable([cfg.d], 2592)  # d == 256

        # state (input)
        self.s = tf.placeholder("float", [None] + cfg.state_size)  # (?, 84, 84, 3)

        h_conv1 = tf.nn.relu(_conv2d(self.s, self.W_conv1, 4) + self.b_conv1)
        # h_conv1 (?, 20, 20, 16)
        h_conv2 = tf.nn.relu(_conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)
        # h_conv2 (?, 9, 9, 32)

        h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
        # h_conv2_flat(?, 2592)
        self.perception = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)
        # h_fc1 (?, d)


class ManagerNetwork:
    def __init__(self):
        # weight & bias for policy output layer
        W_Mspace = _fc_weight_variable([cfg.d, cfg.d])
        b_Mspace = _fc_bias_variable([cfg.d], cfg.d)

        # weight & bias for manager's internal critic
        W_Mcritic = _fc_weight_variable([cfg.d, 1])
        b_Mcritic = _fc_bias_variable([1], cfg.d)

        # perception (input) -> transform by Mspace
        self.ph_perception = tf.placeholder(tf.float32, shape=[None, cfg.d],
                                            name="ph_perception")
        # ph_perception (?, d)

        self.Mspace = tf.nn.relu(tf.matmul(self.ph_perception, W_Mspace) + b_Mspace)
        # Mspace (?, d)

        h_fc_reshaped = tf.reshape(self.Mspace, [1, -1, cfg.d])
        # h_fc_reshaped (1, ?, d)

        # lstm
        self.lstm = DilatedLSTMCell(cfg.d, num_cores=cfg.h)

        # placeholders for LSTM unrolling time step size & initial_lstm_state
        self.step_size = tf.placeholder(tf.float32, [1], name="manager_step_size")
        self.initial_lstm_state = tf.placeholder(tf.float32, [1, self.lstm.state_size],
                                                 name="manager_lstm_state")

        lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                          h_fc_reshaped,
                                                          initial_state=self.initial_lstm_state,
                                                          sequence_length=self.step_size,
                                                          time_major=False,
                                                          scope="manager")
        # lstm_outputs (1, ?, d)
        self.weights = [
            W_Mspace, b_Mspace,
            self.lstm.matrix, self.lstm.bias,
            W_Mcritic, b_Mcritic
        ]

        lstm_outputs = tf.reshape(lstm_outputs, [-1, cfg.d])
        # lstm_outputs (?, d)

        # goal (output)
        self.goal = tf.nn.l2_normalize(lstm_outputs, dim=0)

        # value (output)
        v_ = tf.matmul(lstm_outputs, W_Mcritic) + b_Mcritic
        # v_(?, 1)
        self.v = tf.reshape(v_, [-1])
        # self.v (?,)

        # loss' stuff
        self.stc_minus_st, self.cosine_similarity = None, None
        self._prepare_loss()

        # optimizer's stuff
        self.learning_rate_input, self.optimizer = None, None
        self._prepare_optimizer()

        self.lstm_state_out = None
        self.reset_state()

    def reset_state(self):
        self.lstm_state_out = np.zeros([1, self.lstm.state_size])

    def _prepare_loss(self):
        # tf.losses.cosine_distance(labels, predictions)
        self.stc_minus_st = tf.placeholder(tf.float32, [None, cfg.d], name="stc_minus_st")
        s_diff_normalized = tf.nn.l2_normalize(self.stc_minus_st, dim=0)

        self.cosine_similarity = tf.matmul(s_diff_normalized, tf.transpose(self.goal))

    def _prepare_optimizer(self):
        self.learning_rate_input = tf.placeholder(tf.float32, [], name="manager_lr")

        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate_input,
            decay=cfg.RMSP_ALPHA,
            momentum=0.0,
            epsilon=cfg.RMSP_EPSILON
        )

    def run_goal_value_st(self, sess, z_t):
        s_t, self.lstm_state_out, g_out, v_out =\
            sess.run([self.Mspace, self.lstm_state, self.goal, self.v],
                     feed_dict={self.ph_perception: z_t,
                                self.initial_lstm_state: self.lstm_state_out,
                                self.step_size: [1]})
        # pi_out.shape(1, d), v_out.shape(1, 1)-> reshaped to (1,)
        s_t = np.reshape(s_t, (-1))
        # s_t(1, d)-> s_t(d, )
        return g_out[0], v_out[0], s_t


class _WorkerNetwork(_Perception):
    def __init__(self, thread_index):
        super(_WorkerNetwork, self).__init__()
        # lstm
        self.lstm = CustomBasicLSTMCell(cfg.d)  # d == 256

        # weight & bias for embedding matrix U: action_size * k
        W_fcU = _fc_weight_variable([cfg.d, cfg.action_size * cfg.k])
        b_fcU = _fc_bias_variable([cfg.action_size * cfg.k], cfg.d)

        # weight & bias for worker's internal critic
        W_Wcritic = _fc_weight_variable([cfg.d, 1])
        b_Wcritic = _fc_bias_variable([1], cfg.d)

        # Goal placeholder == d summed up from c horizon (manager's goal input)
        self.ph_goal = tf.placeholder(tf.float32, [None, cfg.d], name="ph_goal")
        W_phi = _fc_weight_variable([cfg.d, cfg.k])  # phi is goal linear transform

        h_fc_reshaped = tf.reshape(self.perception, [1, -1, cfg.d])
        # h_fc_reshaped (1, ?, d)

        # placeholders for LSTM unrolling time step size & initial_lstm_state
        self.step_size = tf.placeholder(tf.float32, [1], name="step_size")
        self.initial_lstm_state = tf.placeholder(tf.float32, [1, self.lstm.state_size],
                                                 name="worker_lstm_state")

        scope = "net_" + str(thread_index)
        lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                          h_fc_reshaped,
                                                          initial_state=self.initial_lstm_state,
                                                          sequence_length=self.step_size,
                                                          time_major=False,
                                                          scope=scope)
        # lstm_outputs (1, ?, d)
        self.weights = [
            self.W_conv1, self.b_conv1,
            self.W_conv2, self.b_conv2,
            self.W_fc1, self.b_fc1,
            self.lstm.matrix, self.lstm.bias,
            W_fcU, b_fcU,
            W_Wcritic, b_Wcritic,
            W_phi
        ]

        lstm_outputs = tf.reshape(lstm_outputs, [-1, cfg.d])
        # lstm_outputs (?, d)

        # U embedding matrix as row vector (worker lstm output)
        U_ = tf.matmul(lstm_outputs, W_fcU) + b_fcU
        # U_(?, action_size * k)

        U_reshaped = tf.reshape(U_, [cfg.action_size, cfg.k, -1])
        # U_reshaped(action_size, k) <-- (?, action_size, k)

        # linear transform from goal to w through phi (without bias)
        w = tf.matmul(self.ph_goal, W_phi)
        # w(?, k)

        w_reshaped = tf.reshape(w, [-1, 1, cfg.k])
        # w_reshaped(?, 1, k)

        # action probs (output)
        pi_ = tf.nn.softmax(tf.matmul(w_reshaped, tf.transpose(U_reshaped)))
        # pi_(?, 1, 18))
        self.pi = tf.reshape(pi_, [-1, cfg.action_size])
        # self.pi(?, 18)

        # value (output)
        v_ = tf.matmul(lstm_outputs, W_Wcritic) + b_Wcritic
        # v_(?, 1)
        self.v = tf.reshape(v_, [-1])
        # self.v (?,)

        self.lstm_state_out = np.zeros([1, self.lstm.state_size])


class GlobalWorkerNetwork(_WorkerNetwork):
    def __init__(self, thread_index=-1):
        super(GlobalWorkerNetwork, self).__init__(thread_index)
        self.learning_rate_input = tf.placeholder(tf.float32, [], name="worker_lr")

        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate_input,
            decay=cfg.RMSP_ALPHA,
            momentum=0.0,
            epsilon=cfg.RMSP_EPSILON
        )


class LocalWorkerNetwork(_WorkerNetwork):
    def __init__(self, thread_index):
        super(LocalWorkerNetwork, self).__init__(thread_index)
        # taken action (input for policy)
        self.a = tf.placeholder(tf.float32, [None, cfg.action_size], name="a")

        # temporary difference (R-V) (input for policy)
        self.td = tf.placeholder(tf.float32, [None], name="td")

        # avoid NaN with getting the maximum with small value
        log_pi = tf.log(tf.maximum(self.pi, 1e-20))

        # policy entropy
        entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)

        # policy loss (output)  (Adding minus, because the original paper's
        # objective function is for gradient ascent, but we use gradient descent optimizer)
        policy_loss = -tf.reduce_sum(
            tf.reduce_sum(tf.multiply(log_pi, self.a), axis=1) * self.td + entropy * cfg.entropy_beta)

        # R (input for value)
        self.r = tf.placeholder(tf.float32, [None], name="r")

        # value loss (output)
        # (Learning rate for Critic is half of Actor's, it's l2 without dividing by 0.5)
        value_loss = tf.reduce_sum(tf.square(self.r - self.v))

        # gradient of policy and value are summed up
        self.total_loss = policy_loss + value_loss

    def sync_from(self, src_network):
        sync_ops = []
        for (src_var, dst_var) in zip(src_network.weights, self.weights):
            sync_op = tf.assign(dst_var, src_var)
            sync_ops.append(sync_op)

        return tf.group(*sync_ops)

    def reset_state(self):
        self.lstm_state_out = np.zeros([1, self.lstm.state_size])

    def run_policy_and_value(self, sess, s_t, goal):
        pi_out, v_out, self.lstm_state_out =\
            sess.run([self.pi, self.v, self.lstm_state],
                     feed_dict={self.s: [s_t],
                                self.ph_goal: [goal],
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
