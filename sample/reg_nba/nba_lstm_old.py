from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

from custom_lstm import CustomBasicLSTMCell as BasicLSTM
from tensorflow.contrib.keras.python.keras.initializers import he_normal

FEAT_IDX = \
    [7, 8, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 70, 71, 74, 75, 80, 83, 86, 87, 90, 91]
NEXT_IDX = [7, 8, 57, 14, 15, 16, 18, 19, 20, 21, 22, 26, 28]
LBL_IDX = [58, 59]  # [58, 59, 83]
PTS_IDX = [90]
STOCHASTIC = [3, 5]
LR = 1e-2


class RegModel(object):
    def __init__(self, scope, two_heads=True):
        num_features = len(FEAT_IDX) + len(NEXT_IDX)   # 23 + 13 = 36
        lstm_size = num_features*2
        self.two_heads = two_heads

        self.input = tf.placeholder(tf.float32, [None, num_features], name="input")

        # weights
        W_fc_inp = _fc_weight_variable([num_features, lstm_size])
        b_fc_inp = _fc_bias_variable([lstm_size])
        W_fc_pts = _fc_weight_variable([lstm_size, len(PTS_IDX)])
        b_fc_pts = _fc_bias_variable([len(PTS_IDX)])

        dense = tf.nn.elu(tf.matmul(self.input, W_fc_inp) + b_fc_inp)
        dense_expanded = tf.expand_dims(dense, 0)

        self.lstm = BasicLSTM(lstm_size)
        self.step_size = tf.placeholder(tf.float32, [1], name="step_size")
        self.initial_lstm_state = tf.placeholder(tf.float32, [1, self.lstm.state_size],
                                                 name="initial_lstm_state")

        lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                          dense_expanded,
                                                          initial_state=self.initial_lstm_state,
                                                          sequence_length=self.step_size,
                                                          time_major=False,
                                                          scope=scope)
        lstm_outputs_reshaped = tf.reshape(lstm_outputs, [-1, lstm_size])

        self.points = tf.matmul(lstm_outputs_reshaped, W_fc_pts) + b_fc_pts

        self.weights = [W_fc_inp, b_fc_inp,
                        self.lstm.matrix, self.lstm.bias,
                        W_fc_pts, b_fc_pts]
        if self.two_heads:
            W_fc_val = _fc_weight_variable([lstm_size, len(LBL_IDX)])
            b_fc_val = _fc_bias_variable([len(LBL_IDX)])
            self.values = tf.matmul(lstm_outputs_reshaped, W_fc_val) + b_fc_val
            self.weights += [W_fc_val, b_fc_val]

        self.lstm_state_out = np.zeros([1, self.lstm.state_size], dtype=np.float32)

        self.prepare_loss()
        self.prepare_optimizer()

        compute_gradients = tf.gradients(self.loss, self.weights)
        self.apply_gradients = self.optimizer.apply_gradients(
            zip(compute_gradients, self.weights)
        )

    def reset(self):
        self.lstm_state_out.fill(.0)

    def prepare_loss(self, max_grad=1.):
        self.pts_gt = tf.placeholder(tf.float32, [None, len(PTS_IDX)], name="pts_gt")
        max_grad = tf.constant(max_grad, name='max_grad')

        pts_abs_err = tf.abs(self.pts_gt - self.points, name='pts_abs_err')

        lin = max_grad * (pts_abs_err - .5 * max_grad)
        quad = .5 * tf.square(pts_abs_err)

        pts_loss = tf.where(pts_abs_err < max_grad, quad, lin)

        if self.two_heads:
            self.val_gt = tf.placeholder(tf.float32, [None, len(LBL_IDX)], name="val_gt")
            val_abs_err = tf.reduce_sum(
                tf.abs(self.val_gt - self.values, name='val_abs_err'), axis=1)

            val_lin = max_grad * (val_abs_err - .5 * max_grad)
            val_quad = .5 * tf.square(val_abs_err)

            values_loss = tf.where(val_abs_err < max_grad, val_quad, val_lin)

            self.loss = tf.reduce_sum(pts_loss + .25 * values_loss)
        else:
            self.loss = tf.reduce_sum(pts_loss)

    def prepare_optimizer(self):
        self.learning_rate_input = tf.placeholder(tf.float32, [], name="lr_input")

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate_input
        )

    def infer(self, sess, state, step_size):
        #prev_lstm_state_out = self.lstm_state_out
        feeds = {self.input: state,
                 self.initial_lstm_state: self.lstm_state_out,
                 self.step_size: [step_size]}
        if self.two_heads:
            self.lstm_state_out, p_out, v_out = \
                sess.run([self.lstm_state, self.points, self.values],
                         feed_dict=feeds)
            #self.lstm_state_out = prev_lstm_state_out
            return p_out, v_out
        else:
            self.lstm_state_out, p_out = \
                sess.run([self.lstm_state, self.points],
                         feed_dict=feeds)
            #self.lstm_state_out = prev_lstm_state_out
            return p_out

    def train(self, sess, state, points, lstm_state, step_size, step, value=None):
        feeds = {self.input: state,
                 self.pts_gt: points,
                 self.initial_lstm_state: lstm_state,
                 self.step_size: [step_size],
                 self.learning_rate_input: LR}
        if self.two_heads:
            feeds[self.val_gt] = value
            self.lstm_state_out, loss, _ = \
                sess.run([self.lstm_state, self.loss, self.apply_gradients], feed_dict=feeds)
            print("Loss at step {}:".format(step), loss)
        else:
            self.lstm_state_out, loss, _ = \
                sess.run([self.lstm_state, self.loss, self.apply_gradients], feed_dict=feeds)
            print("Loss at step {}:".format(step), loss)


# weight initialization
def _fc_weight_variable(shape):
    input_channels = shape[0]
    d = np.sqrt(2.0 / input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


def _fc_bias_variable(shape):
    return tf.Variable(np.zeros(shape, dtype=np.float32))


def get_players_data(file_path_, verbose=False):  # ix -> human | values -> ndarray
    data = read_csv(file_path_, header=0, nrows=703)  # first 5 players
    data.fillna(value=0, inplace=True)

    Snell_1st = data.ix[0:149, :]
    Thompson_3rd = data.ix[193:383, :]
    Matthews_4th = data.ix[384:538, :]
    Towns_5th = data.ix[539:702, :]

    players = [Snell_1st, Thompson_3rd, Matthews_4th, Towns_5th]

    if verbose:
        for p in players:
            print('Features:', np.append(p.values[0], (p.values[1, 0:3])))
            print('Points:', p.values[1, -2])

    return players


def train(file_path_, epoch_num=1, test_num=5):
    players = get_players_data(file_path_)
    # for i in range(len(players)):

    model = RegModel('model', False)
    h_model = RegModel('h_model')
    models = [model]  # [model, h_model]
    data = players[2].values

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())

    length = len(data) - test_num
    for model in models:

        for i in range(epoch_num):
            cur_batch_sz = step = max(STOCHASTIC)
            rand, values = 0, None

            while step < length:  # 10 length
                features_mat = np.concatenate((data[step-cur_batch_sz:step, FEAT_IDX],
                                               data[1+step-cur_batch_sz:step+1, NEXT_IDX]), axis=1)
                points = data[1+step-cur_batch_sz:step+1, PTS_IDX]
                if model.two_heads:
                    values = data[1+step-cur_batch_sz:step+1, LBL_IDX]
                #print(points)

                start_lstm_state = model.lstm_state_out
                #predicts = model.infer(sess, features_mat, cur_batch_sz)
                #print(predicts)

                model.train(sess, features_mat, points, start_lstm_state,
                            cur_batch_sz, step, values)
                #print(model.lstm_state_out)
                rand = np.random.randint(STOCHASTIC[0], STOCHASTIC[1]+1)
                step += rand
            step -= rand
            '''
            cur_batch_sz = length - step
            if cur_batch_sz > 2:
                print('Additional')
                step += cur_batch_sz
                features_mat = np.concatenate((data[step - cur_batch_sz:step, FEAT_IDX],
                                               data[1 + step - cur_batch_sz:step + 1, NEXT_IDX]), axis=1)
                points = data[1 + step - cur_batch_sz:step + 1, PTS_IDX]
                model.train(sess, features_mat, points, model.lstm_state_out, cur_batch_sz, step)
            '''
            if test_num > 2:
                cur_batch_sz = test_num
                step += cur_batch_sz
                print('step', step+1, 'data_len', len(data))
                features_mat = np.concatenate((data[step - cur_batch_sz:step, FEAT_IDX],
                                               data[1 + step - cur_batch_sz:step + 1, NEXT_IDX]), axis=1)
                points = data[1 + step - cur_batch_sz:step + 1, PTS_IDX]
                print('Points at step {}:'.format(step), points)

                predicts = model.infer(sess, features_mat, cur_batch_sz)
                print('Predicts at step {}:'.format(step), predicts)
            model.reset()

    sess.close()


if __name__ == "__main__":
    file_path = '/home/dennis/Downloads/nba_player_games_6_52.csv'
    train(file_path, epoch_num=10)
