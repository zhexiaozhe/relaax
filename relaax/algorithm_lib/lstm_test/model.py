from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.contrib import legacy_seq2seq
from relaax.algorithm_lib.lstm import CustomBasicLSTMCell
from relaax.algorithm_lib.lstm import DilatedBasicLSTMCell
from relaax.algorithm_lib.lstm import DilateBasicLSTMCell
from relaax.algorithm_lib.lstm import DilatedLSTMCell


class Model:
    def __init__(self, args):
        self.args = args

        if args.model == 'basic_lstm':
            self.cell = CustomBasicLSTMCell(args.cell_size)
        elif args.model == 'dilated_lstm':
            self.cell = DilatedBasicLSTMCell(args.cell_size, cores=10)
        elif args.model == 'dilate_lstm':
            self.cell = DilateBasicLSTMCell(args.cell_size, cores=10)
        elif args.model == 'dilated':
            self.cell = DilatedLSTMCell(args.cell_size, cores=10)
        else:
            raise Exception('Unknown network type: {}'.format(args.model))

        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])

        self.initial_lstm_state = tf.placeholder(tf.float32, [1, self.cell.state_size])

        embedding = tf.get_variable("embedding", [args.vocab_size, args.cell_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        print('Input shape:', inputs.get_shape())
        print('Sequence length:', args.seq_length)

        lstm_outputs, self.lstm_state = \
            tf.nn.dynamic_rnn(self.cell,
                              inputs,
                              initial_state=self.initial_lstm_state,
                              sequence_length=[args.seq_length],
                              time_major=False)

        with tf.variable_scope('lstm'):
            softmax_w = tf.get_variable("softmax_w",
                                        [self.cell.output_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

        lstm_outputs = tf.reshape(lstm_outputs, [-1, self.cell.output_size])
        self.logits = tf.matmul(lstm_outputs, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        loss = legacy_seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([args.batch_size * args.seq_length])])
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # tensorboard summaries
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

        self.lstm_state_out = None
        self.reset_state()

    def reset_state(self):
        self.lstm_state_out =\
            np.zeros([1, self.initial_lstm_state.get_shape().as_list()[-1]])

    def train_model(self, sess, feeds, summaries, write):
        if write:
            train_loss, self.lstm_state_out, _, summ =\
                sess.run([self.cost, self.lstm_state, self.train_op, summaries],
                         feed_dict=feeds)
            return train_loss, summ
        train_loss, self.lstm_state_out, _ = \
            sess.run([self.cost, self.lstm_state, self.train_op],
                     feed_dict=feeds)
        return train_loss, None
