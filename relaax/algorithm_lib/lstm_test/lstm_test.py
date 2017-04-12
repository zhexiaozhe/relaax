from __future__ import print_function

import argparse
import os
from six.moves import cPickle
import time
import tensorflow as tf

from utils import TextLoader
from model import Model


def main(model_type):
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data',
                        help='data directory containing input.txt')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')

    parser.add_argument('--cell_size', type=int, default=256,
                        help='size of LSTM cell')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=10,
                        help='LSTM sequence length')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')

    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients by this value')

    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--restore', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                                'config.pkl'        : configuration;
                                'chars_vocab.pkl'   : vocabulary definitions;
                                'checkpoint'        : paths to model file(s) (created by tf).
                                                      Note: this file contains absolute paths, be careful when moving files around;
                                'model.ckpt-*'      : file(s) with model definition (created by tf)
                            """)
    args = parser.parse_args()
    args.model = model_type
    train(args)


def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    model = Model(args)

    with tf.Session() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
                os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.restore is not None:
            ckpt = tf.train.get_checkpoint_state(args.restore)
            saver.restore(sess, ckpt.model_checkpoint_path)

        for e in range(args.num_epochs):
            model.reset_state()     # ? -> should be tested
            for b in range(data_loader.num_batches):
                start = time.time()

if __name__ == '__main__':
    main('basic_lstm')
