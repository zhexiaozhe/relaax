from __future__ import print_function

import argparse

from utils import TextLoader
from relaax.algorithm_lib.lstm import CustomBasicLSTMCell


def main():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data',
                        help='data directory containing input.txt')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to store tensorboard logs')

    parser.add_argument('--cell_size', type=int, default=256,
                        help='size of LSTM cell')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=10,
                        help='LSTM sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')

    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    args = parser.parse_args()
    train(args)


def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    cell = CustomBasicLSTMCell(args.cell_size)  # 256
    print(cell)

if __name__ == '__main__':
    main()
