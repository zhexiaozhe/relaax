from __future__ import print_function
from __future__ import absolute_import

import argparse
from data_loader import DataLoader


def train(epoch_num=1):
    print('Training in {} epochs'.format(epoch_num))


def read_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--path", required=True, type=str,
                        help="Path to the stored data as *.csv file")
    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    players_data = DataLoader(args.path)
    train(epoch_num=10)
