from __future__ import print_function

import sys
sys.path.append('../../server')

import argparse
import logging
import time
import signal
import ruamel.yaml

import algorithms.a3c.bridge
import algorithms.a3c.master
import algorithms.a3c.params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-level',
        type=str,
        default='WARNING',
        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
        help='logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)'
    )
    parser.add_argument('--params', type=str, default=None, help='parameters YAML file')
    parser.add_argument('--bind', type=str, default=None, help='address to serve (host:port)')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='TensorFlow checkpoint directory')
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s',
        level=log_level
    )

    with open(args.params, 'r') as f:
        yaml = ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)

    master = algorithms.a3c.master.Master(algorithms.a3c.params.Params(yaml))

    if master.load_checkpoint(args.checkpoint_dir):
        print('checkpoint loaded from %s' % args.checkpoint_dir)

    def stop_server(_1, _2):
        master.save_checkpoint(args.checkpoint_dir)
        print('checkpoint saved to %s' % args.checkpoint_dir)
        master.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_server)

    # keep the server or else GC will stop it
    server = algorithms.a3c.bridge.start_server(args.bind, master.service())

    last_global_t = None
    while True:
        time.sleep(1)
        global_t = master.global_t()
        if global_t != last_global_t:
            last_global_t = global_t
            print("global_t is %d" % global_t)


if __name__ == '__main__':
    main()