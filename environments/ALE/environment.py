from __future__ import print_function

import logging
import os
import random
from time import time, sleep

from relaax.client import rlx_client

from . import game_process


def run(rlx_server_url, rom, display, seed):
    n_game = 0
    game = game_process.GameProcessFactory(rom, display).new_env(_seed(seed))

    while True:
        try:
            client = rlx_client.Client(rlx_server_url)
            try:
                action = client.init(game.state())
                client_latency, acts = 0, 0
                while True:
                    reward, reset = game.act(action)
                    if reset:
                        episode_score = client.reset(reward)
                        n_game += 1
                        print('Score at game', n_game, '=', episode_score)
                        game.reset()
                        action = _send(client, None, game.state())
                        if acts:
                            client.metrics().scalar('client latency', client_latency / acts)
                            client_latency, acts = 0, 0
                    else:
                        start = time()
                        action = _send(client, reward, game.state())
                        client_latency += time() - start
                        acts += 1
            finally:
                client.disconnect()
        except rlx_client.Failure as e:
            _warning('{} : {}'.format(rlx_server_url, e.message))
            delay = random.randint(1, 10)
            _info('waiting for %ds...', delay)
            sleep(delay)


def _send(client, reward, state):
    action = client.send(reward, state)
    return action


def _seed(value):
    if value is None:
        return random.randrange(1000000)
    return value


def _info(message, *args):
    logging.info('%d:' + message, os.getpid(), *args)


def _warning(message, *args):
    logging.warning('%d:' + message, os.getpid(), *args)