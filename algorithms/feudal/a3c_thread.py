import tensorflow as tf
import numpy as np
import gym

from networks import A3CLocalNetwork
from config import cfg


class A3CTrainingThread(object):
    def __init__(self,
                 thread_index,
                 global_network):
        self.thread_index = thread_index

        self.initial_learning_rate = cfg.learning_rate
        self.max_global_time_step = cfg.MAX_TIME_STEP

        self.local_network = A3CLocalNetwork()

        self.compute_gradients = tf.gradients(self.local_network.total_loss,
                                              self.local_network.weights)

        self.apply_gradients = global_network.optimizer.apply_gradients(
            zip(self.compute_gradients, global_network.weights)
        )

        self.sync = self.local_network.sync_from(global_network)

        self.env = gym.make(cfg.env_name)
        self.env.seed(113 * thread_index)

        self.local_t = 0
        self.episode_reward = 0

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate *\
                        (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    @staticmethod
    def choose_action(pi_probs):
        return np.random.choice(pi_probs, p=pi_probs)

    def process(self, sess, global_t, summaries, summary_writer):
        return self.thread_index
