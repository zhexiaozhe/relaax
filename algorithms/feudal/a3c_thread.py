import tensorflow as tf
import numpy as np
import gym
from scipy.misc import imresize

from networks import A3CLocalNetwork
from config import cfg


class A3CTrainingThread(object):
    def __init__(self,
                 thread_index,
                 global_network):
        self.thread_index = thread_index

        self.initial_learning_rate = cfg.learning_rate
        self.max_global_time_step = cfg.MAX_TIME_STEP

        self.local_network = A3CLocalNetwork(thread_index)

        compute_gradients = tf.gradients(self.local_network.total_loss,
                                         self.local_network.weights)

        self.apply_gradients = global_network.optimizer.apply_gradients(
            zip(compute_gradients, global_network.weights)
        )

        self.sync = self.local_network.sync_from(global_network)

        self.env = gym.make(cfg.env_name)
        self.env.seed(113 * thread_index)
        self.env_state = self.env.reset()

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
        return np.random.choice(len(pi_probs), p=pi_probs)

    def process(self, sess, global_t, summaries, summary_writer):
        states, actions, rewards, values = [], [], [], []

        # copy weights from shared to local
        sess.run(self.sync)

        start_local_t = self.local_t
        start_lstm_state = self.local_network.lstm_state_out

        for i in range(cfg.LOCAL_T_MAX):
            pi_, value_ = self.local_network.run_policy_and_value(sess, _process_state(self.env_state))
            action = self.choose_action(pi_)

        return self.thread_index


def _process_state(screen):
    resized_screen = imresize(screen, (110, 84))
    state = resized_screen[18:102, :]

    state = state.astype(np.float32)
    state *= (1.0 / 255.0)
    return state
