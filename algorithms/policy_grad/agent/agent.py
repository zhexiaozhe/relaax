from __future__ import print_function

import numpy as np
import tensorflow as tf
import time

import relaax.algorithm_base.agent_base
import relaax.common.protocol.socket_protocol

from . import network


class Agent(relaax.algorithm_base.agent_base.AgentBase):
    def __init__(self, config, parameter_server):
        self._config = config
        self._parameter_server = parameter_server
        self._local_network = network.make(config)

        self.global_t = 0           # counter for global steps between all agents
        self.local_t = 0            # steps count for current agent worker
        self.episode_reward = 0     # score accumulator for current episode (game)

        self.states = []            # auxiliary states accumulator through batch_size = 0..N
        self.actions = []           # auxiliary actions accumulator through batch_size = 0..N
        self.rewards = []           # auxiliary rewards accumulator through batch_size = 0..N

        self.episode_t = 0          # episode counter through batch_size = 0..M
        self.latency = 0            # latency accumulator for one episode loop

        if config.preprocess:
            if type(config.state_size) not in [list, tuple]:
                self._config.state_size = [config.state_size]
            self.prev_state = np.zeros(self._config.state_size)

        self._session = tf.Session()

        self._session.run(tf.variables_initializer(tf.global_variables()))

    def act(self, state):
        start = time.time()
        if self._config.preprocess:
            state = self._update_state(state)

        if state.ndim > 1:  # lambda layer
            state = state.flatten()

        if self.episode_t == self._config.batch_size:
            self._update_global()
            self.episode_t = 0

        if self.episode_t == 0:
            # copy weights from shared to local
            self._local_network.assign_values(self._session, self._parameter_server.get_values())

            self.states = []
            self.actions = []
            self.rewards = []

        # Run the policy network and get an action to take
        probs = self._local_network.run_policy(self._session, state)
        action = self.choose_action(probs)

        self.states.append(state)

        action_vec = np.zeros([self._config.action_size])  # one-hot vector to store taken action
        action_vec[action] = 1
        self.actions.append(action_vec)

        if (self.local_t % 100) == 0:   # can add by config
            print("TIMESTEP {}\nProbs: {}".format(self.local_t, probs))
            self.metrics().scalar('server latency', self.latency / 100)
            self.latency = 0

        self.latency += time.time() - start
        return action

    def reward_and_act(self, reward, state):
        if self._reward(reward):
            return self.act(state)
        return None

    def reward_and_reset(self, reward):
        if not self._reward(reward):
            return None

        print("Score =", self.episode_reward)
        score = self.episode_reward

        self.metrics().scalar('episode reward', self.episode_reward)

        self.episode_reward = 0
        self.episode_t = self._config.batch_size

        if self._config.preprocess:
            self.prev_state.fill(0)

        return score

    def _reward(self, reward):
        self.episode_reward += reward
        self.rewards.append(reward)

        self.local_t += 1
        self.episode_t += 1
        self.global_t = self._parameter_server.increment_global_t()

        return self.global_t < self._config.max_global_step

    def _update_global(self):
        feed_dict = {
            self._local_network.s: self.states,
            self._local_network.a: self.actions,
            self._local_network.advantage: self.discounted_reward(np.vstack(self.rewards)),
        }

        self._parameter_server.apply_gradients(
            self._session.run(self._local_network.grads, feed_dict=feed_dict)
        )

    @staticmethod
    def choose_action(pi_values):
        values = np.cumsum(pi_values)
        total = values[-1]
        r = np.random.rand() * total
        return np.searchsorted(values, r)

    def discounted_reward(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * self._config.GAMMA + r[t]
            discounted_r[t] = running_add
        # size the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_r = discounted_r.astype(np.float64)
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r) + 1e-20
        return discounted_r

    def metrics(self):
        return self._parameter_server.metrics()

    def _update_state(self, state):
        # Computes difference from the previous observation (motion-like process)
        self.prev_state = state - self.prev_state
        return self.prev_state


class DiscountedReward(object):
    def __init__(self, gamma_):
        # Define auxiliary variables & interfaces
        ph_rewards = tf.placeholder(tf.float32, [None, 1], name='ph_rewards')
        ph_results = tf.placeholder(tf.float32, [None, 1], name='ph_results')
        ph_length = tf.placeholder(tf.int32, name='ph_length')
        index = tf.constant(0, name='inner_loop_index')
        gamma = tf.constant(gamma_, name='discount_factor')     # 0.99
        running_add = tf.constant(0.0, name='running_add')
        self.inputs = [ph_rewards, ph_length, ph_results]

        # Define two necessary functions for tf.while_loop: condition & body
        loop_condition = lambda idx, r_add, results: tf.less(idx, ph_length)

        def loop_body(idx, r_add, results):
            i = ph_length - idx - 1     # replace further for convenience
            r_add = r_add * gamma + ph_rewards[i][0]
            results += tf.scatter_nd([[i, 0]], [r_add], shape=[ph_length, 1])
            return tf.add(idx, 1), r_add, results

        self.loop = tf.while_loop(loop_condition, loop_body,
                                  loop_vars=[index, running_add, ph_results])

    def __call__(self, sess, rewards, normalize=True):
        rewards = np.vstack(rewards).astype(np.float32)
        feeds = {self.inputs[0]: rewards,
                 self.inputs[1]: rewards.shape[0],
                 self.inputs[2]: np.zeros_like(rewards)}
        result = sess.run(self.loop, feed_dict=feeds)[-1]
        if normalize:
            result -= np.mean(result)
            result /= np.std(result) + 1e-20
        return result


class DiscountedRewardTFOP(object):
    def __init__(self, gamma_):
        # Define auxiliary variables & interfaces
        cur_reward = tf.placeholder(tf.float32, name='current_reward')
        running_add = tf.placeholder(tf.float32, name='running_add')
        gamma = tf.constant(gamma_, name='discount_factor')  # 0.99
        self.inputs = [cur_reward, running_add]

        self.compute = running_add * gamma + cur_reward

    def __call__(self, sess, rewards, normalize=True):
        rewards = np.vstack(rewards).astype(np.float32)
        result = np.zeros_like(rewards)
        print(result.shape)
        result[-1] = rewards[-1]
        for i in reversed(range(0, rewards.shape[0]-1)):
            feeds = {self.inputs[0]: rewards[i][0],
                     self.inputs[1]: result[i+1][0]}
            result[i] = sess.run(self.compute, feed_dict=feeds)
        if normalize:
            result -= np.mean(result)
            result /= np.std(result) + 1e-20
        return result


class DiscountedRewardM(object):
    # with fixed batch size for now --> to test
    def __init__(self, gamma_, size_):
        # Define auxiliary variables & interfaces
        powers = np.power(np.ones(size_) * gamma_, np.arange(size_))
        result = np.zeros((size_, size_))
        for i in range(size_):
            result[i, i:] = powers[:size_-i]
        operator = tf.Variable(
            result,
            # np.fliplr(np.triu(np.tile(np.vstack(np.power(np.ones(size_) * gamma_, np.arange(size_))), (1, size_)))),
            # np.tril(np.tile(np.power(np.ones(size_) * gamma_, np.arange(size_)), (size_, 1))),
            # np.triu(np.vstack((np.ones(size_), np.ones((size_-1, size_))*gamma_))),
            name='gamma_operator', dtype=np.float32
        )
        self.ph_rewards = tf.placeholder(tf.float32, [None, 1], name='ph_rewards')
        self.index = tf.placeholder(tf.int32, name='slice_index')
        self.compute = tf.matmul(operator[:self.index, :self.index], self.ph_rewards)

    def __call__(self, sess, rewards):
        return sess.run(self.compute, feed_dict={self.ph_rewards: rewards, self.index: rewards.shape[0]})
