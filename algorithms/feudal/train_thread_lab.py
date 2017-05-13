from __future__ import print_function
from __future__ import division
import psutil as psu

import tensorflow as tf
import numpy as np


from networks import LocalWorkerNetwork
from networks import LocalManagerNetwork
from config import cfg
from utils import RingBuffer2D
from lab.environment import Lab


class TrainingThread(object):
    def __init__(self,
                 thread_index,
                 global_network,
                 global_manager):
        self.thread_index = thread_index
        self.goal_buffer = RingBuffer2D(element_size=cfg.d,
                                        buffer_size=cfg.c * 2)
        self.st_buffer = RingBuffer2D(element_size=cfg.d,
                                      buffer_size=cfg.c * 2)
        self.first = cfg.c
        self.states = []
        self.cur_c = None
        self.eps = 1e-12

        self.half_of_initial_learning_rate = cfg.learning_rate / 2
        self.anneal_step_limit = cfg.anneal_step_limit

        self.manager_network = LocalManagerNetwork(thread_index)
        self.local_network = LocalWorkerNetwork(thread_index)

        compute_gradients = tf.gradients(self.local_network.total_loss,
                                         self.local_network.weights)
        compute_manager = tf.gradients(self.manager_network.lossM,
                                       self.manager_network.weights)

        self.apply_gradients = global_network.optimizer.apply_gradients(
            zip(compute_gradients, global_network.weights)
        )
        self.apply_manager = global_manager.optimizer.apply_gradients(
            zip(compute_manager, global_manager.weights)
        )

        self.sync = self.local_network.sync_from(global_network)
        self.sync_manager = self.manager_network.sync_from(global_manager)

        self.lrW = global_network.learning_rate_input
        self.lrM = global_manager.learning_rate_input

        self.env = Lab(cfg.level)
        self.env.reset()
        self.state = self.env.state()

        self.local_t = 0
        self.episode_reward = 0

        # summary for tensorboard
        if thread_index == 0:
            self.score_input = tf.placeholder(tf.float32, [], name="episode_reward")
            tf.summary.scalar('episode_reward', self.score_input)
            self.mem_input = tf.placeholder(tf.float32, [], name="memory_usage")
            tf.summary.scalar('memory_usage', self.mem_input)

    def _annealing_lr_by_half(self, global_time_step):
        if global_time_step > self.anneal_step_limit:
            return self.half_of_initial_learning_rate
        learning_rate =\
            self.half_of_initial_learning_rate + \
            self.half_of_initial_learning_rate * \
            (self.anneal_step_limit - global_time_step) / self.anneal_step_limit
        return learning_rate

    @staticmethod
    def choose_action(pi_probs):
        return np.random.choice(len(pi_probs), p=pi_probs)

    def process(self, sess, global_t, summaries, summary_writer):
        # sync local manger's weights with global
        sess.run(self.sync_manager)

        # update the last half of accumulated data
        if self.first == 0:
            zt_batch = sess.run(self.local_network.perception,
                                {self.local_network.s: self.states})
            # print('zt_batch', zt_batch.shape)
            goals_batch, st_batch = \
                self.manager_network.run_goal_and_st(sess, zt_batch)
            # second half is used in intrinsic reward calculation
            self.goal_buffer.replace_second_half(goals_batch)
            self.st_buffer.replace_second_half(st_batch)

        self.states, actions, rewards, values = [], [], [], []
        goals, m_values, states_t, rewards_i, zt_inp = [], [], [], [], []
        terminal_end = False
        self.cur_c = 0

        # copy weights from shared to local
        sess.run(self.sync)
        sess.run(self.manager_network.lstm.reset_timestep)

        start_local_t = self.local_t
        start_lstm_state = self.local_network.lstm_state_out
        manager_lstm_state = self.manager_network.lstm_state_out

        for i in range(cfg.LOCAL_T_MAX + self.first):
            z_t = sess.run(self.local_network.perception,
                           {self.local_network.s: [self.state]})
            zt_inp.append(z_t[0])
            goal, v_t, s_t = self.manager_network.run_goal_value_st(sess, z_t)

            self.goal_buffer.extend(goal)
            goal = self.goal_buffer.get_sum()
            goals.append(goal)

            m_values.append(v_t)
            self.st_buffer.extend(s_t)

            pi_, value_ = \
                self.local_network.run_policy_and_value(sess, self.state, goal)
            action = self.choose_action(pi_)

            self.states.append(self.state)
            actions.append(action)
            values.append(value_)

            if (self.thread_index == 0) and (self.local_t % 100) == 0:
                print("TIMESTEP", self.local_t)
                print("pi=", pi_)
                print(" V=", v_t)  # value_

            # act
            reward, terminal = self.env.act(action)
            self.local_t += 1

            # calc internal rewards produces by manager
            # depends on st and goals buffers within the horizon
            if self.cur_c < cfg.c:
                self.cur_c += 1
            if not terminal:
                self.state = self.env.state()
                z_t = sess.run(self.local_network.perception,
                               {self.local_network.s: [self.state]})
                s_t = self.manager_network.run_st(sess, z_t)

                reward_i = []
                for k in range(1, self.cur_c + 1):
                    cur_st = s_t - self.st_buffer.data[-k, :]
                    cur_st_norm =\
                        np.maximum(np.linalg.norm(cur_st, axis=1), self.eps)
                    st_normed = (cur_st.transpose() / cur_st_norm).transpose()

                    cur_goal = self.goal_buffer.data[-k, :]
                    cur_goal_norm =\
                        np.maximum(np.linalg.norm(cur_goal, axis=1), self.eps)
                    goals_normed = cur_goal.transpose() / cur_goal_norm

                    cosine = np.dot(st_normed, goals_normed)
                    reward_i.append(cosine)

                reward_i = sum(reward_i) / self.cur_c
            else:
                reward_i = 0
            # reward + alpha * reward_i -> alpha in [0,1] >> try 1 or 0.8

            self.episode_reward += reward
            # append reward without clipping
            rewards.append(reward)  # rewards.append(np.clip(reward, -1, 1))
            # append external and intrinsic reward
            rewards_i.append(cfg.alpha * reward_i)

            if terminal:
                terminal_end = True
                self.first = cfg.c
                print("Score:", self.episode_reward)

                if self.thread_index == 0:
                    summary_str = sess.run(summaries,
                                           feed_dict={self.score_input: self.episode_reward,
                                                      self.mem_input: psu.virtual_memory()[2]})
                    summary_writer.add_summary(summary_str, global_t)

                self.episode_reward = 0
                self.env.reset()
                self.state = self.env.state()

                self.st_buffer.reset()
                self.goal_buffer.reset()
                self.local_network.reset_state()
                self.manager_network.reset_state()
                break

        diff_local_t = self.local_t - start_local_t

        R = Ri = 0.0
        if not terminal_end:
            R, z_t = self.local_network.run_value_and_zt(sess, self.state)
            Ri = self.manager_network.run_value(sess, z_t)
            self.first = 0

        if len(self.states) > cfg.c:
            self.states = self.states[cfg.c:]
            actions = actions[cfg.c:]
            rewards = rewards[cfg.c:]
            rewards_i = rewards_i[cfg.c:]
            values = values[cfg.c:]
            m_values = m_values[cfg.c:]
            zt_inp = zt_inp[cfg.c:]
            goals = goals[cfg.c:]

        states = self.states[:]

        actions.reverse()
        rewards.reverse()
        rewards_i.reverse()
        values.reverse()
        m_values.reverse()

        batch_a = []
        batch_tdM = []
        batch_tdW = []
        batch_R = []

        # compute and accumulate gradients
        for (ai, ri, riM, Vi, ViM) in zip(actions, rewards, rewards_i,
                                          values, m_values):
            R = ri + cfg.wGAMMA * R
            Ri = riM + cfg.mGAMMA * Ri
            tdM = R - ViM
            tdW = R + Ri - Vi
            a = np.zeros([cfg.action_size])
            a[ai] = 1

            batch_a.append(a)
            batch_tdM.append(tdM)
            batch_tdW.append(tdW)
            batch_R.append(R + Ri)

        batch_a.reverse()
        batch_tdM.reverse()
        batch_tdW.reverse()
        batch_R.reverse()

        learning_rate = self._annealing_lr_by_half(global_t)
        st_diff = self.st_buffer.get_diff(part=diff_local_t)

        sess.run([self.apply_manager, self.apply_gradients],
                 feed_dict={
                     self.manager_network.ph_perception: zt_inp,
                     self.manager_network.stc_minus_st: st_diff,
                     self.manager_network.tdM: batch_tdM,
                     self.manager_network.initial_lstm_state: manager_lstm_state,
                     self.lrM: learning_rate,
                     self.manager_network.step_size: [diff_local_t],

                     self.local_network.s: states,
                     self.local_network.ph_goal: goals,
                     self.local_network.a: batch_a,
                     self.local_network.td: batch_tdW,
                     self.local_network.r: batch_R,
                     self.local_network.initial_lstm_state: start_lstm_state,
                     self.lrW: learning_rate,
                     self.local_network.step_size: [diff_local_t]})

        # return advanced local step size
        return diff_local_t
