import tensorflow as tf
import numpy as np
import random


def policy_gradient(num_layers=0):
    with tf.variable_scope("policy"):
        if num_layers:
            params = tf.get_variable("input_hidden", [2, 2])
            layer = tf.get_variable("hiddenn_output", [2, 1])
        else:
            params = tf.get_variable("policy_parameters", [2, 1])

        state = tf.placeholder("float", [None, 2])
        actions = tf.placeholder("float", [None, 1])
        advantages = tf.placeholder("float", [None, 1])

        linear = tf.matmul(state, params)
        if num_layers:
            linear = tf.nn.relu(linear)
            linear = tf.matmul(linear, layer)
        probs = tf.nn.sigmoid(linear)

        good_probabilities = tf.square(actions - probs)
        #good_probabilities = tf.reduce_sum(probs, reduction_indices=[1])
        log_like = tf.log(good_probabilities)
        eligibility = log_like * advantages
        loss = -tf.reduce_sum(eligibility)

        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return probs, state, actions, advantages, optimizer, log_like


def step(action, old_observation):
    state = np.random.randint(2, size=2)
    reward = 0

    xor = np.remainder(np.sum(old_observation), 2)
    if action == xor:
        reward += 1

    return state, reward


def run_episode(policy_grad, sess):
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer, log_like = policy_grad
    observation = np.random.randint(2, size=2)
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []

    for _ in xrange(100):
        # calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_calculated, feed_dict={pl_state: obs_vector})
        action = 0 if random.uniform(0, 1) > probs[0] else 1
        # record the transition
        states.append(observation)
        actions.append([action])
        # take the action in the environment
        old_observation = observation
        observation, reward = step(action, old_observation)
        transitions.append((old_observation, action, reward))
        totalreward += reward

    for index, trans in enumerate(transitions):
        # calculate discounted monte-carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in xrange(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.97

        # advantage: how much better was this action than normal
        advantages.append(future_reward)

    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states,
                                      pl_advantages: advantages_vector,
                                      pl_actions: actions})
    return totalreward


policy_grad = policy_gradient()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in xrange(1000):
    reward = run_episode(policy_grad, sess)
    print 'Episode:', i, '| Reward:', reward
sess.close()

policy_grad = policy_gradient(num_layers=1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in xrange(1000):
    reward = run_episode(policy_grad, sess)
    print 'Episode:', i, '| Reward:', reward
sess.close()
