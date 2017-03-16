import tensorflow as tf
import numpy as np

import relaax.algorithm_base.bridge_base
import relaax.algorithm_base.parameter_server_base
import relaax.server.common.saver.tensorflow_checkpoint

from . import network


class ParameterServer(relaax.algorithm_base.parameter_server_base.ParameterServerBase):
    def __init__(self, config, saver_factory, metrics):
        self._network = network.make(config)

        self._session = tf.Session()

        self._saver = saver_factory(
            relaax.server.common.saver.tensorflow_checkpoint.TensorflowCheckpoint(
                self._session
            )
        )

        self._session.run(tf.variables_initializer(tf.global_variables()))

        if config.use_filter:
            init = np.zeros(config.state_size)
            global_t = self.global_t()
            self._filter_state = [global_t, init, init]

        self._bridge = _Bridge(config, metrics, self._network, self._session, self._filter_state)

    def close(self):
        self._session.close()

    def restore_latest_checkpoint(self):
        checkpoint_ids = self._saver.checkpoint_ids()
        if len(checkpoint_ids) > 0:
            self._saver.restore_checkpoint(max(checkpoint_ids))

    def save_checkpoint(self):
        self._saver.save_checkpoint(self.global_t())

    def global_t(self):
        return self._session.run(self._network.global_t)

    def bridge(self):
        return self._bridge


class _Bridge(relaax.algorithm_base.bridge_base.BridgeBase):
    def __init__(self, config, metrics, network, session, filter_state):
        self._config = config
        self._metrics = metrics
        self._network = network
        self._session = session
        self._filter_state = filter_state

    def increment_global_t(self):
        return self._session.run(self._network.increment_global_t)

    def apply_gradients(self, gradients):
        feed_dict = {p: v for p, v in zip(self._network.gradients, gradients)}
        feed_dict[self._network.learning_rate_input] = self._anneal_learning_rate(
            self._session.run(self._network.global_t)
        )
        self._session.run(self._network.apply_gradients, feed_dict=feed_dict)

    def get_values(self):
        return self._session.run(self._network.values)

    def get_filter_state(self):
        return self._filter_state

    def update_filter_state(self, diff):
        self._filter_state[0] = self._session.run(self._network.global_t)
        # diff[0]: agent_t, diff[1]: mean_diff, diff[2]: std_diff
        self._filter_state[1] =\
            (self._filter_state[1]*self._filter_state[0] + diff[1]) / (self._filter_state[0] + diff[0])
        self._filter_state[2] += diff[2]

    def metrics(self):
        return self._metrics

    def _anneal_learning_rate(self, global_time_step):
        factor = (self._config.max_global_step - global_time_step) / self._config.max_global_step
        learning_rate = self._config.INITIAL_LEARNING_RATE * factor
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate
