import tensorflow as tf

from relaax.common.algorithms import subgraph


INPUT_SIZE = 2   # state_size
OUTPUT_SIZE = 1  # action_size
HIDDEN_1 = []    # hidden_sizes
HIDDEN_2 = [2]   # hidden_sizes


class DiffLoss(subgraph.Subgraph):
    def build_graph(self, target, network):
        return tf.reduce_sum(tf.abs(target.node - network.node), axis=[1])
