import tensorflow as tf

from relaax.algorithm_lib.lstm import CustomBasicLSTMCell


class Model:
    def __init__(self, args):
        self.args = args

        if args.model == 'basic_lstm':
            cell = CustomBasicLSTMCell(args.cell_size)
        else:
            raise Exception('Unknown network type: {}'.format(args.model))

        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])

        self.initial_lstm_state = tf.placeholder(tf.float32, [1, cell.state_size])

        embedding = tf.get_variable("embedding", [args.vocab_size, args.cell_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        print(inputs.get_shape())
        print(args.seq_length)

        self.lstm_outputs, self.lstm_state = \
            tf.nn.dynamic_rnn(cell,
                              inputs,
                              initial_state=self.initial_lstm_state,
                              sequence_length=[args.seq_length],
                              time_major=False)
