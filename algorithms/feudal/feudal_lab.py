from __future__ import print_function
from __future__ import division

import tensorflow as tf
import threading
import signal
import time
import os

from config import cfg
from networks import GlobalManagerNetwork
from networks import GlobalWorkerNetwork
from train_thread_lab import TrainingThread


class Trainer:
    def __init__(self):
        self.global_t = 0
        self.perform_training = True
        self.training_threads = []

    def signal_handler(self):
        def handler(sig, frame):
            print('You pressed Ctrl+C!', sig, frame)
            self.perform_training = False
        return handler

    def train_function(self, thread_index, sess, summaries, summary_writer):
        training_thread = self.training_threads[thread_index]

        while self.perform_training:
            diff_global_t = training_thread.process(sess, self.global_t,
                                                    summaries, summary_writer)
            self.global_t += diff_global_t

    def run(self):
        global_manager = GlobalManagerNetwork()
        global_network = GlobalWorkerNetwork()

        for idx in range(cfg.threads_num):
            training_thread = TrainingThread(idx, global_network, global_manager)
            self.training_threads.append(training_thread)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(tf.global_variables_initializer())

        summaries = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            os.path.join('logs', time.strftime("%Y-%m-%d-%H-%M-%S")))
        summary_writer.add_graph(sess.graph)

        train_threads = []
        for idx in range(cfg.threads_num):
            train_threads.append(threading.Thread(
                target=self.train_function,
                args=(idx, sess, summaries, summary_writer)))

        signal.signal(signal.SIGINT, self.signal_handler())

        for t in train_threads:
            t.start()

        print('Press Ctrl+C to stop')
        signal.pause()

        for t in train_threads:
            t.join()


if __name__ == "__main__":
    model = Trainer()
    model.run()
