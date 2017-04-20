class A3CTrainingThread(object):
    def __init__(self,
                 thread_index,
                 global_network):
        self.thread_index = thread_index

    def process(self, sess, global_t, summaries, summary_writer):
        return self.thread_index
