import numpy
import tensorflow


class ParameterServerBase(object):

    def on_run(self, session, graph, ops, feed_dict):
        fd = {}
        for key, value in feed_dict.iteritems():
            self.fill_feed_dict(fd, graph.__operations__[key], value)
        return session.run(
            map(lambda x: graph.__operations__[x], ops),
            feed_dict=fd
        )

    @classmethod
    def fill_feed_dict(cls, feed_dict, key, value):
        if isinstance(value, numpy.ndarray):
            assert isinstance(key, tensorflow.Tensor)
            feed_dict[key] = value
        elif isinstance(value, list):
            assert isinstance(key, list)
            assert len(key) == len(value)
            for k, v in zip(key, value):
                cls.fill_feed_dict(feed_dict, k, v)
        else:
            assert False


class ParameterServerImpl(ParameterServerBase):
    def __init__(self, graph):
        self.graph = graph
        self.session = tensorflow.Session()
        self.session.run(tensorflow.global_variables_initializer())

    def run(self, ops, feed_dict):
        return self.on_run(self.session, self.graph, ops, feed_dict)
