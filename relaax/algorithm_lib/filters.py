import numpy as np
import tensorflow as tf


class ZFilter(object):
    """ y = (x-mean)/std
    using running estimates of mean, std """
    def __init__(self, shape_or_object, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = shape_or_object
        if type(shape_or_object) is tuple:
            self.rs = RunningStat(shape_or_object)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x -= self.rs.mean
        if self.destd:
            x /= (self.rs.std+1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    @staticmethod
    def output_shape(input_space):
        return input_space.shape


class RunningStat(object):
    # http://www.johndcook.com/blog/standard_deviation/
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM)/self._n
            self._S[...] = self._S + (x - oldM)*(x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S/(self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class RunningStatExt(RunningStat):
    def __init__(self, shape):
        super(RunningStatExt, self).__init__(shape)
        self._inN = None
        self._inM = None
        self._inS = None

    def set(self, n, M, S):
        self._n, self._inN = n, n
        assert M.shape == self._M.shape
        self._M = M.copy()
        self._S = S.copy()
        self._inM = M.copy()
        self._inS = S.copy()

    def get_diff(self):
        diffM = self._M*self._n - self._inM*self._inN
        diffS = self._S - self._inS
        return diffM, diffS

    @property
    def old_n(self):
        return self._inN


class ZFilterTF(object):
    """ y = (x-mean)/std using running estimates of mean, std """
    def __init__(self, observation, nb, meanb, varb, demean=True, destd=True, clip=None, scope=None):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStatTF(observation, nb, meanb, varb, scope)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x -= self.rs.mean
        if self.destd:
            x /= (self.rs.std+1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    @staticmethod
    def output_shape(input_space):
        return input_space.shape


class RunningStatTF(object):
    def __init__(self, observation, nb, meanb, varb, scope=None):
        if scope is not None:
            with tf.get_default_graph().name_scope(scope):
                self.create_graph(observation, nb, meanb, varb)
        else:
            self.create_graph(observation, nb, meanb, varb)

    def create_graph(self, observation, nb, meanb, varb):
        shape = observation.get_shape()
        dtype = observation.dtype

        self.n = tf.Variable(0, tf.int64, name='n')
        self.mean = tf.Variable(tf.zeros(shape), dtype, name='mean')
        self._s = tf.Variable(tf.zeros(shape), dtype, name='s')

        self.make_push(observation)

        self.var = tf.cond(self.n > 1, tf.div(self._s, tf.subtract(self.n, 1)), tf.square(self.mean))
        self.std = tf.sqrt(self.var)

        self.make_push_block(nb, meanb, varb)

    def make_push(self, observation):
        new_n = tf.add(self.n, 1)
        new_m = tf.add(self.mean, (observation - self.mean) / new_n)
        new_s = tf.add(self._s, (observation - self.mean) * (observation - new_m))
        self.push = tf.group(tf.assign(self.n, new_n),
                             tf.assign(self.mean, new_m),
                             tf.assign(self._s, new_s))

    def make_push_block(self, nb, meanb, varb):
        new_n = tf.add(self.n, nb)

        delta = tf.subtract(self.mean, meanb)
        self.mean_block = tf.add(self.mean, delta * nb / new_n)

        m_b = varb * (nb - 1)
        M2 = self.var + m_b + delta ** 2 * tf.multiply(self.n, nb) / new_n
        self.std_block = tf.sqrt(M2 / (new_n - 1))

        self.push_blcok = tf.group(tf.assign(self.n, new_n),
                                   tf.assign(self.mean, self.mean_block),
                                   tf.assign(self.std, self.std_block))
