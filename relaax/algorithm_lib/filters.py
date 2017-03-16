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


# rs = RunningStatTF().make_push(state_placeholder)
# zf = ZFilterTF(state_placeholder, rs, push_x=True)


class ZFilterTF(object):
    """ y = (x-mean)/std using running estimates of mean, std """
    def __init__(self, x, rs, push_x=True, demean=True, destd=True, clip=None):
        xx = x
        if push_x:
            with tf.control_dependencies(rs.push):
                xx = tf.identity(xx)
        if demean:
            xx = tf.subtract(xx, rs.mean)
        if destd:
            xx = tf.divide(xx, rs.std)
        if clip is not None:
            xx = tf.clip_by_value(xx, -clip, clip)
        self.result = xx


class RunningStatTF(object):
    def make_assign(self, n, mean, var):
        assert self.assign is None

        self._make_state(mean.get_shape(), mean.dtype)

        self.assign = tf.group(tf.assign(self.n, n),
                               tf.assign(self.mean, mean),
                               tf.assign(self._s, var * (n-1)))

        return self

    def make_push(self, observation):
        assert self.push is None

        self._make_state(observation.get_shape(), observation.dtype)

        new_n = tf.add(self.n, 1)
        new_m = tf.add(self.mean, (observation - self.mean) / new_n)
        new_s = tf.add(self._s, (observation - self.mean) * (observation - new_m))
        self.push = tf.group(tf.assign(self.n, new_n),
                             tf.assign(self.mean, new_m),
                             tf.assign(self._s, new_s))

        return self

    def make_push_block(self, nb, meanb, varb):
        assert self.push_block is None

        self._make_state(meanb.get_shape(), meanb.dtype)

        new_n = tf.add(self.n, nb)

        delta = tf.subtract(meanb, self.mean)
        new_mean = tf.add(self.mean, delta * nb / new_n)

        m_b = varb * (nb - 1)
        new_s = tf.add(self._s, m_b + tf.square(delta) * tf.multiply(self.n, nb) / new_n)

        self.push_block = tf.group(tf.assign(self.n, new_n),
                                   tf.assign(self.mean, new_mean),
                                   tf.assign(self._s, new_s))

        return self

    def _make_state(self, shape, dtype):
        if self.n is None:
            # TODO: mark as nontrainable
            self.n = tf.Variable(0, tf.int64, name='n')
            self.mean = tf.Variable(tf.zeros(shape), dtype, name='mean')
            self._s = tf.Variable(tf.zeros(shape), dtype, name='s')

            self.var = tf.cond(self.n > 1, tf.div(self._s, tf.subtract(self.n, 1)), tf.square(self.mean))
            self.std = tf.sqrt(self.var)
        else:
            assert shape == self.mean.get_shape()
            assert dtype == self.mean.dtype
