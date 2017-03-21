#!/usr/bin/env python

import operator
import numpy as np
import tensorflow as tf
import unittest

from relaax.algorithm_lib.filters import RunningStatTF


class TestRunningStatTF(unittest.TestCase):

    SAMPLES = [
        [ 6, 12],
        [12, 24],
        [18, 36],
        [24, 48],
        [30, 60],
        [36, 72]
    ]

    MEANS = [
        [ 0,  0],
        [ 6, 12],
        [ 9, 18],
        [12, 24],
        [15, 30],
        [18, 36],
        [21, 42]
    ]

    VARIANCES = [
        [  0,   0],
        [  0,   0],
        [ 18,  72],
        [ 36, 144],
        [ 60, 240],
        [ 90, 360],
        [126, 504]
    ]

    def setUp(self):
        tf.reset_default_graph()
        self.session = tf.Session()

    def tearDown(self):
        self.session.close()
        tf.reset_default_graph()

    def test_assign(self):
        self.check_assign(tf.float32)
        self.check_assign(tf.float64)

    def test_push(self):
        self.check_push(tf.float32)
        self.check_push(tf.float64)

    def test_push_block(self):
        self.check_push_block(tf.float32)
        self.check_push_block(tf.float64)

    def test_push_push_block(self):
        self.check_push_push_block(tf.float32)
        self.check_push_push_block(tf.float64)

    def assert_rs_shape(self, rs, dtype, shape):
        self.assertEquals(rs.n       .dtype.base_dtype, tf.int64)
        self.assertEquals(rs.mean    .dtype.base_dtype, dtype   )
        self.assertEquals(rs.variance.dtype           , dtype   )
        self.assertEquals(rs.mean    .get_shape(), shape)
        self.assertEquals(rs.variance.get_shape(), shape)

    def assert_rs(self, rs, n, mean, variance):
        n_, mean_, var_ = self.session.run([rs.n, rs.mean, rs.variance])
        self.assertEquals(n_            , n       )
        self.assertEquals(mean_.tolist(), mean    )
        self.assertEquals(var_ .tolist(), variance)

    def assert_rs_n(self, rs, n):
        self.assert_rs(rs, n, self.MEANS[n], self.VARIANCES[n])

    def push(self, push, x, n):
        self.session.run([push], feed_dict={x: self.SAMPLES[n]})

    def push_block(self, push_block, n, mean, variance, n1, n2):
        self.session.run([push_block], feed_dict={
            n       : n2 - n1,
            mean    : self.mean(n1, n2),
            variance: self.variance(n1, n2)
        })


    def mean(self, n1, n2):
        return map(lambda x: x / (n2 - n1), reduce(
            lambda a, b: map(operator.add, a, b),
            self.SAMPLES[n1:n2],
            [0] * len(self.SAMPLES[0])
        ))

    def variance(self, n1, n2):
        if n1 >= n2 - 1:
            return [0] * len(self.SAMPLES[0])
        mean = self.mean(n1, n2)
        return map(lambda x: x / (n2 - n1 - 1), reduce(
            lambda a, b: map(operator.add, a, map(lambda x: (x[0] - x[1])**2, zip(b, mean))),
            self.SAMPLES[n1:n2],
            [0] * len(self.SAMPLES[0])
        ))

    def check_assign(self, dtype):
        rs = RunningStatTF()
        assign = rs.make_assign(
            tf.constant(10          , tf.int64),
            tf.constant([ 1,  2,  3], dtype   ),
            tf.constant([10, 20, 30], dtype   )
        )

        self.session.run([tf.global_variables_initializer()])

        self.assert_rs_shape(rs, dtype, (3, ))

        self.assert_rs(rs, 0, [0, 0, 0], [0, 0, 0])

        self.session.run([assign])

        self.assert_rs(rs, 10, [1, 2, 3], [10, 20, 30])

    def check_push(self, dtype):
        rs = RunningStatTF()
        x = tf.placeholder(dtype, shape=(2, ))
        push = rs.make_push(x)

        self.session.run([tf.global_variables_initializer()])

        self.assert_rs_shape(rs, dtype, (2, ))

        self.assert_rs_n(rs, 0)
        for i in xrange(len(self.SAMPLES)):
            self.push(push, x, i)
            self.assert_rs_n(rs, i + 1)

    def check_push_block(self, dtype):
        for step in xrange(1, len(self.SAMPLES) + 1):
            rs = RunningStatTF()
            n        = tf.placeholder(tf.int64             )
            mean     = tf.placeholder(dtype   , shape=(2, ))
            variance = tf.placeholder(dtype   , shape=(2, ))
            push_block = rs.make_push_block(n, mean, variance)

            self.session.run([tf.global_variables_initializer()])

            self.assert_rs_shape(rs, dtype, (2, ))

            self.assert_rs_n(rs, 0)
            for i in xrange(0, len(self.SAMPLES) - step + 1, step):
                self.push_block(push_block, n, mean, variance, i, i + step)
                self.assert_rs_n(rs, i + step)

    def check_push_push_block(self, dtype):
        rs = RunningStatTF()

        n        = tf.placeholder(tf.int64             )
        mean     = tf.placeholder(dtype   , shape=(2, ))
        variance = tf.placeholder(dtype   , shape=(2, ))
        push_block = rs.make_push_block(n, mean, variance)

        x = tf.placeholder(dtype, shape=(2, ))
        push = rs.make_push(x)

        self.session.run([tf.global_variables_initializer()])

        self.assert_rs_shape(rs, dtype, (2, ))

        self.assert_rs_n(rs, 0)

        self.push_block(push_block, n, mean, variance, 0, 2)
        self.assert_rs_n(rs, 2)

        self.push(push, x, 2)
        self.assert_rs_n(rs, 3)

        self.push_block(push_block, n, mean, variance, 3, 6)
        self.assert_rs_n(rs, 6)


class TestZFilterTF(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_(self):
        pass


if __name__ == '__main__':
    unittest.main()
