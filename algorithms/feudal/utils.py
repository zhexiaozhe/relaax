import numpy as np


class RingBuffer2D:
    """ A 2D ring buffer using numpy arrays """
    def __init__(self, element_size, buffer_size):
        self.data = np.zeros((buffer_size, element_size), dtype=np.float32)

    def extend(self, array_to_add):
        """ Adds array of element_size to buffer's end removing the beginning """
        self.data = np.vstack((self.data[1:, :], array_to_add))

    def get_sum(self):
        """ Returns the sum of elements within the rows of last half of the buffer """
        # TODO: get last cfg.c instead of split
        _, second = np.split(self.data, 2)
        return np.sum(second, axis=0)

    def get_diff(self):
        """ Returns the difference between the two halfs of data """
        first, second = np.split(self.data, 2)
        return second - first

    def replace_first_half(self, array_to_replace):
        """ Replaces the buffer's hals from beginning by new array """
        _, second = np.split(self.data, 2)
        self.data = np.vstack((array_to_replace, second))

    def reset(self):
        """ Resets buffer's data -> sets all elements to zeros """
        self.data.fill(0)
