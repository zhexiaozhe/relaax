import numpy as np


class RingBuffer:
    """ A 1D ring buffer using numpy arrays """
    def __init__(self, element_size, buffer_size):
        self.el_size = element_size
        self.data = np.zeros(self.el_size * buffer_size, dtype=np.float32)

    def extend(self, array_to_add):
        """ Adds array of element_size to buffer's end removing beginning """
        self.data = np.concatenate((self.data[self.el_size:], array_to_add), axis=0)

    def get_sum(self):
        """ Returns the sum of all elements within the buffer """
        return np.sum(self.data)

    def reset(self):
        """ Resets buffer's data -> sets all elements to zeros """
        self.data.fill(0)
