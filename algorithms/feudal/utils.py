import numpy as np


class RingBuffer2D:
    """ A 2D ring buffer using numpy arrays """
    def __init__(self, element_size, buffer_size):
        self.data = np.zeros((buffer_size, element_size), dtype=np.float32)

    def extend(self, array_to_add):
        """ Adds array of element_size to buffer's end removing the beginning """
        self.data = np.vstack((self.data[1:, :], array_to_add))

    def get_sum(self):
        """ Returns the sum of elements within the rows of buffer """
        return np.sum(self.data, axis=0)

    def reset(self):
        """ Resets buffer's data -> sets all elements to zeros """
        self.data.fill(0)
