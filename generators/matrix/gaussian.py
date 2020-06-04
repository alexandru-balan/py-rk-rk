import threading

import numpy as npy

THREAD_NUMBER = 8


class __NumberGeneratingThread(threading.Thread):
    """
    This class is extending on the threading.Thread class and is used by this
    package to generate random numbers from the gaussian distribution.

    Attributes:
    -----------
    - numbers: [] -- An array for storing the generated numbers.
    - mean: float -- A float representing the median value of the distribution
    - sigma: float -- The deviation
    - sample_size -- How many numbers you want to generate
    - nb_rows -- The number of rows to generate
    """

    def __init__(self, mean, sigma, sample_size, nb_rows):
        threading.Thread.__init__(self)
        self.numbers = []
        self.mean = mean
        self.sigma = sigma
        self.sample_size = sample_size
        self.nb_rows = nb_rows

    def run(self) -> None:
        for i in range(self.nb_rows):
            self.numbers.append(npy.random.normal(self.mean, self.sigma, self.sample_size))


def get_mat_from_normal_dist(rows: int, columns: int, mu: float = 0.0, sigma: float = 1.0) -> npy.array:
    """
    Creat a matrix populated with values from the normal distribution and return it.
    The method uses a multi-threaded paradigm for performance.

    :param rows: int
        The number of rows for the returned matrix
    :param columns: int
        The number of columns for the returned matrix
    :param mu: float
        The mean of the distribution
    :param sigma: float
        The deviation

    :return: npy.array populated with values from the normal distribution
    """
    matrix = npy.empty((rows, columns))

    threads = []

    # Creating a bunch of threads that generate numbers and starting them
    for i in range(0, rows, THREAD_NUMBER):
        if i <= rows - THREAD_NUMBER:
            thread = __NumberGeneratingThread(mu, sigma, columns, THREAD_NUMBER)
        else:
            thread = __NumberGeneratingThread(mu, sigma, columns, rows - i)
        threads.append(thread)
        thread.start()

    # Waiting for the threads to finish the generation of numbers
    for thr in threads:
        thr.join()

    # After they finished we get the generated values, pack them into an array and return it
    row_num = 0

    for thread in threads:
        for row in thread.numbers:
            matrix[row_num] = row
            row_num += 1

    return matrix
