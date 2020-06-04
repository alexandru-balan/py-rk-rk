import threading

import numpy as npy


class __ProbabilityThread(threading.Thread):
    def __init__(self, frobenius: float, rows: npy.ndarray):
        threading.Thread.__init__(self)
        self.frobenius = frobenius
        self.rows = rows
        self.probability = []
        self.num_rows = len(rows)

    def run(self) -> None:
        for i in range(self.num_rows):
            self.probability.append(self.__compute_row_probability(self.rows[i]))

    def __compute_row_probability(self, row: []) -> float:
        """
        This method is used to compute the probability of selecting a row as described in
        the RK-RK algorithm

        :returns: probability = squared euclidean of row / squared frobenius
        """
        euclidean = pow(npy.linalg.norm(row, 2), 2)
        return euclidean / self.frobenius


def __choose_row(probs: []) -> int:
    """
    This method chooses a random row number from a bunch of rows depending
    on the probability of each row. Cool.

    :param probs: The probabilities of each row
    :returns: The number of a row
    """
    choices = range(len(probs))
    return npy.random.choice(
        choices,
        1,
        p=probs
    )
